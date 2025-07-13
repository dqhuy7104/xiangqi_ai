import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque
import pickle
import os
from typing import List, Tuple, Optional
import argparse

from src.board import Board
from src.piece import *
from src.move import Move
from src.square import Square
from src.const import *

class DRQN(nn.Module):
    """Deep Recurrent Q-Network for Xiangqi"""
    
    def __init__(self, input_size: int = 10*9*14, hidden_size: int = 512, 
                 lstm_hidden_size: int = 256, output_size: int = 10*9*10*9, 
                 num_lstm_layers: int = 2):
        super(DRQN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size
        self.num_lstm_layers = num_lstm_layers
        
        # Input processing layers
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # LSTM layers for sequential processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass through DRQN
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden_state: Previous LSTM hidden state (optional)
        Returns:
            output: Q-values of shape (batch_size, sequence_length, output_size)
            hidden_state: New LSTM hidden state
        """
        batch_size, seq_length, _ = x.shape
        
        # Process input through initial layers
        x = x.view(-1, self.input_size)  # Flatten for linear layer
        x = self.input_layer(x)
        x = x.view(batch_size, seq_length, self.hidden_size)  # Reshape for LSTM
        
        # LSTM processing
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        # Output processing
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_hidden_size)
        output = self.output_layer(lstm_out)
        output = output.view(batch_size, seq_length, self.output_size)
        
        return output, hidden_state
    
    def init_hidden_state(self, batch_size, device):
        """Initialize hidden state for LSTM"""
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)

class XiangqiAgent:
    """DRQN Agent for Xiangqi"""
    
    def __init__(self, color: str, learning_rate: float,
                 epsilon_decay: float, epsilon_min: float, batch_size: int,
                 memory_size: int = 10000, epsilon: float = 1.0,
                 sequence_length: int = 10):
        self.color = color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        
        # Neural networks
        self.q_network = DRQN().to(self.device)
        self.target_network = DRQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.gamma = 0.95  # Discount factor
        
        # Experience replay - now stores sequences
        self.memory = deque(maxlen=memory_size)
        
        # Current episode sequence
        self.current_sequence = []
        
        # Hidden states for online network
        self.hidden_state = None
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def reset_hidden_state(self):
        """Reset hidden state at the beginning of each episode"""
        self.hidden_state = None
        self.current_sequence = []
    
    def remember(self, state, action_index, reward, next_state, done):
        """Store experience in current sequence"""
        self.current_sequence.append((state, action_index, reward, next_state, done))
        
        # If episode is done or sequence reaches max length, store in memory
        if done or len(self.current_sequence) >= self.sequence_length:
            if len(self.current_sequence) > 0:
                self.memory.append(copy.deepcopy(self.current_sequence))
            self.current_sequence = []
    
    def act(self, state, legal_moves: List[Move]) -> Move:
        """Choose action using epsilon-greedy policy with DRQN"""
        if len(legal_moves) == 0:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Prepare state for DRQN (add sequence dimension)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Initialize hidden state if needed
        if self.hidden_state is None:
            self.hidden_state = self.q_network.init_hidden_state(1, self.device)
        
        # Get Q-values from DRQN
        with torch.no_grad():
            q_values, self.hidden_state = self.q_network(state_tensor, self.hidden_state)
            q_values = q_values.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions
        
        # Convert legal moves to action indices and find best action
        best_q_value = float('-inf')
        best_move = None
        
        for move in legal_moves:
            action_index = self.move_to_action_index(move)
            q_value = q_values[action_index].item()
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = move
        
        return best_move if best_move else random.choice(legal_moves)
    
    def move_to_action_index(self, move: Move) -> int:
        """Convert move to action index"""
        return move.initial.row * 9 * 10 * 9 + move.initial.col * 10 * 9 + move.final.row * 9 + move.final.col
    
    def action_index_to_move(self, action_index: int) -> Move:
        """Convert action index to move"""
        initial_row = action_index // (9 * 10 * 9)
        remainder = action_index % (9 * 10 * 9)
        initial_col = remainder // (10 * 9)
        remainder = remainder % (10 * 9)
        final_row = remainder // 9
        final_col = remainder % 9
        
        return Move(Square(initial_row, initial_col), Square(final_row, final_col))
    
    def replay(self):
        """Train the network on a batch of sequences using Double DQN"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch of sequences
        batch_sequences = random.sample(self.memory, self.batch_size)
        max_seq_len = max(len(seq) for seq in batch_sequences)

        # Prepare padded batches
        states, actions, rewards, next_states, dones, mask = [], [], [], [], [], []

        for sequence in batch_sequences:
            seq_len = len(sequence)
            pad_len = max_seq_len - seq_len

            seq_states, seq_actions, seq_rewards, seq_next_states, seq_dones = zip(*sequence)
            
            # Padding
            seq_states = list(seq_states) + [seq_states[-1]] * pad_len
            seq_actions = list(seq_actions) + [seq_actions[-1]] * pad_len
            seq_rewards = list(seq_rewards) + [0.0] * pad_len
            seq_next_states = list(seq_next_states) + [seq_next_states[-1]] * pad_len
            seq_dones = list(seq_dones) + [True] * pad_len

            states.append(seq_states)
            actions.append(seq_actions)
            rewards.append(seq_rewards)
            next_states.append(seq_next_states)
            dones.append(seq_dones)
            mask.append([1.0]*seq_len + [0.0]*pad_len)

        # Convert to tensors
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)

        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

        batch_size = states.shape[0]
        hidden_q = self.q_network.init_hidden_state(batch_size, self.device)
        hidden_target = self.target_network.init_hidden_state(batch_size, self.device)

        # Current Q-values
        current_q_values, _ = self.q_network(states, hidden_q)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        # ------------------ Double DQN here ------------------
        with torch.no_grad():
            # Online network chọn action
            next_q_online, _ = self.q_network(next_states, hidden_q)
            next_actions = torch.argmax(next_q_online, dim=2)  # shape: (batch, seq)

            # Target network đánh giá Q
            next_q_target, _ = self.target_network(next_states, hidden_target)
            next_q_values = next_q_target.gather(2, next_actions.unsqueeze(2)).squeeze(2)

        # Target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # MSE Loss with masking
        loss = nn.MSELoss(reduction='none')(current_q_values, target_q_values)
        loss = (loss * mask).sum() / mask.sum()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'color': self.color,
            'sequence_length': self.sequence_length
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        if 'sequence_length' in checkpoint:
            self.sequence_length = checkpoint['sequence_length']