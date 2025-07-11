import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque
import pickle
import os
import argparse
from typing import List, Tuple, Optional

from src.board import Board
from src.piece import *
from src.move import Move
from src.square import Square
from src.const import *
from ai_agent import XiangqiEnvironment
from ai_agent import DQN

parser = argparse.ArgumentParser(description="Choose hyperparameter")
parser.add_argument('-e', '--episodes', type=str, default=100)
parser.add_argument('-l', '--learning_rate', type=str, default=0.001)
parser.add_argument('-me', '--epsilon_min', type=str, default=0.01)
parser.add_argument('-d', '--epsilon_decay', type=str, default=0.9998)
parser.add_argument('-b', '--batch_size', type=str, default=32)
args = parser.parse_args()

class XiangqiAgent:
    """DQN Agent for Xiangqi"""
    
    def __init__(self, color: str, learning_rate: float = args.learning_rate, epsilon: float = 1.0, 
                 epsilon_decay: float = args.epsilon_decay, epsilon_min: float = args.epsilon_min, 
                 memory_size: int = 10000, batch_size: int = args.batch_size):
        self.color = color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Training parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.gamma = 0.95  # Discount factor
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)

        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action_index, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action_index, reward, next_state, done))
    
    def act(self, state, legal_moves: List[Move]) -> Move:
        """Choose action using epsilon-greedy policy"""
        if len(legal_moves) == 0:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Get Q-values for all possible actions
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        # Convert legal moves to action indices and find best action
        best_q_value = float('-inf')
        best_move = None
        
        for move in legal_moves:
            action_index = self.move_to_action_index(move)
            q_value = q_values[0][action_index].item()
            
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
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array([experience[0] for experience in batch])).float().to(self.device)
        actions = torch.tensor([experience[1] for experience in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([experience[2] for experience in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array([experience[3] for experience in batch])).float().to(self.device)
        dones = torch.BoolTensor([experience[4] for experience in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'color': self.color
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class XiangqiTrainer:
    """Trainer for Xiangqi AI agents"""
    
    def __init__(self):
        self.env = XiangqiEnvironment()
        self.red_agent = XiangqiAgent('red')
        self.black_agent = XiangqiAgent('black')
        
        # Training statistics
        self.episode_rewards = {'red': [], 'black': []}
        self.win_rates = {'red': 0, 'black': 0, 'draw': 0}
        
    def train(self, episodes: int = args.episodes, save_freq: int = 100, update_target_freq: int = 10):
        """Train the agents by playing against each other"""
        
        for episode in range(episodes):
            state = self.env.reset()
            total_rewards = {'red': 0, 'black': 0}
            move_count = 0
            max_moves_per_game = 1000  # Prevent infinite games
            
            while not self.env.game_over and move_count < max_moves_per_game:
                current_agent = self.red_agent if self.env.current_player == 'red' else self.black_agent
                
                # Get legal moves
                legal_moves = self.env.get_legal_moves(self.env.current_player)
                
                if len(legal_moves) == 0:
                    # No legal moves available - game should end
                    self.env.game_over = True
                    if self.env.is_checkmate(self.env.current_player):
                        # Checkmate - opponent wins
                        self.env.winner = 'black' if self.env.current_player == 'red' else 'red'
                    else:
                        # Stalemate - draw
                        self.env.winner = None
                    break

                # Agent chooses action
                action = current_agent.act(state, legal_moves)
                
                if action is None:
                    break
                
                # Make move
                next_state, reward, done = self.env.make_move(action, move_count)
                
                # Store experience
                action_index = current_agent.move_to_action_index(action)
                current_agent.remember(state, action_index, reward, next_state, done)
                
                total_rewards[self.env.current_player] += reward
                state = next_state
                move_count += 1

                if not self.env.winner:
                    total_rewards['red'] -= 200
                    total_rewards['black'] -= 200

                # Train the agent
                if len(current_agent.memory) > current_agent.batch_size:
                    current_agent.replay()
            
            # Update statistics
            self.episode_rewards['red'].append(total_rewards['red'])
            self.episode_rewards['black'].append(total_rewards['black'])
            
            if self.env.winner == 'red':
                self.win_rates['red'] += 1
            elif self.env.winner == 'black':
                self.win_rates['black'] += 1
            else:
                self.win_rates['draw'] += 1
            
            # Update target networks
            if episode % update_target_freq == 0:
                self.red_agent.update_target_network()
                self.black_agent.update_target_network()
            
            # Save models
            if episode % save_freq == 0 and episode > 0:
                self.save_models(f'models/episode_{episode}')
            
            # Print progress
            total_games = episode + 1
            red_rate = self.win_rates['red'] / total_games * 100
            black_rate = self.win_rates['black'] / total_games * 100
            draw_rate = self.win_rates['draw'] / total_games * 100
            
            print(f"Episode {episode}:")
            print(f"  Red wins: {red_rate:.1f}%")
            print(f"  Black wins: {black_rate:.1f}%")
            print(f"  Draws: {draw_rate:.1f}%")
            print(f"  Red epsilon: {self.red_agent.epsilon:.3f}")
            print(f"  Black epsilon: {self.black_agent.epsilon:.3f}")
            print(f"  Avg moves: {move_count}")
            print('_________________________')

    def save_models(self, directory: str):
        """Save both agent models"""
        os.makedirs(directory, exist_ok=True)
        self.red_agent.save_model(f'{directory}/red_agent.pth')
        self.black_agent.save_model(f'{directory}/black_agent.pth')
        
        # Save training statistics
        with open(f'{directory}/training_stats.pkl', 'wb') as f:
            pickle.dump({
                'episode_rewards': self.episode_rewards,
                'win_rates': self.win_rates
            }, f)
    
    def load_models(self, directory: str):
        """Load both agent models"""
        self.red_agent.load_model(f'{directory}/red_agent.pth')
        self.black_agent.load_model(f'{directory}/black_agent.pth')
    
    def play_game(self, display: bool = False) -> str:
        """Play a single game between the agents"""
        state = self.env.reset()
        move_count = 0
        max_moves = 1000
        
        while not self.env.game_over and move_count < max_moves:
            current_agent = self.red_agent if self.env.current_player == 'red' else self.black_agent
            
            # Get legal moves
            legal_moves = self.env.get_legal_moves(self.env.current_player)
            
            if len(legal_moves) == 0:
                if self.env.is_in_check(self.env.current_player):
                    # Checkmate
                    self.env.winner = 'black' if self.env.current_player == 'red' else 'red'
                else:
                    # Stalemate
                    self.env.winner = None
                break
            
            # Agent chooses action (with epsilon=0 for pure exploitation)
            original_epsilon = current_agent.epsilon
            current_agent.epsilon = 0
            action = current_agent.act(state, legal_moves)
            current_agent.epsilon = original_epsilon
            
            if action is None:
                break
            
            # Make move
            next_state, reward, done = self.env.make_move(action, move_count)
            state = next_state
            move_count += 1
            
            if display:
                print(f"{self.env.current_player} plays: {action.initial.row},{action.initial.col} -> {action.final.row},{action.final.col}")
        
        return self.env.winner if self.env.winner else "draw"
    

# Example usage
if __name__ == "__main__":
    # Create trainer
    trainer = XiangqiTrainer()
    
    # Train the agents
    print("Starting training...")
    trainer.train()
    
    # Save final models
    trainer.save_models('models/final')
    
    # Play some test games
    print("\nTesting trained agents...")
    results = {'red': 0, 'black': 0, 'draw': 0}
    
    for i in range(10):
        result = trainer.play_game()
        results[result] += 1
    
    print(f"Test results over 10 games:")
    print(f"Red wins: {results['red']}")
    print(f"Black wins: {results['black']}")
    print(f"Draws: {results['draw']}")
    
