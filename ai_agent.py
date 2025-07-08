import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle
import os
from typing import List, Tuple, Dict, Optional
import pygame
from src.board import Board
from src.piece import General
from src.square import Square
from src.move import Move

pygame.init()
pygame.mixer.init()

class XiangqiNet(nn.Module):
    """Neural network for xiangqi position evaluation"""
    def __init__(self, input_size=810, hidden_size=512):
        super(XiangqiNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)  # Single output for position value
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.tanh(x)  # Output between -1 and 1

class DQNAgent:
    """Deep Q-Network Agent for Xiangqi"""
    def __init__(self, input_size=810, lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = XiangqiNet(input_size).to(self.device)
        self.target_network = XiangqiNet(input_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.update_target_freq = 100
        self.steps = 0
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_state(self, board, current_player):
        """Convert board state to neural network input"""
        state = np.zeros((9, 10, 9), dtype=np.float32)
        
        piece_map = {
            'pawn': 0, 'cannon': 1, 'chariot': 2, 'horse': 3,
            'elephant': 4, 'advisor': 5, 'general': 6
        }
        
        # Fill the state tensor
        for row in range(10):
            for col in range(9):
                piece = board.squares[row][col].piece
                if piece:
                    piece_idx = piece_map.get(piece.name, 6)
                    color_multiplier = 1 if piece.color == 'red' else -1
                    state[piece_idx][row][col] = color_multiplier
        
        # Add current player layer
        current_player_val = 1 if current_player == 'red' else -1
        state[7] = np.full((10, 9), current_player_val)
        
        # Add game phase layer (opening, middle, endgame)
        piece_count = sum(1 for row in range(10) for col in range(9) 
                         if board.squares[row][col].piece)
        if piece_count > 20:
            phase = 1  # Opening
        elif piece_count > 10:
            phase = 0  # Middle
        else:
            phase = -1  # Endgame
        state[8] = np.full((10, 9), phase)
        
        return state.flatten()
    
    def get_valid_moves(self, board, color):
        """Get all valid moves for a given color"""
        valid_moves = []
        
        for row in range(10):
            for col in range(9):
                piece = board.squares[row][col].piece
                if piece and piece.color == color:
                    board.cal_move(piece, row, col)
                    for move in piece.moves_empty + piece.moves_rival:
                        valid_moves.append((row, col, move.final.row, move.final.col))
        
        return valid_moves
    
    def is_in_check(self, board, color):
        """Check if a player is in check"""
        general_pos = None
        for row in range(10):
            for col in range(9):
                piece = board.squares[row][col].piece
                if piece and isinstance(piece, General) and piece.color == color:
                    general_pos = (row, col)
                    break
        
        if not general_pos:
            return False
        
        opponent_color = 'black' if color == 'red' else 'red'
        for row in range(10):
            for col in range(9):
                piece = board.squares[row][col].piece
                if piece and piece.color == opponent_color:
                    board.cal_move(piece, row, col, bool=False)
                    for move in piece.moves_rival:
                        if move.final.row == general_pos[0] and move.final.col == general_pos[1]:
                            return True
        
        return False
    
    def act(self, state, valid_moves):
        """Choose action using epsilon-greedy policy"""
        if not valid_moves:
            return None
        
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        
        best_move = None
        best_value = float('-inf')
        
        for move in valid_moves:
            move_hash = move[0] * 1000 + move[1] * 100 + move[2] * 10 + move[3]
            q_value = self.q_network(state_tensor).item()
            move_score = q_value + random.uniform(-0.1, 0.1)
            
            if move_score > best_value:
                best_value = move_score
                best_move = move
        
        return best_move
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).squeeze()
        next_q_values = self.target_network(next_states).squeeze()
        
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.update_target_network()
    
    def save(self, filename):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
    
    def load(self, filename):
        """Load the model"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']

class XiangqiTrainer:
    """Training system for two AI agents"""
    def __init__(self):
        self.agent1 = DQNAgent()
        self.agent2 = DQNAgent()
        self.training_stats = {
            'games_played': 0,
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'avg_game_length': 0
        }
    
    def is_game_over(self, board, current_player):
        """Check if the game is over"""
        # Check if generals are still present
        red_general = False
        black_general = False
        for row in range(10):
            for col in range(9):
                piece = board.squares[row][col].piece
                if piece and isinstance(piece, General):
                    if piece.color == 'red':
                        red_general = True
                    else:
                        black_general = True
        
        if not red_general or not black_general:
            return True
        
        # Check if generals are facing each other
        if board.general_facing_each_other():
            return True
        
        # Check if current player has no valid moves
        valid_moves = self.agent1.get_valid_moves(board, current_player)
        if not valid_moves:
            return True
        
        return False
    
    def calculate_reward(self, board, captured_piece, player_color, done, opponent_no_moves):
        """Calculate reward for the move"""
        reward = 0
        
        piece_values = {
            'pawn': 1, 'cannon': 4, 'chariot': 9, 'horse': 4,
            'elephant': 2, 'advisor': 2, 'general': 1000
        }
        
        if captured_piece:
            reward += piece_values.get(captured_piece.name, 0)
            if captured_piece.name == 'general':
                reward += 1000
        
        if done and opponent_no_moves:
            reward += 1000  # Win bonus for opponent having no moves
        
        reward -= 0.1  # Small penalty for each move
        
        if self.agent1.is_in_check(board, player_color):
            reward -= 5
        
        opponent_color = 'black' if player_color == 'red' else 'red'
        if self.agent1.is_in_check(board, opponent_color):
            reward += 5
        
        return reward
    
    def train(self, num_episodes=1000, save_freq=100):
        """Train two agents by playing against each other"""
        print("Starting training...")
        
        for episode in range(num_episodes):
            board = Board()
            current_player = 'red'
            state = self.agent1.get_state(board, current_player)
            
            game_length = 0
            max_moves = 200
            
            while game_length < max_moves:
                agent = self.agent1 if current_player == 'red' else self.agent2
                valid_moves = agent.get_valid_moves(board, current_player)
                
                if not valid_moves:
                    break
                
                action = agent.act(state, valid_moves)
                
                if action is None:
                    break
                
                from_row, from_col, to_row, to_col = action
                piece = board.squares[from_row][from_col].piece
                if not piece:
                    state, reward, done = state, -10, True
                else:
                    initial = Square(from_row, from_col)
                    final_piece = board.squares[to_row][to_col].piece
                    final = Square(to_row, to_col, final_piece)
                    move_obj = Move(initial, final)
                    
                    if not board.valid_move(piece, move_obj):
                        state, reward, done = state, -10, True
                    else:
                        captured_piece = board.squares[to_row][to_col].piece
                        board.move(piece, move_obj)
                        
                        next_state = agent.get_state(board, current_player)
                        opponent_color = 'black' if current_player == 'red' else 'red'
                        opponent_moves = agent.get_valid_moves(board, opponent_color)
                        done = self.is_game_over(board, opponent_color)
                        reward = self.calculate_reward(board, captured_piece, current_player, done, not opponent_moves)
                        
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        game_length += 1
                        
                        if len(agent.memory) > agent.batch_size:
                            agent.replay()
                        
                        current_player = opponent_color
                
                if done:
                    break
            
            self.update_stats(board, game_length, current_player)
            
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}")
                print(f"Agent 1 wins: {self.training_stats['agent1_wins']}")
                print(f"Agent 2 wins: {self.training_stats['agent2_wins']}")
                print(f"Draws: {self.training_stats['draws']}")
                print(f"Average game length: {self.training_stats['avg_game_length']:.2f}")
                print(f"Agent 1 epsilon: {self.agent1.epsilon:.3f}")
                print(f"Agent 2 epsilon: {self.agent2.epsilon:.3f}")
                print("-" * 50)
            
            if episode % save_freq == 0:
                self.save_models(f"models/episode_{episode}")
        
        print("Training completed!")
        self.save_models("models/final")
    
    def update_stats(self, board, game_length, last_player):
        """Update training statistics"""
        self.training_stats['games_played'] += 1
        
        red_general = False
        black_general = False
        for row in range(10):
            for col in range(9):
                piece = board.squares[row][col].piece
                if piece and isinstance(piece, General):
                    if piece.color == 'red':
                        red_general = True
                    else:
                        black_general = True
        
        if not black_general or (not self.agent1.get_valid_moves(board, 'black') and last_player == 'red'):
            self.training_stats['agent1_wins'] += 1
        elif not red_general or (not self.agent1.get_valid_moves(board, 'red') and last_player == 'black'):
            self.training_stats['agent2_wins'] += 1
        else:
            self.training_stats['draws'] += 1
        
        total_length = (self.training_stats['avg_game_length'] * 
                        (self.training_stats['games_played'] - 1) + game_length)
        self.training_stats['avg_game_length'] = total_length / self.training_stats['games_played']
    
    def save_models(self, prefix):
        """Save both agent models"""
        os.makedirs("models", exist_ok=True)
        self.agent1.save(f"{prefix}_agent1.pth")
        self.agent2.save(f"{prefix}_agent2.pth")
        
        with open(f"{prefix}_stats.pkl", 'wb') as f:
            pickle.dump(self.training_stats, f)
    
    def load_models(self, prefix):
        """Load both agent models"""
        self.agent1.load(f"{prefix}_agent1.pth")
        self.agent2.load(f"{prefix}_agent2.pth")
        
        stats_file = f"{prefix}_stats.pkl"
        if os.path.exists(stats_file):
            with open(stats_file, 'rb') as f:
                self.training_stats = pickle.load(f)
    
    def play_game(self, human_vs_ai=False, human_color='red'):
        """Play a game between agents or human vs AI"""
        board = Board()
        current_player = 'red'
        state = self.agent1.get_state(board, current_player)
        
        game_length = 0
        max_moves = 200
        
        while game_length < max_moves:
            valid_moves = self.agent1.get_valid_moves(board, current_player)
            
            if not valid_moves:
                print(f"No valid moves for {current_player}. Game over!")
                break
            
            if human_vs_ai and current_player == human_color:
                print(f"Valid moves for {current_player}: {valid_moves[:5]}...")
                action = valid_moves[0]  # Placeholder for human input
            else:
                agent = self.agent1 if current_player == 'red' else self.agent2
                old_epsilon = agent.epsilon
                agent.epsilon = 0
                action = agent.act(state, valid_moves)
                agent.epsilon = old_epsilon
            
            if action is None:
                break
            
            from_row, from_col, to_row, to_col = action
            piece = board.squares[from_row][from_col].piece
            if not piece:
                print("Invalid move: No piece at position")
                break
            
            initial = Square(from_row, from_col)
            final_piece = board.squares[to_row][to_col].piece
            final = Square(to_row, to_col, final_piece)
            move = Move(initial, final)
            
            if not board.valid_move(piece, move):
                print("Invalid move")
                break
            
            board.move(piece, move)
            state = self.agent1.get_state(board, current_player)
            game_length += 1
            
            print(f"Move {game_length}: {current_player} plays {action}")
            
            if self.is_game_over(board, current_player):
                print("Game finished!")
                break
            
            current_player = 'black' if current_player == 'red' else 'red'
        
        return board

if __name__ == "__main__":
    trainer = XiangqiTrainer()
    
    try:
        trainer.load_models("models/final")
        print("Loaded existing models")
    except:
        print("No existing models found, starting fresh")
    
    trainer.train(num_episodes=1000, save_freq=100)
    
    print("\nPlaying demonstration game...")
    trainer.play_game()