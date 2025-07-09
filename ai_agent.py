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

from src.board import Board
from src.piece import *
from src.move import Move
from src.square import Square
from src.const import *

class XiangqiEnvironment:
    """Environment wrapper for Xiangqi game"""
    
    def __init__(self):
        self.board = Board()
        self.current_player = 'red'
        self.game_over = False
        self.winner = None
        self.move_count = 0
        self.max_moves = 1000  # Maximum moves before draw
        
    def reset(self):
        """Reset the game to initial state"""
        self.board = Board()
        self.current_player = 'red'
        self.game_over = False
        self.winner = None
        self.move_count = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Convert board state to neural network input"""
        # Create 10x9x14 tensor (10 rows, 9 cols, 14 channels for different pieces)
        state = np.zeros((10, 9, 14), dtype=np.float32)
        
        piece_to_channel = {
            'general': 0, 'advisor': 1, 'elephant': 2, 'horse': 3,
            'chariot': 4, 'cannon': 5, 'pawn': 6
        }
        
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board.squares[row][col].piece
                if piece is not None:
                    channel = piece_to_channel[piece.name]
                    if piece.color == 'red':
                        state[row, col, channel] = 1.0
                    else:
                        state[row, col, channel + 7] = 1.0
        
        return state.flatten()
    
    def get_legal_moves(self, color: str) -> List[Move]:
        """Get all legal moves for a color"""
        legal_moves = []
        
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board.squares[row][col].piece
                if piece is not None and piece.color == color:
                    self.board.cal_move(piece, row, col)
                    legal_moves.extend(piece.moves_empty + piece.moves_rival)
                    piece.clear_moves()
        
        return legal_moves
    
    def make_move(self, move: Move, cur_move):
        """Make a move and return new state, reward, and done flag"""
        if not self.is_valid_move(move):
            return self.get_state(), -10, True  # Invalid move penalty
        
        # Check if move captures a piece
        captured_piece = self.board.squares[move.final.row][move.final.col].piece
        
        # Make the move
        piece = self.board.squares[move.initial.row][move.initial.col].piece
        self.board.move(piece, move)
        
        self.move_count += 1
        
        # Calculate reward
        reward = self.calculate_reward(captured_piece, cur_move)
        
        # Check for game over conditions
        done = self.check_game_over()
        
        # Switch player
        self.current_player = 'black' if self.current_player == 'red' else 'red'
        
        return self.get_state(), reward, done
    
    def is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid"""
        piece = self.board.squares[move.initial.row][move.initial.col].piece
        if piece is None or piece.color != self.current_player:
            return False
        
        self.board.cal_move(piece, move.initial.row, move.initial.col)
        valid = move in piece.moves_empty or move in piece.moves_rival
        piece.clear_moves()
        
        return valid
    
    def calculate_reward(self, captured_piece, cur_move) -> float:
        """Calculate reward for the current move"""
        reward = 0
        
        # Reward for capturing pieces
        if captured_piece is not None:
            piece_values = {
                'general': 1000, 'advisor': 20, 'elephant': 20, 'horse': 40,
                'chariot': 90, 'cannon': 50, 'pawn': 10
            }
            reward += piece_values.get(captured_piece.name, 0)
        
        # Small reward for making a move (to encourage activity)
        reward += 0.1
        
        # Check if opponent is in checkmate after this move
        if cur_move < 200:
            opponent = 'black' if self.current_player == 'red' else 'red'
            if self.is_checkmate(opponent):
                reward += 1000
            elif self.is_in_check(opponent):
                reward += 1  # Reward for putting opponent in check
        elif 200 <= cur_move < 1000:
            opponent = 'black' if self.current_player == 'red' else 'red'
            if self.is_checkmate(opponent):
                reward += 1000
            elif self.is_in_check(opponent):
                reward += 0 # Reward for putting opponent in check
            else:
                reward -= 0.2
        return reward
    
    def check_game_over(self) -> bool:
        """Check if the game is over"""
        # Check for maximum moves (draw)
        if self.move_count >= self.max_moves:
            self.game_over = True
            self.winner = None  # Draw
            return True
        
        # Check if current player (who just moved) has won
        opponent = 'black' if self.current_player == 'red' else 'red'
        
        if self.is_checkmate(opponent):
            self.game_over = True
            self.winner = self.current_player  # Current player wins
            return True
        
        # Check for stalemate (no legal moves but not in check)
        if len(self.get_legal_moves(opponent)) == 0:
            if not self.is_in_check(opponent):
                self.game_over = True
                self.winner = None  # Draw due to stalemate
                return True
        
        return False
    
    def is_checkmate(self, color: str) -> bool:
        """Check if a color is in checkmate"""
        if not self.is_in_check(color):
            return False
        
        # Try all possible moves to see if any can escape check
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board.squares[row][col].piece
                if piece is not None and piece.color == color:
                    self.board.cal_move(piece, row, col)
                    
                    # Try each possible move
                    for move in piece.moves_empty + piece.moves_rival:
                        # Make a copy of the board and try the move
                        temp_board = copy.deepcopy(self.board)
                        temp_piece = temp_board.squares[move.initial.row][move.initial.col].piece
                        temp_board.move(temp_piece, move)
                        
                        # Check if this move gets out of check
                        if not self.is_in_check_on_board(temp_board, color):
                            piece.clear_moves()
                            return False  # Found a move that escapes check
                    
                    piece.clear_moves()
        
        return True  # No moves can escape check
    
    def is_in_check(self, color: str) -> bool:
        """Check if a color's general is in check"""
        return self.is_in_check_on_board(self.board, color)
    
    def is_in_check_on_board(self, board: Board, color: str) -> bool:
        """Check if a color's general is in check on a given board"""
        # Find general position
        general_pos = None
        for row in range(ROWS):
            for col in range(COLS):
                piece = board.squares[row][col].piece
                if isinstance(piece, General) and piece.color == color:
                    general_pos = (row, col)
                    break
        
        if general_pos is None:
            return True  # General not found (captured)
        
        # Check if any opponent piece can attack the general
        opponent = 'black' if color == 'red' else 'red'
        for row in range(ROWS):
            for col in range(COLS):
                piece = board.squares[row][col].piece
                if piece is not None and piece.color == opponent:
                    board.cal_move(piece, row, col, bool=False)
                    for move in piece.moves_rival:
                        if move.final.row == general_pos[0] and move.final.col == general_pos[1]:
                            piece.clear_moves()
                            return True
                    piece.clear_moves()
        
        return False


class DQN(nn.Module):
    """Deep Q-Network for Xiangqi"""
    
    def __init__(self, input_size: int = 10*9*14, hidden_size: int = 512, output_size: int = 10*9*10*9):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


class XiangqiAgent:
    """DQN Agent for Xiangqi"""
    
    def __init__(self, color: str, learning_rate: float = 0.001, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.9998, epsilon_min: float = 0.3, 
                 memory_size: int = 10000, batch_size: int = 32):
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
        
    def train(self, episodes: int, save_freq: int = 100, update_target_freq: int = 10):
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
    trainer.train(episodes=50)
    
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