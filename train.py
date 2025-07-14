import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque
import pickle
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Tuple, Optional

from src.board import Board
from src.piece import *
from src.move import Move
from src.square import Square
from src.const import *
from ai_agent import XiangqiEnvironment
from ai_agent import XiangqiAgent

parser = argparse.ArgumentParser(description="Choose hyperparameter")
parser.add_argument('-e', '--episodes', type=int, default=100)
parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
parser.add_argument('-me', '--epsilon_min', type=float, default=0.01)
parser.add_argument('-d', '--epsilon_decay', type=float, default=0.9998)
parser.add_argument('-b', '--batch_size', type=int, default=32)
args = parser.parse_args()

class XiangqiTrainer:
    """Trainer for Xiangqi AI agents"""
    
    def __init__(self):
        self.env = XiangqiEnvironment()
        self.red_agent = XiangqiAgent('red', args.learning_rate, args.epsilon_decay, args.epsilon_min, args.batch_size)
        self.black_agent = XiangqiAgent('black', args.learning_rate, args.epsilon_decay, args.epsilon_min, args.batch_size)
        
        # Training statistics
        self.episode_rewards = {'red': [], 'black': []}
        self.win_rates = {'red': 0, 'black': 0, 'draw': 0}
        
    def train(self, episodes: int = args.episodes, save_freq: int = 100, update_target_freq: int = 10):
        reward_history = []
        loss_history = []

        for episode in range(episodes):
            state = self.env.reset()
            total_rewards = {'red': 0, 'black': 0}
            losses = {'red': [], 'black': []}
            move_count = 0
            max_moves_per_game = 400

            while not self.env.game_over and move_count < max_moves_per_game:
                current_agent = self.red_agent if self.env.current_player == 'red' else self.black_agent
                legal_moves = self.env.get_legal_moves(self.env.current_player)

                if len(legal_moves) == 0:
                    self.env.game_over = True
                    if self.env.is_in_check(self.env.current_player):
                        self.env.winner = 'black' if self.env.current_player == 'red' else 'red'
                    else:
                        self.env.winner = None
                    break

                action = current_agent.act(state, legal_moves)
                if action is None:
                    break

                next_state, reward, done = self.env.make_move(action, move_count)

                action_index = current_agent.move_to_action_index(action)
                current_agent.remember(state, action_index, reward, next_state, done)

                total_rewards[self.env.current_player] += reward
                state = next_state
                move_count += 1

                if len(current_agent.memory) > current_agent.batch_size:
                    loss = current_agent.replay()
                    if loss is not None:
                        losses[self.env.current_player].append(loss)
            
            if move_count > max_moves_per_game:
                total_rewards['red'] -= 500
                total_rewards['black'] -= 500

            # Ghi nhận reward
            self.episode_rewards['red'].append(total_rewards['red'])
            self.episode_rewards['black'].append(total_rewards['black'])

            # Ghi nhận win-rate
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

            # Tính toán trung bình
            avg_red_loss = np.mean(losses['red']) if losses['red'] else 0
            avg_black_loss = np.mean(losses['black']) if losses['black'] else 0
            avg_reward = (total_rewards['red'] + total_rewards['black']) / 2

            reward_history.append(avg_reward)
            loss_history.append((avg_red_loss + avg_black_loss) / 2)

            # In thông tin
            total_games = episode + 1
            print(f"Episode {episode}:")
            print(f"  Red wins: {self.win_rates['red'] / total_games * 100:.1f}%")
            print(f"  Black wins: {self.win_rates['black'] / total_games * 100:.1f}%")
            print(f"  Draws: {self.win_rates['draw'] / total_games * 100:.1f}%")
            print(f"  Avg moves: {move_count}")
            print(f"  Red reward: {total_rewards['red']:.1f} | Red loss: {avg_red_loss:.4f}")
            print(f"  Black reward: {total_rewards['black']:.1f} | Black loss: {avg_black_loss:.4f}")
            print("_________________________")

        return reward_history, loss_history


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
    
    def play_game(self, display: bool = True) -> str:
        """Play a single game between the agents"""
        state = self.env.reset()
        move_count = 0
        max_moves = 400
        
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
                print(f"{self.env.current_player} plays: {action.initial.row},{action.initial.col} -> {action.final.row},{action.final.col}, reward: {reward}")
        
        return self.env.winner if self.env.winner else "draw"
    

# Example usage
if __name__ == "__main__":
    # Create trainer
    trainer = XiangqiTrainer()
    
    # Train the agents
    print("Starting training...")
    reward_history, loss_history = trainer.train()

    # Save final models
    trainer.save_models('models/final')

    print('Model saved at models/final')
    
    episodes = range(1, args.episodes + 1)
    plt.figure(figsize=(10, 5))

    # Vẽ Reward
    plt.plot(episodes, reward_history, label='Avg Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Vẽ Loss
    plt.plot(episodes, loss_history, label='Avg Loss', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

