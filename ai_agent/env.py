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
        
        # Store the player making the move BEFORE switching
        moving_player = self.current_player
        
        # Check if move captures a piece
        captured_piece = self.board.squares[move.final.row][move.final.col].piece
        
        # Make the move
        piece = self.board.squares[move.initial.row][move.initial.col].piece
        self.board.move(piece, move)
        
        self.move_count += 1
        
        # Calculate reward for the player who just moved
        reward = self.calculate_reward(captured_piece, cur_move, moving_player)
        
        # Check for game over conditions BEFORE switching player
        done = self.check_game_over_for_player(moving_player)
        
        # Switch player AFTER everything else
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
    
    def calculate_reward(self, captured_piece, cur_move, moving_player) -> float:
        """Calculate reward for the moving player"""
        reward = 0
        
        # Reward for capturing pieces
        if captured_piece is not None:
            piece_values = {
                'general': 1000, 'advisor': 20, 'elephant': 20, 'horse': 40,
                'chariot': 90, 'cannon': 50, 'pawn': 10
            }
            reward += piece_values.get(captured_piece.name, 0)
        
        # Check if opponent is in checkmate after this move
        opponent = 'black' if moving_player == 'red' else 'red'
        
        if cur_move < 200:
            if self.is_checkmate(opponent):
                reward -= 1000  # reward for winning
            elif self.is_in_check(opponent):
                reward += 0.5  # Small reward for putting opponent in check
        elif 200 <= cur_move < 500:
            if self.is_checkmate(opponent):
                reward -= 1000  # reward for winning
            elif self.is_in_check(opponent):
                reward += 0  # No reward for check in late game
            else:
                reward -= 0.5  # Small penalty for taking long progress
        return reward
    
    def check_game_over_for_player(self, moving_player) -> bool:
        """Check if the game is over after a player's move"""
        # Check for maximum moves (draw)
        if self.move_count >= self.max_moves:
            self.game_over = True
            self.winner = None  # Draw
            return True
        
        # Check if the moving player has won (opponent in checkmate)
        opponent = 'black' if moving_player == 'red' else 'red'
        
        if self.is_checkmate(opponent):
            self.game_over = True
            self.winner = moving_player  # Moving player wins
            return True
        
        # Check for stalemate (no legal moves but not in check)
        if len(self.get_legal_moves(opponent)) == 0:
            if not self.is_in_check(opponent):
                self.game_over = True
                self.winner = None  # Draw due to stalemate
                return True
        
        return False
    
    def check_game_over(self) -> bool:
        """Check if the game is over - kept for backward compatibility"""
        return self.check_game_over_for_player(self.current_player)
    
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