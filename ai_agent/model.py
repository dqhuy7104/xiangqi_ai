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


