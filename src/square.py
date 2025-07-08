from src.const import *

class Square:
    def __init__(self, row, col, piece=None):
        self.row = row
        self.col = col
        self.piece = piece

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def has_piece(self):
        return self.piece != None
    
    def has_team_piece(self, color):
        return self.has_piece() and self.piece.color == color
    
    def has_rival_piece(self, color):
        return self.has_piece() and self.piece.color != color

    def is_empty(self):
        return not self.has_piece()
    
    def isempty_or_rival(self, color):
        return self.is_empty() or self.has_rival_piece(color)
    
    @staticmethod
    def in_range_other(*args):
        for i, arg in enumerate(args):
            if i % 2 == 0:  # row
                if arg < 0 or arg > (ROWS-1):
                    return False
            else:  # col
                if arg < 0 or arg > (COLS-1):
                    return False
        return True
    
    @staticmethod
    def in_range_advisor_general(color, row, col):
        if 3 <= col <= 5:
            if color == "red" and 7 <= row <= 9 and 3 <= col <= 5:
                return True
            elif color == "black" and 0 <= row <= 2 and 3 <= col <= 5:
                return True
        return False

    
    @staticmethod
    def in_range_elephant(color, row):
        if color == "red" and 5 <= row <= 9:
            return True
        elif color == "black" and 0 <= row <= 4:
            return True
        return False
    

