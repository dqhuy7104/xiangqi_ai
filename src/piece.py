import os

class Piece:
    def __init__(self, name, color, value, texture=None, texture_rect=None):
        self.name = name
        self.color = color
        self.value = value
        self.texture = self.set_texture()
        self.texture_rect = texture_rect
        self.moves_rival = []
        self.moves_empty = []
        self.moved = False
        self.alive = True


    def set_texture(self):
        return os.path.join(f'assets/{self.color}/{self.name}.png')

    def add_move_empty(self, move):
        self.moves_empty.append(move)

    def add_move_rival(self, move):
        self.moves_rival.append(move)

    def clear_moves(self):
        self.moves_rival = []
        self.moves_empty = []


class General(Piece):
    def __init__(self, color):
        super(). __init__('general', color, 10000)

class Advisor(Piece):
    def __init__(self, color):
        super(). __init__('advisor', color, 2)

class Elephant(Piece):
    def __init__(self, color):
        super(). __init__('elephant', color, 2)

class Horse(Piece):
    def __init__(self, color):
        super(). __init__('horse', color, 4)

class Chariot(Piece):
    def __init__(self, color):
        super(). __init__('chariot', color, 8)

class Cannon(Piece):
    def __init__(self, color):
        super(). __init__('cannon', color, 5)

class Pawn(Piece):
    def __init__(self, color):
        super(). __init__('pawn', color, 1)

    def get_value(self, row):
        # Tính giá trị hiện tại dựa vào hàng (row) của Square.
        if (self.color == "red" and row <= 4) or (self.color == "black" and row >= 5):
            return self.value * 2
        return self.value


