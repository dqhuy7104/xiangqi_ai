import pygame

from src.const import *
from src.board import Board
from src.dragger import Dragger
from src.sound import Sound

class Game:
    def __init__(self):
        self.margin = 50  # top & bottom margin
        self.border = 50 # left & right border
        self.sqsize_x = (WIDTH - 2 * self.border) // (COLS - 1)
        self.sqsize_y = (HEIGHT - 2 * self.margin) // (ROWS - 1)
        self.board = Board()
        self.dragger = Dragger()
        self.sound = Sound()
        self.next_player = 'red'
    
    # Render method
    def show_board(self, surface):
        surface.fill(B_YELLOW)

        # Draw horizontal lines
        for row in range(ROWS):
            y = self.margin + row * self.sqsize_y
            pygame.draw.line(surface, B_BLACK, (self.border, y), (WIDTH - self.border, y), 4)

        for col in range(COLS):
            x = self.border + col * self.sqsize_x
            if col == 0 or col == COLS - 1:
                # Vẽ đường thẳng từ trên xuống dưới
                pygame.draw.line(surface, B_BLACK, (x, self.margin), (x, self.margin + (ROWS - 1) * self.sqsize_y), 4)
            else:
                # Vẽ từ trên đến hết hàng 4
                pygame.draw.line(surface, B_BLACK, (x, self.margin), (x, self.margin + 4 * self.sqsize_y), 4)
                # Vẽ từ hàng 5 đến hàng 9
                pygame.draw.line(surface, B_BLACK, (x, self.margin + 5 * self.sqsize_y), (x, self.margin + 9 * self.sqsize_y), 4)

        # Palace Diagonals (Top)
        cx = self.border + 4 * self.sqsize_x
        y1 = self.margin
        y3 = self.margin + 2 * self.sqsize_y
        pygame.draw.line(surface, B_BLACK, (cx - self.sqsize_x, y1), (cx + self.sqsize_x, y3), 4)
        pygame.draw.line(surface, B_BLACK, (cx + self.sqsize_x, y1), (cx - self.sqsize_x, y3), 4)

        # Palace Diagonals (Bottom)
        y7 = self.margin + 7 * self.sqsize_y
        y9 = self.margin + 9 * self.sqsize_y
        pygame.draw.line(surface, B_BLACK, (cx - self.sqsize_x, y7), (cx + self.sqsize_x, y9), 4)
        pygame.draw.line(surface, B_BLACK, (cx + self.sqsize_x, y7), (cx - self.sqsize_x, y9), 4)

    def show_pieces(self, surface):
        for row in range(ROWS):
            for col in range(COLS):
                square = self.board.squares[row][col]
                if square.has_piece():
                    piece = square.piece

                    # Load hình ảnh quân cờ
                    img = pygame.image.load(piece.texture).convert_alpha()
                    img = pygame.transform.scale(img, (70, 70))

                    # Tính vị trí để vẽ (center alignment)
                    x = self.border + col * self.sqsize_x
                    y = self.margin + row * self.sqsize_y
                    piece.texture_rect = img.get_rect(center=(x, y))

                    # Vẽ quân cờ lên màn hình
                    surface.blit(img, piece.texture_rect)

    def show_moves(self, surface):
        if self.dragger.dragging:
            piece = self.dragger.piece

            for move in piece.moves_rival:
                color = "#A74949"
                center_x = self.border + move.final.col * self.sqsize_x
                center_y = self.margin + move.final.row * self.sqsize_y
                radius = 45  # bán kính 45 pixel
                pygame.draw.circle(surface, color, (center_x, center_y), radius)

            for move in piece.moves_empty:
                color = "#59A859"
                center_x = self.border + move.final.col * self.sqsize_x
                center_y = self.margin + move.final.row * self.sqsize_y
                radius = 25  # bán kính 25 pixel
                pygame.draw.circle(surface, color, (center_x, center_y), radius)

    def show_last_move(self, surface):
        if self.board.last_move:
            initial = self.board.last_move.initial
            final = self.board.last_move.final

            color = "#FFEE59" 
            center_x = self.border + initial.col * self.sqsize_x
            center_y = self.margin + initial.row * self.sqsize_y
            radius = 20  # bán kính 45 pixel
            pygame.draw.circle(surface, color, (center_x, center_y), radius)

            color = "#FFEE59" 
            center_x = self.border + final.col * self.sqsize_x
            center_y = self.margin + final.row * self.sqsize_y
            radius = 40  # bán kính 45 pixel
            pygame.draw.circle(surface, color, (center_x, center_y), radius)

    
    # Logic method
    def next_play(self):
        self.next_player = 'red' if self.next_player == 'black' else 'black'

    def play_sound(self, captured=False):
        if captured:
            self.sound.play_capture()
        else:
            self.sound.play_move()

    def reset(self):
        self.board = Board()
        self.dragger = Dragger()
        self.sound = Sound()
        self.next_player = 'red'


            