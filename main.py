# Import libraries
import pygame
import sys

from src.const import *
from src.game import Game
from src.square import Square
from src.move import Move

# Main class 
class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chinese Chess')
        self.game = Game()
        

    def main_loop(self):
        # Show board background
        game = self.game
        screen = self.screen
        board = self.game.board
        dragger = self.game.dragger

        while True:
            game.show_board(screen)
            game.show_moves(screen)
            game.show_last_move(screen)
            game.show_pieces(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            for event in pygame.event.get():

                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragger.update_mouse(event.pos)

                    clicked_row = dragger.mouseY // SQSIZEY
                    clicked_col = dragger.mouseX // SQSIZEX

                    if board.squares[clicked_row][clicked_col].has_piece():
                        piece = board.squares[clicked_row][clicked_col].piece
                        if piece.color == game.next_player:
                            board.cal_move(piece, clicked_row, clicked_col, bool=True)
                            dragger.save_initial(event.pos)
                            dragger.drag_piece(piece)
                            # Show board background
                            game.show_board(screen)
                            game.show_moves(screen)
                            game.show_pieces(screen)

                elif event.type == pygame.MOUSEMOTION:
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)

                        # Show board background
                        game.show_board(screen)
                        game.show_moves(screen)
                        game.show_pieces(screen)

                        dragger.update_blit(screen)

                elif event.type == pygame.MOUSEBUTTONUP:

                    if dragger.dragging:
                        dragger.update_mouse(event.pos)

                        released_row = dragger.mouseY // SQSIZEY
                        released_col = dragger.mouseX // SQSIZEX

                        # Create move
                        initial = Square(dragger.initial_row, dragger.initial_col)
                        final = Square(released_row, released_col)
                        move = Move(initial, final)

                        if board.valid_move(dragger.piece, move):
                            captured = board.squares[released_row][released_col].has_piece()
                            board.move(dragger.piece, move)

                            # sounds
                            game.play_sound(captured)

                            game.show_board(screen)
                            game.show_last_move(screen)
                            game.show_pieces(screen)
                            game.next_play()

                    dragger.undrag_piece()

                elif event.type == pygame.KEYDOWN:
                    # changing themes
                    if event.key == pygame.K_r:
                        game.reset()
                        game = self.game
                        board = self.game.board
                        dragger = self.game.dragger


                # Quit app     
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()

main = Main()
main.main_loop()
