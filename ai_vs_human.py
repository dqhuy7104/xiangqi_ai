# Import libraries
import pygame
import sys
import threading
import time
import argparse

from src.const import *
from src.game import Game
from src.square import Square
from src.move import Move
from ai_agent import XiangqiAgent, XiangqiEnvironment

parser = argparse.ArgumentParser(description="Người chơi chọn phe")

# Add argument
parser.add_argument('-a', '--ai', type=str, default='black', choices=['red', 'black'])
parser.add_argument('-p', '--player', type=str, default='red', choices=['red', 'black'])
args = parser.parse_args()

# Make sure ai and player have opposite side
args.player = 'black' if args.ai == 'red' else 'red'
args.ai = 'red' if args.player == 'black' else 'black'

# Main class 
class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chinese Chess - Human vs AI')
        self.game = Game()
        
        # AI setup
        self.ai_environment = XiangqiEnvironment()
        self.ai_agent = XiangqiAgent(args.ai, 0.001, 0.9998, 0.01, 32)  # AI plays as black
        self.human_color =  args.player  # Human plays as red
        
        # Try to load pre-trained model
        try:
            self.ai_agent.load_model('models/final/red_agent.pth')
            print("Loaded pre-trained AI model")
        except:
            print("No pre-trained model found, using random AI")
        
        # Set AI to exploitation mode (no random moves)
        self.ai_agent.epsilon = 0.1  # Small epsilon for occasional exploration
        
        # AI thinking indicator
        self.ai_thinking = False
        self.ai_move_delay = 1.0  # Delay to make AI moves visible
        
        # Game state synchronization
        self.pending_ai_move = None
        
    def sync_ai_environment(self):
        """Synchronize the AI environment with the game board"""
        self.ai_environment.board = self.game.board
        self.ai_environment.current_player = self.game.next_player
        self.ai_environment.move_count = 0  # Reset for simplicity
        self.ai_environment.game_over = False
        self.ai_environment.winner = None
        
    def ai_make_move(self):
        """AI makes a move in a separate thread"""
        if self.ai_thinking:
            return
            
        self.ai_thinking = True
        
        def ai_thread():
            time.sleep(self.ai_move_delay)  # Add delay for better UX
            
            # Sync environment
            self.sync_ai_environment()
            
            # Get AI's legal moves
            legal_moves = self.ai_environment.get_legal_moves(args.ai)
            
            if legal_moves:
                # AI chooses move
                state = self.ai_environment.get_state()
                ai_move = self.ai_agent.act(state, legal_moves)
                
                if ai_move:
                    self.pending_ai_move = ai_move
            
            self.ai_thinking = False
        
        thread = threading.Thread(target=ai_thread)
        thread.daemon = True
        thread.start()
    
    def execute_ai_move(self):
        """Execute pending AI move on the main thread"""
        if self.pending_ai_move:
            move = self.pending_ai_move
            
            # Validate move
            piece = self.game.board.squares[move.initial.row][move.initial.col].piece
            if piece and piece.color == args.ai:
                self.game.board.cal_move(piece, move.initial.row, move.initial.col, bool=True)
                
                if self.game.board.valid_move(piece, move):
                    captured = self.game.board.squares[move.final.row][move.final.col].has_piece()
                    self.game.board.move(piece, move)
                    
                    # Play sound
                    self.game.play_sound(captured)
                    
                    # Switch to human player
                    self.game.next_play()
            
            self.pending_ai_move = None
    
    def draw_ai_status(self):
        """Draw AI status on screen"""
        font = pygame.font.Font(None, 36)
        
        if self.ai_thinking:
            text = font.render("AI is thinking...", True, (255, 0, 0))
            self.screen.blit(text, (10, 10))
        
        # Show whose turn it is
        turn_text = f"Current player: {self.game.next_player.upper()}"
        if self.game.next_player == args.player:
            turn_text += " (Human)"
        else:
            turn_text += " (AI)"
        
        turn_surface = font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(turn_surface, (10, HEIGHT - 40))
    
    def check_game_over(self):
        """Check if the game is over"""
        # This is a simplified check - you might want to implement proper checkmate detection
        red_general_exists = False
        black_general_exists = False
        
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.board.squares[row][col].piece
                if piece and piece.name == 'general':
                    if piece.color == args.player:
                        red_general_exists = True
                    else:
                        black_general_exists = True
        
        if not red_general_exists:
            return 'black'
        elif not black_general_exists:
            return 'red'
        
        return None
    
    def draw_game_over(self, winner):
        """Draw game over screen"""
        font = pygame.font.Font(None, 72)
        
        if winner == args.player:
            text = "Human Wins!"
            color = (0, 255, 0)
        elif winner == args.ai:
            text = "AI Wins!"
            color = (255, 0, 0)
        else:
            text = "Draw!"
            color = (255, 255, 0)
        
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Draw text
        self.screen.blit(text_surface, text_rect)
        
        # Draw restart instruction
        restart_font = pygame.font.Font(None, 36)
        restart_text = restart_font.render("Press R to restart", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100))
        self.screen.blit(restart_text, restart_rect)

    def main_loop(self):
        # Show board background
        if args.ai == 'black':

            game = self.game
            screen = self.screen
            board = self.game.board
            dragger = self.game.dragger
            
            game_winner = None
            clock = pygame.time.Clock()

            while True:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN and not game_winner:
                        # Only allow human moves when it's red's turn and not AI thinking
                        if game.next_player == self.human_color and not self.ai_thinking:
                            dragger.update_mouse(event.pos)

                            clicked_row = dragger.mouseY // SQSIZEY
                            clicked_col = dragger.mouseX // SQSIZEX

                            if board.squares[clicked_row][clicked_col].has_piece():
                                piece = board.squares[clicked_row][clicked_col].piece
                                if piece.color == game.next_player:
                                    board.cal_move(piece, clicked_row, clicked_col, bool=True)
                                    dragger.save_initial(event.pos)
                                    dragger.drag_piece(piece)

                    elif event.type == pygame.MOUSEMOTION and not game_winner:
                        if dragger.dragging and game.next_player == self.human_color:
                            dragger.update_mouse(event.pos)

                    elif event.type == pygame.MOUSEBUTTONUP and not game_winner:
                        if dragger.dragging and game.next_player == self.human_color:
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
                                game.next_play()

                                # After human move, trigger AI move
                                if game.next_player == args.ai:
                                    self.ai_make_move()

                        dragger.undrag_piece()

                    elif event.type == pygame.KEYDOWN:
                        # Reset game
                        if event.key == pygame.K_r:
                            game.reset()
                            game = self.game
                            board = self.game.board
                            dragger = self.game.dragger
                            game_winner = None
                            self.ai_thinking = False
                            self.pending_ai_move = None

                    # Quit app     
                    elif event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Execute pending AI move
                if self.pending_ai_move and not self.ai_thinking:
                    self.execute_ai_move()

                # Check for game over
                if not game_winner:
                    game_winner = self.check_game_over()

                # Draw everything
                game.show_board(screen)
                game.show_moves(screen)
                game.show_last_move(screen)
                game.show_pieces(screen)

                if dragger.dragging:
                    dragger.update_blit(screen)

                # Draw AI status
                self.draw_ai_status()

                # Draw game over screen
                if game_winner:
                    self.draw_game_over(game_winner)

                pygame.display.update()

        else:
            game = self.game
            screen = self.screen
            board = self.game.board
            dragger = self.game.dragger
            
            game_winner = None
            clock = pygame.time.Clock()

            self.ai_make_move()

            while True:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN and not game_winner:
                        # Only allow human moves when it's red's turn and not AI thinking
                        if game.next_player == self.human_color and not self.ai_thinking:
                            dragger.update_mouse(event.pos)

                            clicked_row = dragger.mouseY // SQSIZEY
                            clicked_col = dragger.mouseX // SQSIZEX

                            if board.squares[clicked_row][clicked_col].has_piece():
                                piece = board.squares[clicked_row][clicked_col].piece
                                if piece.color == game.next_player:
                                    board.cal_move(piece, clicked_row, clicked_col, bool=True)
                                    dragger.save_initial(event.pos)
                                    dragger.drag_piece(piece)

                    elif event.type == pygame.MOUSEMOTION and not game_winner:
                        if dragger.dragging and game.next_player == self.human_color:
                            dragger.update_mouse(event.pos)

                    elif event.type == pygame.MOUSEBUTTONUP and not game_winner:
                        if dragger.dragging and game.next_player == self.human_color:
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
                                game.next_play()

                                # After human move, trigger AI move
                                if game.next_player == args.ai:
                                    self.ai_make_move()

                        dragger.undrag_piece()

                    elif event.type == pygame.KEYDOWN:
                        # Reset game
                        if event.key == pygame.K_r:
                            game.reset()
                            game = self.game
                            board = self.game.board
                            dragger = self.game.dragger
                            game_winner = None
                            self.ai_thinking = False
                            self.pending_ai_move = None

                    # Quit app     
                    elif event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # Execute pending AI move
                if self.pending_ai_move and not self.ai_thinking:
                    self.execute_ai_move()

                # Check for game over
                if not game_winner:
                    game_winner = self.check_game_over()

                # Draw everything
                game.show_board(screen)
                game.show_moves(screen)
                game.show_last_move(screen)
                game.show_pieces(screen)

                if dragger.dragging:
                    dragger.update_blit(screen)

                # Draw AI status
                self.draw_ai_status()

                # Draw game over screen
                if game_winner:
                    self.draw_game_over(game_winner)

                pygame.display.update()
                clock.tick(1)  

if __name__ == "__main__":
    main = Main()
    main.main_loop()