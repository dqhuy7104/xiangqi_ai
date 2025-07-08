# ai_vs_human.py - Integration with pygame interface
import pygame
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from game import Game
from ai_agent import DQNAgent, XiangqiEnvironment
from src.const import *
from src.square import Square
from src.move import Move

class AIGame(Game):
    """Extended game class with AI integration"""
    def __init__(self):
        super().__init__()
        self.ai_agent = DQNAgent()
        self.ai_color = 'black'  # AI plays as black
        self.ai_enabled = False
        self.env = XiangqiEnvironment(self)
        
    def load_ai_model(self, model_path):
        """Load trained AI model"""
        try:
            self.ai_agent.load(model_path)
            self.ai_enabled = True
            print("AI model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading AI model: {e}")
            self.ai_enabled = False
            return False
    
    def get_ai_move(self):
        """Get AI move"""
        if not self.ai_enabled:
            return None
        
        state = self.env.get_state()
        valid_moves = self.env.get_valid_moves(self.ai_color)
        
        if not valid_moves:
            return None
        
        # Set epsilon to 0 for no random moves
        old_epsilon = self.ai_agent.epsilon
        self.ai_agent.epsilon = 0
        
        move = self.ai_agent.act(state, valid_moves)
        
        self.ai_agent.epsilon = old_epsilon
        
        return move
    
    def make_ai_move(self):
        """Execute AI move"""
        if self.next_player != self.ai_color:
            return False
        
        ai_move = self.get_ai_move()
        if ai_move is None:
            return False
        
        from_row, from_col, to_row, to_col = ai_move
        
        # Get the piece
        piece = self.board.squares[from_row][from_col].piece
        if not piece:
            return False
        
        # Create move object
        from move import Move
        from square import Square
        
        initial = Square(from_row, from_col)
        final_piece = self.board.squares[to_row][to_col].piece
        final = Square(to_row, to_col, final_piece)
        move = Move(initial, final)
        
        # Validate move
        self.board.cal_move(piece, from_row, from_col)
        valid_moves = piece.moves_empty + piece.moves_rival
        
        is_valid = any(m.final.row == to_row and m.final.col == to_col for m in valid_moves)
        
        if is_valid:
            # Make the move
            self.board.move(piece, move)
            
            # Play sound
            self.play_sound(captured=final_piece is not None)
            
            # Switch turns
            self.next_player = 'red' if self.next_player == 'black' else 'black'
            
            return True
        
        return False

def main_with_ai():
    """Main function with AI support"""
    pygame.init()
    
    # Create game
    game = AIGame()
    
    # Ask if player wants to play against AI
    print("Xiangqi Game")
    print("1. Human vs Human")
    print("2. Human vs AI")
    choice = input("Enter your choice: ")
    
    if choice == '2':
        model_path = input("Enter AI model path (default: models/final_agent2.pth): ") or "models/final_agent2.pth"
        if not game.load_ai_model(model_path):
            print("Failed to load AI, playing Human vs Human")
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Xiangqi')
    
    # Game loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                game.dragger.update_mouse(event.pos)
                
                clicked_row = game.dragger.mouseY // SQSIZEY
                clicked_col = game.dragger.mouseX // SQSIZEX
                
                if game.board.squares[clicked_row][clicked_col].has_piece():
                    piece = game.board.squares[clicked_row][clicked_col].piece
                    
                    if piece.color == game.next_player:
                        game.board.cal_move(piece, clicked_row, clicked_col)
                        game.dragger.save_initial(event.pos)
                        game.dragger.drag_piece(piece)
                        
                        game.show_moves(screen)
                        game.show_last_move(screen)
                        
            elif event.type == pygame.MOUSEMOTION:
                
                if game.dragger.dragging:
                    game.dragger.update_mouse(event.pos)
                    
                    game.show_board(screen)
                    game.show_last_move(screen)
                    game.show_moves(screen)
                    game.show_pieces(screen)
                    game.dragger.update_blit(screen)
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if game.dragger.dragging:
                    game.dragger.update_mouse(event.pos)
                    
                    released_row = game.dragger.mouseY // SQSIZEY
                    released_col = game.dragger.mouseX // SQSIZEX
                    
                    initial = Square(game.dragger.initial_row, game.dragger.initial_col)
                    final = Square(released_row, released_col)
                    move = Move(initial, final)
                    
                    if game.board.valid_move(game.dragger.piece, move):
                        captured = game.board.squares[released_row][released_col].has_piece()
                        game.board.move(game.dragger.piece, move)
                        
                        game.play_sound(captured)
                        
                        game.show_board(screen)
                        game.show_last_move(screen)
                        game.show_pieces(screen)
                        
                        game.next_player = 'red' if game.next_player == 'black' else 'black'
                        
                    else:
                        game.dragger.piece = None
                        game.dragger.dragging = False
                        game.dragger.mouseX = 0
                        game.dragger.mouseY = 0
                        
                game.dragger.undrag_piece()
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    game.change_theme()
                    
                elif event.key == pygame.K_r:
                    game.reset()
                    game = AIGame()
                    if choice == '2':
                        game.load_ai_model(model_path)
        
        # AI move
        if game.ai_enabled and game.next_player == game.ai_color:
            pygame.time.wait(50)  # Small delay for better UX
            game.make_ai_move()
        
        # Draw everything
        game.show_board(screen)
        game.show_last_move(screen)
        game.show_moves(screen)
        game.show_pieces(screen)

        
        if game.dragger.dragging:
            game.dragger.update_blit(screen)
            
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_with_ai()