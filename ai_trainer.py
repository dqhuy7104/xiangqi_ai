from src.game import Game
from ai_agent import XiangqiTrainer
import pygame

def main():
    pygame.init()
    pygame.mixer.init()
    print("Xiangqi AI Training System")
    print("=" * 50)
    
    # Create trainer
    trainer = XiangqiTrainer()
    
    # Menu system
    while True:
        print("\n1. Start new training")
        print("2. Continue training (load existing models)")
        print("3. Play game (AI vs AI)")
        print("4. Evaluate agents")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            episodes = int(input("Enter number of episodes (default 1000): ") or "1000")
            trainer.train(num_episodes=episodes)
            
        elif choice == '2':
            try:
                model_name = input("Enter model prefix (default 'models/final'): ") or "models/final"
                trainer.load_models(model_name)
                print("Models loaded successfully!")
                episodes = int(input("Enter number of additional episodes: "))
                trainer.train(num_episodes=episodes)
            except Exception as e:
                print(f"Error loading models: {e}")
                
        elif choice == '3':
            try:
                model_name = input("Enter model prefix (default 'models/final'): ") or "models/final"
                trainer.load_models(model_name)
                print("Playing AI vs AI game...")
                trainer.play_game()
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '4':
            try:
                model_name = input("Enter model prefix (default 'models/final'): ") or "models/final"
                trainer.load_models(model_name)
                print("Evaluating agents...")
                evaluate_agents(trainer)
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")

def evaluate_agents(trainer, num_games=100):
    """Evaluate trained agents"""
    print(f"Running {num_games} evaluation games...")
    
    stats = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'game_lengths': []
    }
    
    for i in range(num_games):
        game = trainer.game_class()
        env = trainer.XiangqiEnvironment(game)
        state = env.reset()
        
        game_length = 0
        max_moves = 200
        
        # Set agents to evaluation mode (no exploration)
        old_epsilon1 = trainer.agent1.epsilon
        old_epsilon2 = trainer.agent2.epsilon
        trainer.agent1.epsilon = 0
        trainer.agent2.epsilon = 0
        
        while game_length < max_moves:
            current_player = env.game.next_player
            agent = trainer.agent1 if current_player == 'red' else trainer.agent2
            
            valid_moves = env.get_valid_moves(current_player)
            if not valid_moves:
                break
            
            action = agent.act(state, valid_moves)
            if action is None:
                break
            
            next_state, reward, done = env.make_move(action)
            state = next_state
            game_length += 1
            
            if done:
                break
        
        # Check winner
        red_general = False
        black_general = False
        
        for row in range(10):
            for col in range(9):
                piece = env.game.board.squares[row][col].piece
                if piece and piece.name == 'general':
                    if piece.color == 'red':
                        red_general = True
                    else:
                        black_general = True
        
        if red_general and not black_general:
            stats['agent1_wins'] += 1
        elif black_general and not red_general:
            stats['agent2_wins'] += 1
        else:
            stats['draws'] += 1
        
        stats['game_lengths'].append(game_length)
        
        # Restore epsilon values
        trainer.agent1.epsilon = old_epsilon1
        trainer.agent2.epsilon = old_epsilon2
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_games} games")
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Agent 1 (Red) wins: {stats['agent1_wins']} ({stats['agent1_wins']/num_games*100:.1f}%)")
    print(f"Agent 2 (Black) wins: {stats['agent2_wins']} ({stats['agent2_wins']/num_games*100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(stats['game_lengths'])/len(stats['game_lengths']):.1f} moves")

if __name__ == "__main__":
    main()