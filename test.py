import torch
from train import XiangqiTrainer

if torch.cuda.is_available():
    print("✅ CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("❌ CUDA is not available. Using CPU.")


trainer = XiangqiTrainer()
trainer.load_models('models/final')
    
# Play some test games
print("\nTesting trained agents...")
results = {'red': 0, 'black': 0, 'draw': 0}

for i in range(100):
    result = trainer.play_game()
    results[result] += 1

print(f"Test results over 10 games:")
print(f"Red wins: {results['red']}")
print(f"Black wins: {results['black']}")
print(f"Draws: {results['draw']}")
