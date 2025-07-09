import torch

if torch.cuda.is_available():
    print("✅ CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("❌ CUDA is not available. Using CPU.")
