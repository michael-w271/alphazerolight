import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.config_connect4 import MODEL_CONFIG

# Load game and model
game = ConnectFour()
model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden'])
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint 40
checkpoint = torch.load('checkpoints/connect4/model_40.pt', map_location=model.device)
model.load_state_dict(checkpoint)
model.eval()

# Test state
state = np.zeros((6, 7), dtype=np.float32)
state[5, 3] = 1
state[4, 3] = 1
state[3, 3] = 1

print("Test state:")
print(state)

# Get model output
encoded = game.get_encoded_state(state)
with torch.no_grad():
    policy_logits, value = model(torch.tensor(encoded, device=model.device).unsqueeze(0))
    policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

print(f"\nRaw model policy output: {policy_probs}")
print(f"Value: {value.item():.4f}")
print(f"\nColumn 3 (winning move) probability: {policy_probs[3]:.6f}")
