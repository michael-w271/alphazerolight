import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import TRAINING_CONFIG, MCTS_CONFIG, MODEL_CONFIG

# Load game and model
game = ConnectFour()
model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden'])
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint 40
checkpoint = torch.load('checkpoints/connect4/model_40.pt', map_location=model.device)
model.load_state_dict(checkpoint)
model.eval()

# Create MCTS
args = {**TRAINING_CONFIG, **MCTS_CONFIG, **MODEL_CONFIG}
mcts = MCTS(game, args, model)

# Test state: 3 pieces vertically in column 3
state = np.zeros((6, 7), dtype=np.float32)
state[5, 3] = 1  # Bottom
state[4, 3] = 1
state[3, 3] = 1

print("Test board (3 in a row vertically in column 3):")
print(state)
print(f"\nValid moves: {game.get_valid_moves(state)}")
print(f"\nRunning MCTS search...")

# Run MCTS
action_probs = mcts.search(state)
print(f"\nAction probabilities: {action_probs}")
print(f"Column 3 probability (win move): {action_probs[3]:.6f}")
print(f"Sum of probabilities: {action_probs.sum():.6f}")
