#!/usr/bin/env python3
"""Debug MCTS visit counts to see which nodes are being explored"""
import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS, Node

game = ConnectFour()
model = ResNet(game, num_res_blocks=10, num_hidden=128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
checkpoint = torch.load('checkpoints/connect4/model_74.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

state = np.zeros((6, 7), dtype=np.float32)
state[5, 3] = 1
state[4, 3] = 1
state[3, 3] = 1

# Patch MCTS to save root node
class DebugMCTS(MCTS):
    def search(self, state, add_noise=True, temperature=None):
        result = super().search(state, add_noise, temperature)
        self.last_root = root  # Will fail but let's try another way
        return result

args = {'C': 2, 'num_searches': 100, 'dirichlet_epsilon': 0.0, 'dirichlet_alpha': 0.3}
mcts = MCTS(game, args, model)

# Run search
print("Running MCTS with 100 searches...")
action_probs = mcts.search(state, add_noise=False)

print(f"\nFinal action probabilities: {action_probs}")
print(f"Best move: column {action_probs.argmax()}")

# Can't access root from outside, so let's trace manually
# Actually, let me modify the MCTS code temporarily to print debug info
