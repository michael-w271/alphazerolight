import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.config_connect4 import MODEL_CONFIG

game = ConnectFour()
model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden'])
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/connect4/model_40.pt', map_location=model.device)
model.load_state_dict(checkpoint)
model.eval()

# ACTUAL WIN STATE - 4 in a row
state = np.zeros((6, 7), dtype=np.float32)
state[5, 3] = 1
state[4, 3] = 1
state[3, 3] = 1
state[2, 3] = 1  # 4 in a row = WIN!

print("Winning state (4 in a row):")
print(state)
print(f"Is this a win? {game.check_win(state, 3)}")

encoded = game.get_encoded_state(state)
with torch.no_grad():
    policy_logits, value = model(torch.tensor(encoded, device=model.device).unsqueeze(0))
    value = value.item()

print(f"\nModel value for WINNING position: {value:.4f}")
print("(Should be close to +1.0 since we won!)")
