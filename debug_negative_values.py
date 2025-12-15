#!/usr/bin/env python3
"""
Debug why all model values are negative
"""

import torch
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.config_connect4 import MODEL_CONFIG

game = ConnectFour(6, 7, 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model 100
model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden']).to(device)
model.load_state_dict(torch.load("checkpoints/connect4/model_100.pt", map_location=device))
model.eval()

print("Testing if model outputs are all negative...")
print("="*70)

# Test empty board
state = game.get_initial_state()
print("\n1. Empty Board (should be ~0.0):")
print(state)

# IMPORTANT: State is already from Player 1's perspective
# When encoding, it's treated as "current player" = 1
encoded = game.get_encoded_state(state)
print(f"\nEncoded shape: {encoded.shape}")
print(f"Encoded planes - [opponent=-1, empty=0, current=1]:\n{encoded}")

tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    policy, value = model(tensor)
    value = value.item()

print(f"\nValue from Player 1's perspective: {value:+.4f}")

# Test position where player 1 is winning
winning_state = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],
])

print("\n2. Position where Player 1 has strong position:")
print(winning_state)

encoded2 = game.get_encoded_state(winning_state)
print(f"\nEncoded player planes:\n{encoded2}")

tensor2 = torch.tensor(encoded2, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    policy2, value2 = model(tensor2)
    value2 = value2.item()

print(f"\nValue: {value2:+.4f} (should be positive!)")

#  Check if model output layer has bias issues
print("\n" + "="*70)
print("CHECKING MODEL OUTPUT LAYER")
print("="*70)

# Get final layer
print(f"\nModel value head final layer:")
for name, param in model.named_parameters():
    if 'value_head' in name and 'weight' in name:
        print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
    if 'value_head' in name and 'bias' in name:
        print(f"{name}: {param.data.item():.4f}")

# Test on 100 random positions
print("\n" + "="*70)
print("TESTING ON 100 RANDOM POSITIONS")
print("="*70)

values = []
for _ in range(100):
    random_state = game.get_initial_state()
    # Make 5-15 random moves
    num_moves = np.random.randint(5, 16)
    player = 1
    for _ in range(num_moves):
        valid_moves = game.get_valid_moves(random_state)
        if not np.any(valid_moves):
            break
        move = np.random.choice(np.where(valid_moves)[0])
        random_state = game.get_next_state(random_state, move, player)
        value, is_terminal = game.get_value_and_terminated(random_state, move)
        if is_terminal:
            break
        player = game.get_opponent(player)
    
    encoded = game.get_encoded_state(random_state)
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, value = model(tensor)
        values.append(value.item())

values = np.array(values)
print(f"\nValue predictions on 100 random positions:")
print(f"  Mean: {values.mean():.4f}")
print(f"  Std:  {values.std():.4f}")
print(f"  Min:  {values.min():.4f}")
print(f"  Max:  {values.max():.4f}")
print(f"  Negative count: {np.sum(values < 0)}/100")
print(f"  Positive count: {np.sum(values > 0)}/100")

if values.mean() < -0.5:
    print(f"\n❌ MODEL IS SEVERELY BIASED NEGATIVE!")
    print(f"   This suggests training collected wrong outcomes")
    print(f"   OR perspective is flipped during evaluation")
elif abs(values.mean()) < 0.2:
    print(f"\n✅ Model values are relatively balanced")
else:
    print(f"\n⚠️  Model has bias (mean != 0)")
