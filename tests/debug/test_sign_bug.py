#!/usr/bin/env python3
"""
Final check: Does the model understand that winning is good?
Test on positions it WAS trained on (pre-terminal, not terminal).
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet

# Load model
game = ConnectFour()
model = ResNet(game, num_res_blocks=10, num_hidden=128)
model.load_state_dict(torch.load("checkpoints/connect4/model_9.pt", map_location='cpu'))
model.eval()

print("="*70)
print("TESTING: Does model know winning is good?")
print("="*70)

# Test 1: Position where player 1 is ABOUT TO WIN (not yet won)
state = game.get_initial_state()
state = game.get_next_state(state, 0, 1)
state = game.get_next_state(state, 0, -1)
state = game.get_next_state(state, 1, 1)
state = game.get_next_state(state, 1, -1)
state = game.get_next_state(state, 2, 1)
state = game.get_next_state(state, 2, -1)
# Now player 1 can win with column 3

print("\n1. Position where player 1 can win immediately:")
print("   Bottom row:", state[5, :])
print("   Player 1 has 0,1,2 and can play 3 to win")

# From player 1's perspective
neutral_state = game.change_perspective(state, 1)
encoded = game.get_encoded_state(neutral_state)

with torch.no_grad():
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
    policy_logits, value = model(tensor)
    value_pred = value.item()
    policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

print(f"\n   From player 1's view:")
print(f"   Value prediction: {value_pred:+.4f}")
print(f"   Policy for winning move (col 3): {policy_probs[3]:.4f}")
print(f"   Expected: value > 0 (player 1 is winning)")

if value_pred > 0:
    print(f"   ✅ Correct sign! Model knows player 1 is winning")
else:
    print(f"   ❌ WRONG SIGN! Model thinks player 1 is losing")

# From player -1's perspective (same position, opponent's turn)
neutral_state = game.change_perspective(state, -1)
encoded = game.get_encoded_state(neutral_state)

with torch.no_grad():
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
    policy_logits, value = model(tensor)
    value_pred_p2 = value.item()

print(f"\n   From player -1's view (opponent):")
print(f"   Value prediction: {value_pred_p2:+.4f}")
print(f"   Expected: value < 0 (player -1 is losing)")

if value_pred_p2 < 0:
    print(f"   ✅ Correct sign! Model knows player -1 is losing")
else:
    print(f"   ❌ WRONG SIGN! Model thinks player -1 is winning")

# Test 2: Position where player -1 is ABOUT TO WIN
print("\n" + "="*70)
state = game.get_initial_state()
state = game.get_next_state(state, 0, -1)
state = game.get_next_state(state, 4, 1)
state = game.get_next_state(state, 1, -1)
state = game.get_next_state(state, 4, 1)
state = game.get_next_state(state, 2, -1)
state = game.get_next_state(state, 4, 1)
# Now player -1 can win with column 3

print("\n2. Position where player -1 can win immediately:")
print("   Bottom row:", state[5, :])
print("   Player -1 has 0,1,2 and can play 3 to win")

# From player -1's perspective
neutral_state = game.change_perspective(state, -1)
encoded = game.get_encoded_state(neutral_state)

with torch.no_grad():
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
    policy_logits, value = model(tensor)
    value_pred = value.item()

print(f"\n   From player -1's view:")
print(f"   Value prediction: {value_pred:+.4f}")
print(f"   Expected: value > 0 (player -1 is winning)")

if value_pred > 0:
    print(f"   ✅ Correct sign! Model knows player -1 is winning")
else:
    print(f"   ❌ WRONG SIGN! Model thinks player -1 is losing")

# From player 1's perspective
neutral_state = game.change_perspective(state, 1)
encoded = game.get_encoded_state(neutral_state)

with torch.no_grad():
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
    policy_logits, value = model(tensor)
    value_pred_p1 = value.item()

print(f"\n   From player 1's view:")
print(f"   Value prediction: {value_pred_p1:+.4f}")
print(f"   Expected: value < 0 (player 1 is losing)")

if value_pred_p1 < 0:
    print(f"   ✅ Correct sign! Model knows player 1 is losing")
else:
    print(f"   ❌ WRONG SIGN! Model thinks player 1 is winning")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
if all([value_pred > 0, value_pred_p2 < 0, value_pred_p1 < 0]):
    print("✅ ALL SIGNS CORRECT - No sign bug in training!")
    print("   Model correctly predicts positive values for winning side")
    print("   Model correctly predicts negative values for losing side")
else:
    print("❌ SIGN BUG DETECTED!")
    print("   Model has inverted understanding of wins/losses")
