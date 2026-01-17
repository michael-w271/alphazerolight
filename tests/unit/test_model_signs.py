#!/usr/bin/env python3
"""
Test current model to verify signs are correct - quick sanity check.
"""

import torch
import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import MODEL_CONFIG, MCTS_CONFIG

game = ConnectFour(6, 7, 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Find latest model
checkpoint_dir = Path("checkpoints/connect4")
models = sorted(checkpoint_dir.glob("model_*.pt"))
if not models:
    print("❌ No models found!")
    exit(1)

latest_model = models[-1]
iteration = int(latest_model.stem.split('_')[1])

print("="*70)
print(f"TESTING MODEL AT ITERATION {iteration} - SIGN VERIFICATION")
print("="*70)

# Load model
model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden']).to(device)
model.load_state_dict(torch.load(latest_model, map_location=device))
model.eval()

def test_position(name, state, expected_sign, description):
    """Test a position and check if value sign is correct"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"\n{description}")
    print("\nBoard (Player 1's perspective):")
    print(state)
    
    # Encode and predict
    encoded = game.get_encoded_state(state)
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy, value = model(tensor)
        value = value.item()
    
    print(f"\n📊 Model Prediction:")
    print(f"   Value: {value:+.4f}")
    
    # Check sign
    if expected_sign == "positive":
        if value > 0:
            result = "✅ CORRECT - Value is positive (Player 1 winning)"
        else:
            result = "❌ WRONG - Value is negative (should be positive!)"
    elif expected_sign == "negative":
        if value < 0:
            result = "✅ CORRECT - Value is negative (Player 1 losing)"
        else:
            result = "❌ WRONG - Value is positive (should be negative!)"
    else:  # balanced
        if abs(value) < 0.3:
            result = f"✅ CORRECT - Value is balanced (~0)"
        else:
            result = f"⚠️  BIASED - Value is {value:+.4f} (should be ~0)"
    
    print(f"\n{result}")
    return value

# Test 1: Empty board (should be balanced)
state1 = game.get_initial_state()
v1 = test_position(
    "Empty Board",
    state1,
    "balanced",
    "No pieces, should be neutral ~0.0"
)

# Test 2: Player 1 has strong position (should be positive)
state2 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],
])
v2 = test_position(
    "Player 1 Strong Position",
    state2,
    "positive",
    "Player 1 has 3-in-a-row bottom + vertical threat"
)

# Test 3: Player 1 can win immediately (should be very positive)
state3 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],  # Win at column 3
])
v3 = test_position(
    "Player 1 Can Win",
    state3,
    "positive",
    "Player 1 can win on next move at column 3"
)

# Test 4: Player -1 strong position (should be negative)
state4 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0],
])
v4 = test_position(
    "Player -1 Strong Position",
    state4,
    "negative",
    "Player -1 has 3-in-a-row bottom + vertical threat (from Player 1's view)"
)

# Test 5: Player -1 can win (should be very negative)
state5 = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0],  # Player -1 wins at column 3
])
v5 = test_position(
    "Player -1 Can Win",
    state5,
    "negative",
    "Player -1 can win on next move at column 3 (from Player 1's view)"
)

# Summary
print(f"\n{'='*70}")
print("SUMMARY - SIGN VERIFICATION")
print(f"{'='*70}")

results = [
    ("Empty board", v1, "~0", abs(v1) < 0.3),
    ("Player 1 strong", v2, ">0", v2 > 0),
    ("Player 1 can win", v3, ">0", v3 > 0),
    ("Player -1 strong", v4, "<0", v4 < 0),
    ("Player -1 can win", v5, "<0", v5 < 0),
]

passed = sum(1 for _, _, _, correct in results if correct)
total = len(results)

print(f"\nTests Passed: {passed}/{total}")
for name, value, expected, correct in results:
    status = "✅" if correct else "❌"
    print(f"  {status} {name:20s}: {value:+.4f} (expected {expected})")

if passed == total:
    print(f"\n🎉 ALL SIGNS CORRECT! The encoding fix is working!")
    print(f"   Model correctly predicts positive values for Player 1 winning positions")
    print(f"   and negative values for Player -1 winning positions.")
elif passed >= 3:
    print(f"\n⚠️  MOSTLY CORRECT - {passed}/{total} tests passed")
    print(f"   Model is learning the right direction but still weak (iteration {iteration})")
else:
    print(f"\n❌ SIGNS ARE WRONG - Only {passed}/{total} tests passed")
    print(f"   The encoding bug might still be present!")

print(f"\n💡 Note: Magnitude doesn't matter at iteration {iteration}")
print(f"   What matters: Signs are correct (+ for Player 1 winning, - for losing)")
