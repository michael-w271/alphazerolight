#!/usr/bin/env python3
"""
Quick test to verify PATCH_MCTS_01 (UCB sign fix) is working.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import MODEL_CONFIG, MCTS_CONFIG
import torch

print("="*70)
print("TESTING UCB SIGN FIX - CRITICAL VALIDATION")
print("="*70)

game = ConnectFour(6, 7, 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a fresh untrained model (to avoid biases from broken training)
model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden']).to(device)
model.eval()

# Test position: Player 1 can win immediately at column 3
state = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],  # Win at column 3
])

print("\nTest Position: Player 1 can WIN at column 3")
print(state)

# Run MCTS with 50 searches
args = {**MCTS_CONFIG, 'num_searches': 50, 'mcts_batch_size': 1}
mcts = MCTS(game, args, model)

print(f"\nRunning MCTS with {args['num_searches']} searches, batch_size={args.get('mcts_batch_size', 1)}...")
action_probs = mcts.search(state, 1)  # Player 1's turn

print("\n📊 MCTS Visit Distribution:")
for col in range(7):
    visits = action_probs[col]
    bar = "█" * int(visits * 50)
    symbol = " ← WIN!" if col == 3 else ""
    print(f"  Column {col}: {visits:.3f} {bar}{symbol}")

win_move_prob = action_probs[3]
best_move = np.argmax(action_probs)

print(f"\n{'='*70}")
print("RESULT")
print(f"{'='*70}")

if best_move == 3:
    print(f"✅ SUCCESS! MCTS chose column {best_move} (the winning move)")
    print(f"   Win move probability: {win_move_prob:.1%}")
    print(f"\n🎉 UCB sign fix is WORKING!")
    print(f"   MCTS correctly recognizes winning moves now.")
else:
    print(f"❌ FAILED! MCTS chose column {best_move} instead of 3")
    print(f"   Win move probability: {win_move_prob:.1%}")
    print(f"\n⚠️  This suggests the UCB fix may not be applied correctly,")
    print(f"   or there's another issue preventing MCTS from finding wins.")

print(f"\n💡 Note: With an untrained model, visits should still concentrate")
print(f"   on the winning move because MCTS searches show it leads to a win.")
print(f"   If visits are uniform, UCB/backprop has a sign error.")
