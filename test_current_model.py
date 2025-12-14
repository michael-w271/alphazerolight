#!/usr/bin/env python3
"""Test the current Connect Four model during training"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS


def load_latest_model(game):
    """Load the most recent model checkpoint"""
    checkpoint_dir = Path("checkpoints/connect4")
    
    # Find latest model file
    model_files = list(checkpoint_dir.glob("model_*.pt"))
    if not model_files:
        print("No model checkpoints found!")
        return None
    
    latest_model = max(model_files, key=lambda p: int(p.stem.split('_')[1]))
    iteration = int(latest_model.stem.split('_')[1])
    
    print(f"Loading model from iteration {iteration}...")
    
    # Create model with correct architecture
    model = ResNet(game, num_res_blocks=15, num_hidden=256)
    model.load_state_dict(torch.load(latest_model, map_location='cpu'))
    model.eval()
    
    return model, iteration


def test_model_tactics(model, game, iteration):
    """Test model on tactical positions"""
    args = {
        'num_searches': 100,
        'C': 2.0,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3
    }
    
    mcts = MCTS(game, args, model)
    
    print(f"\n{'='*70}")
    print(f"Testing Model from Iteration {iteration}")
    print(f"{'='*70}")
    
    # Test 1: Empty board value
    print("\n🧪 Test 1: Empty Board Value Prediction")
    state = game.get_initial_state()
    with torch.no_grad():
        encoded = game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value = model(tensor)
        value_pred = value.item()
    
    print(f"Value prediction: {value_pred:+.4f}")
    print(f"Expected: ~0.0 (balanced position)")
    if abs(value_pred) < 0.3:
        print("✅ PASS - Model has realistic value predictions")
    else:
        print("❌ FAIL - Model predicting strong bias")
    
    # Test 2: Immediate win detection
    print("\n🧪 Test 2: Immediate Win Detection (3 in column, can win with 4th)")
    state = game.get_initial_state()
    # Stack 3 pieces in column 3 for player 1
    for _ in range(3):
        state = game.get_next_state(state, 3, 1)
        state = game.get_next_state(state, 0, -1)
    
    neutral_state = game.change_perspective(state, 1)
    action_probs = mcts.search(neutral_state)
    best_move = int(np.argmax(action_probs))
    
    print(f"Best move chosen: Column {best_move}")
    print(f"Probability of winning move (col 3): {action_probs[3]:.4f}")
    print(f"Top 3 moves: ", end="")
    top_3 = np.argsort(action_probs)[-3:][::-1]
    for col in top_3:
        print(f"col{col}={action_probs[col]:.3f} ", end="")
    print()
    
    if best_move == 3 and action_probs[3] > 0.5:
        print("✅ PASS - Model finds immediate win")
    elif best_move == 3:
        print("⚠️  PARTIAL - Finds win but not confident")
    else:
        print("❌ FAIL - Misses immediate win")
    
    # Test 3: Block opponent threat
    print("\n🧪 Test 3: Defensive Block (opponent has 3 in column 5)")
    state = game.get_initial_state()
    # Opponent has 3 in column 5
    state = game.get_next_state(state, 0, 1)
    for _ in range(3):
        state = game.get_next_state(state, 5, -1)
        if _ < 2:
            state = game.get_next_state(state, 1, 1)
    
    neutral_state = game.change_perspective(state, 1)
    action_probs = mcts.search(neutral_state)
    best_move = int(np.argmax(action_probs))
    
    print(f"Best move chosen: Column {best_move}")
    print(f"Probability of blocking move (col 5): {action_probs[5]:.4f}")
    print(f"Top 3 moves: ", end="")
    top_3 = np.argsort(action_probs)[-3:][::-1]
    for col in top_3:
        print(f"col{col}={action_probs[col]:.3f} ", end="")
    print()
    
    if best_move == 5 and action_probs[5] > 0.5:
        print("✅ PASS - Model blocks threat correctly")
    elif best_move == 5:
        print("⚠️  PARTIAL - Blocks but not confident")
    else:
        print("❌ FAIL - Misses defensive block")
    
    # Test 4: Opening preference
    print("\n🧪 Test 4: Opening Move (center preference)")
    state = game.get_initial_state()
    neutral_state = game.change_perspective(state, 1)
    action_probs = mcts.search(neutral_state)
    best_move = int(np.argmax(action_probs))
    
    print(f"Opening move: Column {best_move}")
    print(f"Center (col 3) probability: {action_probs[3]:.4f}")
    print(f"Move distribution: ", end="")
    for col in range(7):
        print(f"col{col}={action_probs[col]:.3f} ", end="")
    print()
    
    if best_move == 3:
        print("✅ GOOD - Prefers center opening")
    else:
        print("⚠️  OK - Center not required but often good")


def main():
    game = ConnectFour(row_count=6, column_count=7, win_length=4)
    
    model, iteration = load_latest_model(game)
    if model is None:
        return
    
    test_model_tactics(model, game, iteration)
    
    print(f"\n{'='*70}")
    print("Test Complete - Training can continue running in background")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
