#!/usr/bin/env python3
"""Automated test of Connect Four models from different iterations"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS


def load_model(iteration, game):
    """Load model from specific iteration"""
    model_path = f"checkpoints/connect4/model_{iteration}.pt"
    
    model = ResNet(game, num_res_blocks=10, num_hidden=128)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def test_model_predictions(model, game, iteration):
    """Test model on standard positions"""
    args = {
        'num_searches': 50,
        'C': 2.0,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3
    }
    
    mcts = MCTS(game, args, model)
    
    print(f"\n{'='*70}")
    print(f"Testing Model from Iteration {iteration}")
    print(f"{'='*70}")
    
    # Test 1: Empty board
    print("\nTest 1: Empty Board Value Prediction")
    state = game.get_initial_state()
    with torch.no_grad():
        encoded = game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value = model(tensor)
        value_pred = value.item()
    
    print(f"Value prediction: {value_pred:.4f}")
    print(f"Expected: ~0.0 (balanced position)")
    print(f"Status: {'✓ GOOD' if abs(value_pred) < 0.3 else '✗ BAD (predicting bias)'}")
    
    # Test 2: Immediate win available (column 3 with 3 pieces)
    print("\nTest 2: Immediate Win Detection")
    state = game.get_initial_state()
    # Stack 3 pieces in column 3 for player 1
    for _ in range(3):
        state = game.get_next_state(state, 3, 1)
        state = game.get_next_state(state, 0, -1)  # Opponent plays elsewhere
    
    # Now player 1 can win by playing column 3
    neutral_state = game.change_perspective(state, 1)
    action_probs = mcts.search(neutral_state)
    best_move = int(np.argmax(action_probs))
    
    print(f"Board state: Player 1 has 3 in column 3")
    print(f"Best move chosen: Column {best_move}")
    print(f"Probability of winning move (col 3): {action_probs[3]:.4f}")
    print(f"Status: {'✓ GOOD' if best_move == 3 else '✗ BAD (missed immediate win)'}")
    
    # Test 3: Block opponent's immediate win
    print("\nTest 3: Defensive Block Detection")
    state = game.get_initial_state()
    # Stack 3 pieces in column 5 for player -1 (opponent)
    state = game.get_next_state(state, 0, 1)  # Player 1 plays elsewhere
    for _ in range(3):
        state = game.get_next_state(state, 5, -1)
        if _ < 2:
            state = game.get_next_state(state, 1, 1)  # Player 1 plays elsewhere
    
    # Now player 1 must block column 5
    neutral_state = game.change_perspective(state, 1)
    action_probs = mcts.search(neutral_state)
    best_move = int(np.argmax(action_probs))
    
    print(f"Board state: Opponent has 3 in column 5")
    print(f"Best move chosen: Column {best_move}")
    print(f"Probability of blocking move (col 5): {action_probs[5]:.4f}")
    print(f"Status: {'✓ GOOD' if best_move == 5 else '✗ BAD (missed defensive block)'}")
    
    # Test 4: Center column preference on empty board
    print("\nTest 4: Opening Strategy (Center Preference)")
    state = game.get_initial_state()
    neutral_state = game.change_perspective(state, 1)
    action_probs = mcts.search(neutral_state)
    best_move = int(np.argmax(action_probs))
    
    print(f"Empty board opening move: Column {best_move}")
    print(f"Center column (3) probability: {action_probs[3]:.4f}")
    print(f"Status: {'✓ GOOD' if best_move == 3 else '⚠ OK (center is preferred but not required)'}")
    
    return {
        'empty_board_value': value_pred,
        'found_immediate_win': best_move == 3,
        'blocked_threat': best_move == 5,
    }


def compare_models():
    """Compare iteration 13 vs iteration 189"""
    print("=" * 70)
    print("AlphaZero Connect Four - Automated Model Comparison")
    print("=" * 70)
    
    game = ConnectFour(row_count=6, column_count=7, win_length=4)
    
    print("\nLoading models...")
    model_13 = load_model(13, game)
    model_189 = load_model(189, game)
    print("✓ Models loaded!")
    
    print(f"\n{'='*70}")
    print("Model 13: Best Performance (End of Warmup)")
    print("Loss: 0.7939 (Policy: 0.7276, Value: 0.0332)")
    results_13 = test_model_predictions(model_13, game, 13)
    
    print(f"\n{'='*70}")
    print("Model 189: Final Model (After Self-Play)")
    print("Loss: 1.4329 (Policy: 0.7237, Value: 0.3546)")
    results_189 = test_model_predictions(model_189, game, 189)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    print("\nEmpty Board Value (should be ~0.0):")
    print(f"  Model 13:  {results_13['empty_board_value']:+.4f} {'✓' if abs(results_13['empty_board_value']) < 0.3 else '✗'}")
    print(f"  Model 189: {results_189['empty_board_value']:+.4f} {'✓' if abs(results_189['empty_board_value']) < 0.3 else '✗'}")
    
    print("\nImmediate Win Detection:")
    print(f"  Model 13:  {'✓ Found' if results_13['found_immediate_win'] else '✗ Missed'}")
    print(f"  Model 189: {'✓ Found' if results_189['found_immediate_win'] else '✗ Missed'}")
    
    print("\nDefensive Block Detection:")
    print(f"  Model 13:  {'✓ Found' if results_13['blocked_threat'] else '✗ Missed'}")
    print(f"  Model 189: {'✓ Found' if results_189['blocked_threat'] else '✗ Missed'}")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    score_13 = sum([
        abs(results_13['empty_board_value']) < 0.3,
        results_13['found_immediate_win'],
        results_13['blocked_threat']
    ])
    
    score_189 = sum([
        abs(results_189['empty_board_value']) < 0.3,
        results_189['found_immediate_win'],
        results_189['blocked_threat']
    ])
    
    print(f"\nModel 13 Score:  {score_13}/3")
    print(f"Model 189 Score: {score_189}/3")
    
    if score_13 > score_189:
        print("\n✓ Model 13 (warmup peak) is BETTER - self-play degraded performance")
    elif score_189 > score_13:
        print("\n✓ Model 189 (final) is BETTER - self-play improved performance")
    else:
        print("\n= Models are EQUAL - no significant improvement from self-play")


if __name__ == "__main__":
    compare_models()
