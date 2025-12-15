#!/usr/bin/env python3
"""
Test model's ability to detect immediate wins and immediate losses.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS


def load_model(iteration):
    """Load specific model iteration"""
    game = ConnectFour()
    checkpoint_path = Path(f"checkpoints/connect4/model_{iteration}.pt")
    
    if not checkpoint_path.exists():
        print(f"Model iteration {iteration} not found!")
        return None, None
    
    print(f"Loading model from iteration {iteration}...")
    
    # Create model with current architecture (10 blocks, 128 hidden)
    model = ResNet(game, num_res_blocks=10, num_hidden=128)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    return model, game


def print_board(state):
    """Print the board in a readable format"""
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            if val == 1:
                print("X ", end="")
            elif val == -1:
                print("O ", end="")
            else:
                print(". ", end="")
        print()
    print()


def test_immediate_win(model, game, iteration):
    """Test: AI can win in one move"""
    print(f"\n{'='*70}")
    print(f"TEST 1: IMMEDIATE WIN DETECTION (Iteration {iteration})")
    print(f"{'='*70}")
    
    # Create a position where AI (player 1) can win immediately
    state = game.get_initial_state()
    # Build: X X X _ . . . (can win at column 3)
    state = game.get_next_state(state, 0, 1)   # X
    state = game.get_next_state(state, 0, -1)  # O somewhere else
    state = game.get_next_state(state, 1, 1)   # X
    state = game.get_next_state(state, 4, -1)  # O somewhere else
    state = game.get_next_state(state, 2, 1)   # X
    state = game.get_next_state(state, 5, -1)  # O somewhere else
    
    print("\nBoard position (AI is X, can win at column 3):")
    print_board(state)
    
    # Test from AI's perspective (player 1)
    player = 1
    neutral_state = game.change_perspective(state, player)
    
    # Run MCTS
    args = {
        'num_searches': 100,
        'C': 2.0,
        'dirichlet_epsilon': 0.0,  # No noise for testing
        'dirichlet_alpha': 0.3
    }
    
    mcts = MCTS(game, args, model)
    action_probs = mcts.search(neutral_state, add_noise=False)
    
    print("MCTS Results:")
    print(f"Action probabilities:")
    for col in range(7):
        valid = game.get_valid_moves(neutral_state)[col]
        if valid:
            print(f"  Column {col}: {action_probs[col]:.4f}")
    
    best_move = int(np.argmax(action_probs))
    winning_move = 3
    
    print(f"\nBest move chosen: Column {best_move}")
    print(f"Winning move: Column {winning_move}")
    print(f"Probability of winning move: {action_probs[winning_move]:.4f}")
    
    # Evaluate what happens if we play the winning move
    winning_state = game.get_next_state(neutral_state, winning_move, 1)
    is_win = game.check_win(winning_state, winning_move)
    
    print(f"\nVerification: Playing column {winning_move} wins = {is_win}")
    
    # Test neural network value prediction on winning position
    with torch.no_grad():
        encoded = game.get_encoded_state(neutral_state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value = model(tensor)
        value_pred = value.item()
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    
    print(f"\nNeural Network Predictions:")
    print(f"  Position value: {value_pred:+.4f} (should be positive if AI sees advantage)")
    print(f"  Policy for winning move (col {winning_move}): {policy_probs[winning_move]:.4f}")
    
    # Score the test
    if best_move == winning_move and action_probs[winning_move] > 0.7:
        print(f"\n✅ EXCELLENT: Strongly prefers winning move ({action_probs[winning_move]:.1%})")
        return "PASS"
    elif best_move == winning_move:
        print(f"\n⚠️  PARTIAL: Finds winning move but not confident ({action_probs[winning_move]:.1%})")
        return "PARTIAL"
    else:
        print(f"\n❌ FAIL: Misses winning move (chose column {best_move} instead)")
        return "FAIL"


def test_immediate_loss(model, game, iteration):
    """Test: AI must block opponent's immediate win"""
    print(f"\n{'='*70}")
    print(f"TEST 2: IMMEDIATE LOSS PREVENTION (Iteration {iteration})")
    print(f"{'='*70}")
    
    # Create a position where opponent (player -1) threatens to win
    state = game.get_initial_state()
    # Opponent has O O O _ . . . (will win at column 3 if not blocked)
    state = game.get_next_state(state, 0, 1)   # X plays column 0
    state = game.get_next_state(state, 1, -1)  # O plays column 1
    state = game.get_next_state(state, 0, 1)   # X plays column 0
    state = game.get_next_state(state, 2, -1)  # O plays column 2
    state = game.get_next_state(state, 4, 1)   # X plays column 4
    state = game.get_next_state(state, 3, -1)  # O plays column 3
    # Now O has 3 in a row at columns 1,2,3. Must block at column 0 OR column 4
    # Wait, let me fix this - need horizontal threat
    
    # Reset and create clearer threat
    state = game.get_initial_state()
    # O O O at bottom of columns 1,2,3, can win at 0 or 4
    state = game.get_next_state(state, 1, -1)  # O col 1
    state = game.get_next_state(state, 5, 1)   # X col 5
    state = game.get_next_state(state, 2, -1)  # O col 2
    state = game.get_next_state(state, 5, 1)   # X col 5
    state = game.get_next_state(state, 3, -1)  # O col 3 (O has 1,2,3)
    state = game.get_next_state(state, 6, 1)   # X col 6
    # O can win at column 0 or column 4
    
    print("\nBoard position (AI is X, opponent O threatens to win):")
    print_board(state)
    print("Opponent has 3 in a row at columns 1,2,3")
    print("Must block at column 0 or column 4!")
    
    # Test from AI's perspective (player 1)
    player = 1
    neutral_state = game.change_perspective(state, player)
    
    # Run MCTS
    args = {
        'num_searches': 100,
        'C': 2.0,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3
    }
    
    mcts = MCTS(game, args, model)
    action_probs = mcts.search(neutral_state, add_noise=False)
    
    print("\nMCTS Results:")
    print(f"Action probabilities:")
    for col in range(7):
        valid = game.get_valid_moves(neutral_state)[col]
        if valid:
            print(f"  Column {col}: {action_probs[col]:.4f}")
    
    best_move = int(np.argmax(action_probs))
    blocking_moves = [0, 4]
    
    print(f"\nBest move chosen: Column {best_move}")
    print(f"Blocking moves: Columns {blocking_moves}")
    print(f"Probability of blocking: col 0 = {action_probs[0]:.4f}, col 4 = {action_probs[4]:.4f}")
    
    # Test what happens if we DON'T block
    if best_move not in blocking_moves:
        bad_state = game.get_next_state(neutral_state, best_move, 1)
        bad_state_opp = game.change_perspective(bad_state, -1)
        # Can opponent win now?
        for col in blocking_moves:
            test_state = game.get_next_state(bad_state_opp, col, 1)
            if game.check_win(test_state, col):
                print(f"\n⚠️  Warning: If AI plays column {best_move}, opponent wins at column {col}!")
    
    # Test neural network value prediction
    with torch.no_grad():
        encoded = game.get_encoded_state(neutral_state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value = model(tensor)
        value_pred = value.item()
    
    print(f"\nNeural Network Predictions:")
    print(f"  Position value: {value_pred:+.4f} (should be negative - AI is in danger)")
    
    # Score the test
    blocking_prob = action_probs[0] + action_probs[4]
    if best_move in blocking_moves and max(action_probs[0], action_probs[4]) > 0.7:
        print(f"\n✅ EXCELLENT: Strongly blocks threat (prob={max(action_probs[0], action_probs[4]):.1%})")
        return "PASS"
    elif best_move in blocking_moves:
        print(f"\n⚠️  PARTIAL: Blocks but not confident (prob={max(action_probs[0], action_probs[4]):.1%})")
        return "PARTIAL"
    else:
        print(f"\n❌ FAIL: Doesn't block (chose column {best_move})")
        return "FAIL"


def test_value_predictions(model, game, iteration):
    """Test value predictions on various positions"""
    print(f"\n{'='*70}")
    print(f"TEST 3: VALUE PREDICTION ACCURACY (Iteration {iteration})")
    print(f"{'='*70}")
    
    with torch.no_grad():
        # Test 1: Empty board
        state = game.get_initial_state()
        encoded = game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        _, value = model(tensor)
        empty_value = value.item()
        
        print(f"\nEmpty board value: {empty_value:+.4f}")
        print(f"Expected: ~0.0 (balanced position)")
        
        # Test 2: Winning position
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, 1)
        state = game.get_next_state(state, 0, -1)
        state = game.get_next_state(state, 1, 1)
        state = game.get_next_state(state, 1, -1)
        state = game.get_next_state(state, 2, 1)
        state = game.get_next_state(state, 2, -1)
        state = game.get_next_state(state, 3, 1)  # X wins!
        
        is_win = game.check_win(state, 3)
        encoded = game.get_encoded_state(state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        _, value = model(tensor)
        win_value = value.item()
        
        print(f"\nWinning position value: {win_value:+.4f}")
        print(f"Expected: +1.0 (AI won)")
        print(f"Actual win: {is_win}")
        
        # Test 3: Losing position
        state = game.get_initial_state()
        state = game.get_next_state(state, 0, -1)
        state = game.get_next_state(state, 4, 1)
        state = game.get_next_state(state, 1, -1)
        state = game.get_next_state(state, 4, 1)
        state = game.get_next_state(state, 2, -1)
        state = game.get_next_state(state, 4, 1)
        state = game.get_next_state(state, 3, -1)  # O wins!
        
        # From X's perspective (lost)
        neutral_state = game.change_perspective(state, 1)
        encoded = game.get_encoded_state(neutral_state)
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        _, value = model(tensor)
        loss_value = value.item()
        
        print(f"\nLosing position value: {loss_value:+.4f}")
        print(f"Expected: -1.0 (AI lost)")
        
        # Summary
        print(f"\n{'='*40}")
        print("Value Prediction Summary:")
        print(f"  Empty:  {empty_value:+.4f} (target: ~0.0)")
        print(f"  Win:    {win_value:+.4f} (target: +1.0)")
        print(f"  Loss:   {loss_value:+.4f} (target: -1.0)")
        
        if abs(empty_value) < 0.3:
            print("\n✅ Empty board: Good (balanced)")
        else:
            print(f"\n⚠️  Empty board: Biased ({empty_value:+.4f})")


def main():
    # Find latest model
    import glob
    models = glob.glob("checkpoints/connect4/model_*.pt")
    if not models:
        print("No models found!")
        return
    
    latest_iteration = max([int(Path(m).stem.split('_')[1]) for m in models])
    
    model, game = load_model(latest_iteration)
    if model is None:
        return
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE MODEL EVALUATION - ITERATION {latest_iteration}")
    print(f"{'='*70}")
    print("\nTesting model's tactical awareness with fixed MCTS...")
    
    result1 = test_immediate_win(model, game, latest_iteration)
    result2 = test_immediate_loss(model, game, latest_iteration)
    test_value_predictions(model, game, latest_iteration)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Iteration: {latest_iteration}")
    print(f"  Immediate Win Detection: {result1}")
    print(f"  Immediate Loss Prevention: {result2}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
