#!/usr/bin/env python3
"""
Debug MCTS to understand why it's not blocking threats.
Tests the MCTS logic step-by-step.
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
    model = ResNet(game, num_res_blocks=10, num_hidden=128)
    model.load_state_dict(torch.load(latest_model, map_location='cpu'))
    model.eval()
    
    return model, iteration


def print_board(state):
    """Print the board in a readable format"""
    print("\n  0 1 2 3 4 5 6")
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


def test_blocking_scenario(model, game, iteration):
    """Test why model doesn't block"""
    print(f"\n{'='*70}")
    print(f"MCTS BLOCKING DEBUG - Iteration {iteration}")
    print(f"{'='*70}")
    
    # Create a blocking scenario:
    # Opponent (O = -1) has 3 in column 5
    # Current player (X = 1) MUST block at column 5
    state = game.get_initial_state()
    state = game.get_next_state(state, 0, 1)   # X plays column 0
    state = game.get_next_state(state, 5, -1)  # O plays column 5
    state = game.get_next_state(state, 1, 1)   # X plays column 1
    state = game.get_next_state(state, 5, -1)  # O plays column 5
    state = game.get_next_state(state, 2, 1)   # X plays column 2
    state = game.get_next_state(state, 5, -1)  # O plays column 5
    
    print("Current Board (X = current player, O = opponent with 3 in col 5):")
    print_board(state)
    
    # Now it's X's turn from perspective of player 1
    neutral_state = game.change_perspective(state, 1)
    
    print("What happens if we DON'T block (e.g., play column 3)?")
    test_state = game.get_next_state(neutral_state, 3, 1)
    test_state_opp = game.change_perspective(test_state, -1)
    
    # Opponent can now win
    winning_state = game.get_next_state(test_state_opp, 5, 1)
    is_win = game.check_win(winning_state, 5)
    print(f"  After X plays col 3, O can win at col 5: {is_win}")
    
    print("\nWhat happens if we DO block (play column 5)?")
    block_state = game.get_next_state(neutral_state, 5, 1)
    is_win_blocked = game.check_win(block_state, 5)
    print(f"  After X blocks at col 5: X wins = {is_win_blocked}")
    
    # Now run MCTS with detailed debug
    print(f"\n{'='*70}")
    print("Running MCTS with DEBUG enabled...")
    print(f"{'='*70}")
    
    os.environ['DEBUG_MCTS'] = '1'
    
    args = {
        'num_searches': 100,
        'C': 2.0,
        'dirichlet_epsilon': 0.0,  # No noise for clearer analysis
        'dirichlet_alpha': 0.3
    }
    
    mcts = MCTS(game, args, model)
    action_probs = mcts.search(neutral_state, add_noise=False)
    
    print(f"\n{'='*70}")
    print("MCTS Results:")
    print(f"{'='*70}")
    
    for col in range(7):
        print(f"Column {col}: {action_probs[col]:.4f}")
    
    best_move = int(np.argmax(action_probs))
    print(f"\nBest move: Column {best_move}")
    print(f"Should be: Column 5 (blocking move)")
    
    if best_move == 5:
        print("✅ PASS - Model correctly blocks!")
    else:
        print("❌ FAIL - Model doesn't block threat")
        
        # Analyze why
        print("\n" + "="*70)
        print("ANALYSIS: Why didn't MCTS choose column 5?")
        print("="*70)
        
        # Check what value the model assigns to each move
        print("\nEvaluating each possible move:")
        for col in [0, 3, 5]:  # Sample a few columns
            if neutral_state[0, col] != 0:  # Skip if column full
                continue
                
            test_state = game.get_next_state(neutral_state, col, 1)
            test_state_opp = game.change_perspective(test_state, -1)
            
            # Get model's value prediction
            encoded = game.get_encoded_state(test_state_opp)
            tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = model(tensor)
                value_pred = value.item()
            
            # Check if terminal
            value_term, is_term = game.get_value_and_terminated(test_state_opp, col)
            
            print(f"\nColumn {col}:")
            print(f"  Model value (opponent's perspective): {value_pred:+.4f}")
            print(f"  Terminal: {is_term}, Terminal value: {value_term}")
            print(f"  Visit probability: {action_probs[col]:.4f}")
            
            # If opponent plays optimally next
            if not is_term:
                # What's opponent's best move?
                valid = game.get_valid_moves(test_state_opp)
                print(f"  Opponent has {np.sum(valid)} valid moves")
                
                # Check if opponent can win immediately
                for opp_col in range(7):
                    if valid[opp_col]:
                        opp_move_state = game.get_next_state(test_state_opp, opp_col, 1)
                        if game.check_win(opp_move_state, opp_col):
                            print(f"  ⚠️  Opponent can win immediately at column {opp_col}!")


def main():
    game = ConnectFour(row_count=6, column_count=7, win_length=4)
    
    model, iteration = load_latest_model(game)
    if model is None:
        return
    
    test_blocking_scenario(model, game, iteration)


if __name__ == "__main__":
    main()
