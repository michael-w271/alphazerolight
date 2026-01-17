#!/usr/bin/env python3
"""Test Connect Four models from different iterations"""

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


def play_game(model, game, model_plays_first=True, num_searches=50):
    """Play a game against the model"""
    args = {
        'num_searches': num_searches,
        'c_puct': 2.0,
        'dirichlet_epsilon': 0.0,  # No exploration noise for testing
        'dirichlet_alpha': 0.3
    }
    
    mcts = MCTS(game, args, model)
    state = game.get_initial_state()
    player = 1
    
    print("\nGame Start!")
    print("=" * 50)
    
    move_count = 0
    while True:
        move_count += 1
        print(f"\nMove {move_count}")
        print(game.get_display_state(state))
        
        # Determine who plays
        human_turn = (player == 1 and not model_plays_first) or (player == -1 and model_plays_first)
        
        if human_turn:
            # Human move
            valid_moves = game.get_valid_moves(state)
            print(f"Valid moves: {[i for i, v in enumerate(valid_moves) if v == 1]}")
            
            while True:
                try:
                    action = int(input(f"Player {player}, enter column (0-6): "))
                    if valid_moves[action] == 1:
                        break
                    print("Invalid move! Try again.")
                except (ValueError, IndexError):
                    print("Invalid input! Try again.")
        else:
            # AI move
            print(f"AI (Player {player}) is thinking...")
            neutral_state = game.change_perspective(state, player)
            action_probs = mcts.search(neutral_state)
            action = int(np.argmax(action_probs))
            print(f"AI plays column {action}")
            print(f"Move probabilities: {[f'{p:.3f}' for p in action_probs]}")
        
        # Make move
        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            print("\nFinal Board:")
            print(game.get_display_state(state))
            
            if value == 1:
                winner = player
                print(f"\n{'AI' if not human_turn else 'Human'} (Player {winner}) wins!")
            else:
                print("\nDraw!")
            
            return value, player
        
        player = game.get_opponent(player)


def compare_models():
    """Compare iteration 13 vs iteration 189"""
    print("=" * 70)
    print("AlphaZero Connect Four - Model Comparison")
    print("=" * 70)
    
    game = ConnectFour(row_count=6, column_count=7, win_length=4)
    
    print("\nLoading models...")
    model_13 = load_model(13, game)
    model_189 = load_model(189, game)
    print("✓ Models loaded!")
    
    print("\n" + "=" * 70)
    print("Test 1: Playing against Model from Iteration 13 (Best Model)")
    print("Loss: 0.7939 (Policy: 0.7276, Value: 0.0332)")
    print("=" * 70)
    
    play_game(model_13, game, model_plays_first=False)
    
    print("\n" + "=" * 70)
    print("Test 2: Playing against Model from Iteration 189 (Final Model)")
    print("Loss: 1.4329 (Policy: 0.7237, Value: 0.3546)")
    print("=" * 70)
    
    play_game(model_189, game, model_plays_first=False)


if __name__ == "__main__":
    compare_models()
