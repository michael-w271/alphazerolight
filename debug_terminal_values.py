#!/usr/bin/env python3
"""Debug why winning positions get negative values"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet


def test_terminal_value_encoding():
    game = ConnectFour()
    
    # Load latest model
    model = ResNet(game, num_res_blocks=10, num_hidden=128)
    model.load_state_dict(torch.load("checkpoints/connect4/model_9.pt", map_location='cpu'))
    model.eval()
    
    print("="*70)
    print("DEBUGGING TERMINAL STATE VALUE PREDICTIONS")
    print("="*70)
    
    # Create a winning position for player 1
    state = game.get_initial_state()
    state = game.get_next_state(state, 0, 1)   # X
    state = game.get_next_state(state, 0, -1)  # O
    state = game.get_next_state(state, 1, 1)   # X
    state = game.get_next_state(state, 1, -1)  # O
    state = game.get_next_state(state, 2, 1)   # X
    state = game.get_next_state(state, 2, -1)  # O
    state = game.get_next_state(state, 3, 1)   # X wins!
    
    print("\nBoard (player 1 won):")
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            print("X " if val == 1 else ("O " if val == -1 else ". "), end="")
        print()
    
    is_win = game.check_win(state, 3)
    value, is_terminal = game.get_value_and_terminated(state, 3)
    
    print(f"\nGame state verification:")
    print(f"  check_win(state, 3) = {is_win}")
    print(f"  get_value_and_terminated(state, 3) = value={value}, terminal={is_terminal}")
    print(f"  Expected: value=1 (player 1 won from their perspective)")
    
    # Now encode from player 1's perspective
    print(f"\n{'='*70}")
    print("Testing from Player 1's perspective (the winner):")
    print(f"{'='*70}")
    
    player = 1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nAfter change_perspective(state, {player}):")
    print("  Bottom row:")
    print(f"  Raw state: {state[5, :]}")
    print(f"  Neutral:   {neutral_state[5, :]}")
    print(f"  (AI's pieces should be 1)")
    
    encoded = game.get_encoded_state(neutral_state)
    print(f"\nEncoded state shape: {encoded.shape}")
    print(f"  Channel 0 (opponent): {encoded[0].sum():.0f} cells")
    print(f"  Channel 1 (empty):    {encoded[1].sum():.0f} cells")
    print(f"  Channel 2 (AI):       {encoded[2].sum():.0f} cells")
    
    # Get model prediction
    with torch.no_grad():
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value_pred = model(tensor)
        value_pred = value_pred.item()
    
    print(f"\nModel prediction: {value_pred:+.4f}")
    print(f"Expected: +1.0 (AI won)")
    
    if value_pred < 0:
        print(f"\n❌ BUG: Model thinks AI LOST when AI actually WON!")
    
    # Now test from player -1's perspective (the loser)
    print(f"\n{'='*70}")
    print("Testing from Player -1's perspective (the loser):")
    print(f"{'='*70}")
    
    player = -1
    neutral_state = game.change_perspective(state, player)
    
    print(f"\nAfter change_perspective(state, {player}):")
    print("  Bottom row:")
    print(f"  Raw state: {state[5, :]}")
    print(f"  Neutral:   {neutral_state[5, :]}")
    print(f"  (AI's pieces should be 1, but AI is player -1, so opponent's X should be -1)")
    
    encoded = game.get_encoded_state(neutral_state)
    
    with torch.no_grad():
        tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value_pred = model(tensor)
        value_pred = value_pred.item()
    
    print(f"\nModel prediction: {value_pred:+.4f}")
    print(f"Expected: -1.0 (AI lost because opponent won)")
    
    # Check what the training data would have been
    print(f"\n{'='*70}")
    print("Training Data Analysis:")
    print(f"{'='*70}")
    
    print("\nIn training, when player 1 wins:")
    print("  - Player 1's moves get outcome = +1")
    print("  - Player -1's moves get outcome = -1")
    print("\nIn inference, the model sees positions from 'current player' perspective")
    print("  - Current player's pieces are always encoded as channel 2 (+1)")
    print("  - If current player is about to lose, value should be negative")
    print("  - If current player won, value should be positive")
    
    print("\nPOTENTIAL ISSUE:")
    print("  When we evaluate a terminal winning state,")
    print("  are we evaluating it from the WINNER's perspective")
    print("  or from the NEXT player's perspective (who doesn't exist)?")


if __name__ == "__main__":
    test_terminal_value_encoding()
