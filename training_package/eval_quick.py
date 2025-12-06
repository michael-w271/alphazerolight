#!/usr/bin/env python3
"""
Quick evaluation: Test latest model against random player
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_zero_light.game.gomoku_9x9 import Gomoku9x9
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
import torch
import numpy as np

def play_game(game, mcts_player, random_player=True):
    """Play one game, return 1 if AI wins, -1 if loses, 0 if draw"""
    state = game.get_initial_state()
    player = 1  # AI starts
    
    for _ in range(81):  # Max moves
        if player == 1:
            # AI move
            action_probs = mcts_player.search(state)
            valid_moves = game.get_valid_moves(state)
            action_probs *= valid_moves
            action = np.argmax(action_probs)
        else:
            # Random opponent
            valid_moves = game.get_valid_moves(state)
            valid_actions = np.where(valid_moves == 1)[0]
            if len(valid_actions) == 0:
                return 0  # Draw
            action = np.random.choice(valid_actions)
        
        state = game.get_next_state(state, action, player)
        
        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            return value if player == 1 else -value
        
        player = game.get_opponent(player)
    
    return 0  # Draw

def main():
    checkpoint_dir = Path("checkpoints/gomoku_large")
    checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    latest = checkpoints[-1]
    print(f"Testing: {latest.name}")
    print("="*50)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = Gomoku9x9()
    model = ResNet(game, num_res_blocks=15, num_hidden=512).to(device)
    model.load_state_dict(torch.load(latest, map_location=device))
    model.eval()
    
    # MCTS with moderate searches for eval
    mcts = MCTS(game, {'C': 2, 'num_searches': 50}, model)
    
    # Play 20 games
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(20):
        result = play_game(game, mcts)
        if result == 1:
            wins += 1
            print(f"Game {i+1}: AI Won ✓")
        elif result == -1:
            losses += 1
            print(f"Game {i+1}: AI Lost ✗")
        else:
            draws += 1
            print(f"Game {i+1}: Draw -")
    
    print("="*50)
    print(f"Results: {wins}W - {losses}L - {draws}D")
    print(f"Win Rate: {wins/20*100:.1f}%")
    
    if wins > 15:
        print("🎉 Excellent! AI is very strong")
    elif wins > 10:
        print("✅ Good! AI is learning")
    elif wins > 5:
        print("⚠️  Okay, still learning")
    else:
        print("❌ Poor, needs more training")

if __name__ == "__main__":
    main()
