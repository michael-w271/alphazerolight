#!/usr/bin/env python3
"""
Tournament script to find the best Connect4 model.
Plays multiple models against each other.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from itertools import combinations
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS

def load_model(checkpoint_path, game, device):
    """Load a model from checkpoint"""
    model = ResNet(game, num_res_blocks=10, num_hidden=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def play_game(game, mcts1, mcts2, verbose=False):
    """
    Play one game between two models.
    Returns 1 if mcts1 wins, -1 if mcts2 wins, 0 for draw
    """
    board = game.get_initial_state()
    current_player = 1
    move_count = 0
    max_moves = 42  # Connect4 board size
    
    while move_count < max_moves:
        # Current player's MCTS
        if current_player == 1:
            mcts = mcts1
        else:
            mcts = mcts2
        
        # Get AI's perspective
        ai_state = game.change_perspective(board.copy(), player=current_player)
        action_probs = mcts.search(ai_state)
        
        # Mask invalid moves
        valid_moves = game.get_valid_moves(board)
        action_probs *= valid_moves
        
        if np.sum(valid_moves) == 0:
            # No valid moves - draw
            return 0
        
        action = np.argmax(action_probs)
        board = game.get_next_state(board, action, current_player)
        move_count += 1
        
        # Check terminal
        value, is_terminal = game.get_value_and_terminated(board, action)
        if is_terminal:
            if value == 1:
                # Current player won
                return 1 if current_player == 1 else -1
            else:
                return 0
        
        current_player = game.get_opponent(current_player)
    
    return 0  # Draw

def run_tournament(models_to_test, games_per_matchup=5):
    """Run a round-robin tournament"""
    print("=" * 80)
    print("🏆 Connect4 Model Tournament")
    print("=" * 80)
    
    checkpoint_dir = Path("/mnt/ssd2pro/alpha-zero-checkpoints/connect4")
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load all models
    print("Loading models...")
    models_dict = {}
    mcts_dict = {}
    
    args = {
        'C': 2.0,
        'num_searches': 100,  # Faster for tournament
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3,
    }
    
    for iter_num in models_to_test:
        checkpoint = checkpoint_dir / f"model_{iter_num}.pt"
        if not checkpoint.exists():
            print(f"⚠️  Model {iter_num} not found, skipping")
            continue
        
        print(f"  Loading model_{iter_num}.pt...")
        model = load_model(checkpoint, game, device)
        models_dict[iter_num] = model
        mcts_dict[iter_num] = MCTS(game, args, model)
    
    print(f"\n✅ Loaded {len(models_dict)} models\n")
    
    # Initialize scores
    scores = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'points': 0})
    
    # Run tournament
    matchups = list(combinations(sorted(models_dict.keys()), 2))
    total_games = len(matchups) * games_per_matchup * 2  # Each matchup plays both sides
    game_count = 0
    
    print(f"Running {len(matchups)} matchups, {games_per_matchup} games per side")
    print("=" * 80)
    
    for iter1, iter2 in matchups:
        print(f"\n📊 Model {iter1} vs Model {iter2}")
        
        m1_wins = 0
        m2_wins = 0
        draws = 0
        
        # Play games with iter1 as player 1
        for game_num in range(games_per_matchup):
            result = play_game(game, mcts_dict[iter1], mcts_dict[iter2])
            game_count += 1
            
            if result == 1:
                m1_wins += 1
                scores[iter1]['wins'] += 1
                scores[iter1]['points'] += 3
                scores[iter2]['losses'] += 1
            elif result == -1:
                m2_wins += 1
                scores[iter2]['wins'] += 1
                scores[iter2]['points'] += 3
                scores[iter1]['losses'] += 1
            else:
                draws += 1
                scores[iter1]['draws'] += 1
                scores[iter2]['draws'] += 1
                scores[iter1]['points'] += 1
                scores[iter2]['points'] += 1
            
            print(f"  Game {game_num + 1}/{games_per_matchup} (as P1): ", end="")
            if result == 1:
                print(f"Model {iter1} wins")
            elif result == -1:
                print(f"Model {iter2} wins")
            else:
                print("Draw")
        
        # Play games with iter2 as player 1
        for game_num in range(games_per_matchup):
            result = play_game(game, mcts_dict[iter2], mcts_dict[iter1])
            game_count += 1
            
            if result == 1:
                m2_wins += 1
                scores[iter2]['wins'] += 1
                scores[iter2]['points'] += 3
                scores[iter1]['losses'] += 1
            elif result == -1:
                m1_wins += 1
                scores[iter1]['wins'] += 1
                scores[iter1]['points'] += 3
                scores[iter2]['losses'] += 1
            else:
                draws += 1
                scores[iter1]['draws'] += 1
                scores[iter2]['draws'] += 1
                scores[iter1]['points'] += 1
                scores[iter2]['points'] += 1
            
            print(f"  Game {game_num + 1}/{games_per_matchup} (as P2): ", end="")
            if result == 1:
                print(f"Model {iter2} wins")
            elif result == -1:
                print(f"Model {iter1} wins")
            else:
                print("Draw")
        
        total = games_per_matchup * 2
        print(f"  Result: Model {iter1}: {m1_wins}/{total}  |  Model {iter2}: {m2_wins}/{total}  |  Draws: {draws}/{total}")
        print(f"  Progress: {game_count}/{total_games} games completed")
    
    # Print final standings
    print("\n" + "=" * 80)
    print("🏆 FINAL STANDINGS")
    print("=" * 80)
    
    # Sort by points
    sorted_models = sorted(scores.items(), key=lambda x: x[1]['points'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<10} {'Wins':<6} {'Draws':<6} {'Losses':<8} {'Points':<8} {'Win Rate':<10}")
    print("-" * 80)
    
    for rank, (iter_num, stats) in enumerate(sorted_models, 1):
        total = stats['wins'] + stats['draws'] + stats['losses']
        win_rate = stats['wins'] / total * 100 if total > 0 else 0
        
        print(f"{rank:<6} {f'model_{iter_num}':<10} {stats['wins']:<6} {stats['draws']:<6} "
              f"{stats['losses']:<8} {stats['points']:<8} {win_rate:.1f}%")
    
    print("\n" + "=" * 80)
    winner = sorted_models[0][0]
    print(f"🥇 CHAMPION: model_{winner}.pt")
    print(f"   Stats: {scores[winner]['wins']}W-{scores[winner]['draws']}D-{scores[winner]['losses']}L "
          f"({scores[winner]['points']} points)")
    print("=" * 80)
    
    return winner

if __name__ == "__main__":
    # Test these models (pick key iterations)
    models_to_test = [100, 110, 114, 120, 125, 127]
    
    print("\nModels to test:", [f"model_{i}.pt" for i in models_to_test])
    print("Scoring: Win = 3 points, Draw = 1 point, Loss = 0 points")
    print("Games per matchup: 5 games each side (10 total per matchup)\n")
    
    winner = run_tournament(models_to_test, games_per_matchup=5)
    
    print(f"\n✅ Use model_{winner}.pt for the best performance!")
