#!/usr/bin/env python3
"""
Alpha-Beta (WDL) vs MCTS Tournament.

Rigorous comparison of alpha-beta search with WDL evaluation
against MCTS baseline over 100 games with time controls.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from time import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.engine.alphabeta import AlphaBetaEngine
from alpha_zero_light.engine.wdl_evaluator import WDLEvaluator
from alpha_zero_light.mcts.mcts import MCTS


def play_game(game, player1, player2, player1_name, player2_name, time_limit_ms=1000, verbose=False):
    """
    Play one game between two players.
    
    Returns:
        (result, num_moves, times): 
            result: 1 if player1 wins, -1 if player2 wins, 0 for draw
            num_moves: Total moves played
            times: (p1_time_ms, p2_time_ms)
    """
    state = game.get_initial_state()
    current_player = 1
    move_count = 0
    max_moves = 42
    
    p1_time = 0
    p2_time = 0
    
    while move_count < max_moves:
        if current_player == 1:
            player = player1
            player_name = player1_name
        else:
            player = player2
            player_name = player2_name
        
        # Get move with timing
        start = time() * 1000
        
        if isinstance(player, AlphaBetaEngine):
            result = player.search(state, current_player, time_limit_ms=time_limit_ms)
            action = result.best_move
            if verbose:
                print(f"  {player_name}: col {action} (depth {result.depth_reached}, {result.nodes_searched} nodes)")
        else:  # MCTS
            ai_state = game.change_perspective(state.copy(), current_player)
            action_probs = player.search(ai_state, add_noise=False, temperature=0)
            
            # Mask invalid moves
            valid_moves = game.get_valid_moves(state)
            action_probs *= valid_moves
            
            # Greedy selection
            action = np.argmax(action_probs)
            if verbose:
                print(f"  {player_name}: col {action} (P={action_probs[action]:.3f})")
        
        elapsed = time() * 1000 - start
        if current_player == 1:
            p1_time += elapsed
        else:
            p2_time += elapsed
        
        # Make move
        state = game.get_next_state(state, action, current_player)
        move_count += 1
        
        # Check terminal
        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            if value == 1:
                result = 1 if current_player == 1 else -1
            else:
                result = 0
            return result, move_count, (p1_time, p2_time)
        
        current_player = game.get_opponent(current_player)
    
    # Draw by max moves
    return 0, move_count, (p1_time, p2_time)


def evaluate_tactical_puzzles(game, player, player_name, device):
    """
    Test player on tactical puzzles.
    
    Returns:
        (correct, total): Number of correct puzzles
    """
    print(f"\n{'='*70}")
    print(f"TACTICAL PUZZLES: {player_name}")
    print(f"{'='*70}")
    
    correct = 0
    total = 0
    
    # Puzzle 1: Win in 1 (horizontal)
    total += 1
    state1 = game.get_initial_state()
    state1[5, 0] = 1  # X
    state1[5, 1] = 1  # X
    state1[5, 2] = 1  # X
    state1[5, 4] = -1  # O
    state1[5, 5] = -1  # O
    # Column 3 wins for player 1
    
    if isinstance(player, AlphaBetaEngine):
        result = player.search(state1, 1, time_limit_ms=2000)
        move = result.best_move
    else:  # MCTS
        action_probs = player.search(state1, add_noise=False, temperature=0)
        move = np.argmax(action_probs)
    
    if move == 3:
        correct += 1
        print(f"  \u2713 Puzzle 1 (Win in 1): Correct (col {move})")
    else:
        print(f"  \u2717 Puzzle 1 (Win in 1): Wrong (col {move}, expected 3)")
    
    # Puzzle 2: Block win (vertical)
    total += 1
    state2 = game.get_initial_state()
    state2[5, 0] = -1  # O
    state2[4, 0] = -1  # O
    state2[3, 0] = -1  # O
    # Must block at column 0
    
    if isinstance(player, AlphaBetaEngine):
        result = player.search(state2, 1, time_limit_ms=2000)
        move = result.best_move
    else:
        action_probs = player.search(state2, add_noise=False, temperature=0)
        move = np.argmax(action_probs)
    
    if move == 0:
        correct += 1
        print(f"  \u2713 Puzzle 2 (Block): Correct (col {move})")
    else:
        print(f"  \u2717 Puzzle 2 (Block): Wrong (col {move}, expected 0)")
    
    # Puzzle 3: Win in 1 (diagonal)
    total += 1
    state3 = game.get_initial_state()
    state3[5, 3] = 1  # X
    state3[4, 4] = 1  # X
    state3[3, 5] = 1  # X
    state3[5, 2] = -1  # Filler
    state3[4, 3] = -1
    state3[3, 4] = -1
    # Column 6 row 2 wins diagonal
    # Need to stack to row 2
    state3[5, 6] = -1
    state3[4, 6] = -1
    state3[3, 6] = -1
    # Actually, let's make this simpler - center column threat
    
    # Simplified puzzle 3: Another horizontal
    state3 = game.get_initial_state()
    state3[5, 1] = 1
    state3[5, 3] = 1
    state3[5, 4] = 1
    # Column 2 wins
    
    if isinstance(player, AlphaBetaEngine):
        result = player.search(state3, 1, time_limit_ms=2000)
        move = result.best_move
    else:
        action_probs = player.search(state3, add_noise=False, temperature=0)
        move = np.argmax(action_probs)
    
    if move == 2:
        correct += 1
        print(f"  \u2713 Puzzle 3 (Win in 1 v2): Correct (col {move})")
    else:
        print(f"  \u2717 Puzzle 3 (Win in 1 v2): Wrong (col {move}, expected 2)")
    
    print(f"\nTactical Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    return correct, total


def run_tournament(num_games=100, time_per_move_ms=1000):
    """Run complete tournament between alpha-beta and MCTS."""
    
    print("="*70)
    print(" "*15 + "ALPHA-BETA (WDL) vs MCTS TOURNAMENT")
    print("="*70)
    
    # Setup
    game = ConnectFour()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load WDL evaluator for alpha-beta
    wdl_model_path = Path('checkpoints/connect4/model_wdl_best.pt')
    if not wdl_model_path.exists():
        print(f"\n\u26a0 WDL model not found: {wdl_model_path}")
        print("Run training first: python training/scripts/train_supervised.py")
        return
    
    print(f"\nLoading Alpha-Beta engine with WDL evaluator...")
    wdl_eval = WDLEvaluator(wdl_model_path, game, device)
    alpha_beta = AlphaBetaEngine(
        game=game,
        evaluator=wdl_eval.evaluate,
        tt_size_mb=256,
        use_killer_moves=True,
        use_history_heuristic=True
    )
    print(f"\u2713 Alpha-Beta engine ready")
    
    # Load MCTS baseline
    print(f"\nLoading MCTS baseline (model_195.pt)...")
    mcts_model = ResNet(game, num_res_blocks=20, num_hidden=256).to(device)
    mcts_model.load_state_dict(torch.load('checkpoints/connect4/model_195.pt', map_location=device))
    mcts_model.eval()
    
    mcts_args = {
        'C': 2.0,
        'num_searches': 400,
        'dirichlet_epsilon': 0.0,  # No noise for deterministic play
        'dirichlet_alpha': 1.0,
        'mcts_batch_size': 1
    }
    mcts = MCTS(game, mcts_args, mcts_model)
    print(f"\u2713 MCTS engine ready")
    
    # Tactical puzzles first
    ab_correct, ab_total = evaluate_tactical_puzzles(game, alpha_beta, "Alpha-Beta", device)
    mcts_correct, mcts_total = evaluate_tactical_puzzles(game, mcts, "MCTS", device)
    
    # Tournament
    print(f"\n{'='*70}")
    print(f"STARTING TOURNAMENT: {num_games} games, {time_per_move_ms}ms/move")
    print(f"{'='*70}\n")
    
    ab_wins = 0
    mcts_wins = 0
    draws = 0
    
    ab_times = []
    mcts_times = []
    move_counts = []
    
    # Half games: Alpha-Beta starts
    for i in range(num_games // 2):
        print(f"Game {i+1}/{num_games} (Alpha-Beta starts)...", end="", flush=True)
        result, num_moves, (p1_time, p2_time) = play_game(
            game, alpha_beta, mcts, "AB", "MCTS", time_per_move_ms, verbose=False
        )
        
        ab_times.append(p1_time)
        mcts_times.append(p2_time)
        move_counts.append(num_moves)
        
        if result == 1:
            ab_wins += 1
            print(" Alpha-Beta wins!")
        elif result == -1:
            mcts_wins += 1
            print(" MCTS wins!")
        else:
            draws += 1
            print(" Draw")
    
    # Half games: MCTS starts
    for i in range(num_games // 2, num_games):
        print(f"Game {i+1}/{num_games} (MCTS starts)...", end="", flush=True)
        result, num_moves, (p1_time, p2_time) = play_game(
            game, mcts, alpha_beta, "MCTS", "AB", time_per_move_ms, verbose=False
        )
        
        mcts_times.append(p1_time)
        ab_times.append(p2_time)
        move_counts.append(num_moves)
        
        if result == 1:
            mcts_wins += 1
            print(" MCTS wins!")
        elif result == -1:
            ab_wins += 1
            print(" Alpha-Beta wins!")
        else:
            draws += 1
            print(" Draw")
    
    # Final report
    print(f"\n{'='*70}")
    print(" "*25 + "FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nGame Results:")
    print(f"  Alpha-Beta:  {ab_wins} wins, {draws} draws, {mcts_wins} losses")
    print(f"  MCTS:        {mcts_wins} wins, {draws} draws, {ab_wins} losses")
    print(f"  Win Rate:    AB={ab_wins/num_games*100:.1f}%, MCTS={mcts_wins/num_games*100:.1f}%")
    
    print(f"\nTactical Accuracy:")
    print(f"  Alpha-Beta:  {ab_correct}/{ab_total} ({ab_correct/ab_total*100:.1f}%)")
    print(f"  MCTS:        {mcts_correct}/{mcts_total} ({mcts_correct/mcts_total*100:.1f}%)")
    
    print(f"\nPerformance:")
    print(f"  Avg game length: {np.mean(move_counts):.1f} moves")
    print(f"  AB avg time/move: {np.mean(ab_times)/np.mean(move_counts):.1f}ms")
    print(f"  MCTS avg time/move: {np.mean(mcts_times)/np.mean(move_counts):.1f}ms")
    
    print(f"\n{'='*70}")
    print("TOURNAMENT COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha-Beta vs MCTS tournament')
    parser.add_argument('--games', type=int, default=100, help='Number of games')
    parser.add_argument('--time', type=int, default=1000, help='Time limit per move (ms)')
    args = parser.parse_args()
    
    run_tournament(args.games, args.time)
