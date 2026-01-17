#!/usr/bin/env python3
"""
Compare two model iterations comprehensively.
Tests them on tactical scenarios, vs random, and head-to-head.
With fixes for: deterministic eval, legal positions, net-only tests, statistical significance.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import argparse
import math
from datetime import datetime

# Add src to path
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import MODEL_CONFIG, MCTS_CONFIG

class Tee:
    """Duplicate output to multiple file objects"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def setup_logging(iter1, iter2):
    """Setup logging to save output to timestamped file"""
    log_dir = base_dir / 'logs' / 'evaluations'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'compare_{iter1}_vs_{iter2}_{timestamp}.log'
    
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    
    print(f"📝 Logging to: {log_path}")
    return log_path, log_file

def immediate_winning_moves(game, state, player):
    """Return list of actions that win immediately for player from state"""
    valid = game.get_valid_moves(state)
    wins = []
    for a in np.where(valid)[0]:
        s2 = game.get_next_state(state, int(a), player)
        if game.check_win(s2, player):
            wins.append(int(a))
    return wins

def policy_probs_from_net(game, model, state, device='cpu'):
    """Return (policy_probs, value) from network for current state"""
    with torch.no_grad():
        enc = game.get_encoded_state(state)
        x = torch.tensor(enc, dtype=torch.float32, device=device).unsqueeze(0)
        p_logits, v = model(x)
        p = torch.softmax(p_logits, dim=1).squeeze(0).cpu().numpy()
        return p, float(v.item())

def new_tactics_stats():
    """Create new tactics stats container"""
    return {
        'turns_analyzed': 0,
        'had_immediate_win': 0,
        'took_immediate_win': 0,
        'had_to_block': 0,
        'did_block': 0,
        'avg_net_prob_on_win_moves': 0.0,
        'avg_mcts_prob_on_win_moves': 0.0,
        'avg_net_prob_on_blocks': 0.0,
        'avg_mcts_prob_on_blocks': 0.0,
        'n_win_prob_samples': 0,
        'n_block_prob_samples': 0
    }

def update_running_avg(stats, key, n_key, value):
    """Update running average in stats dict"""
    n = stats[n_key]
    stats[key] = (stats[key] * n + value) / (n + 1)
    stats[n_key] = n + 1

def print_tactics_summary(name, stats):
    """Print tactical in-game metrics summary"""
    def pct(a, b):
        return 0.0 if b == 0 else 100.0 * a / b
    
    print(f'\n--- Tactical In-Game Metrics: {name} ---')
    print(f"Turns analyzed: {stats['turns_analyzed']}")
    print(f"Immediate-win opportunities: {stats['had_immediate_win']} | conversion: {pct(stats['took_immediate_win'], stats['had_immediate_win']):.1f}%")
    print(f"Must-block situations: {stats['had_to_block']} | block rate: {pct(stats['did_block'], stats['had_to_block']):.1f}%")
    if stats['n_win_prob_samples'] > 0:
        print(f"Avg prob on chosen WIN moves: Net={stats['avg_net_prob_on_win_moves']:.3f}, MCTS={stats['avg_mcts_prob_on_win_moves']:.3f}")
    if stats['n_block_prob_samples'] > 0:
        print(f"Avg prob on chosen BLOCK moves: Net={stats['avg_net_prob_on_blocks']:.3f}, MCTS={stats['avg_mcts_prob_on_blocks']:.3f}")

def wilson_interval(successes, n, confidence=0.95):
    """Wilson score confidence interval for binomial proportion"""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return p, max(0, centre - margin), min(1, centre + margin)

def elo_diff_estimate(score):
    """Estimate Elo difference from score (wins + 0.5*draws)/games"""
    if score <= 0 or score >= 1:
        return float('inf') if score > 0.5 else float('-inf')
    return 400 * math.log10(score / (1 - score))

def state_from_moves(game, moves, starting_player=1):
    """Build a legal state by applying moves with alternating players"""
    state = game.get_initial_state()
    player = starting_player
    for action in moves:
        state = game.get_next_state(state, action, player)
        player = -player
    return state, player

def net_policy_value(model, game, state, device='cpu'):
    """Get network-only policy and value (no MCTS)"""
    with torch.no_grad():
        encoded = game.get_encoded_state(state)
        x = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
        policy_logits, value = model(x)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        return policy, float(value.item())

def load_model(iteration, game):
    """Load model from checkpoint"""
    checkpoint_path = base_dir / f"checkpoints/connect4/model_{iteration}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model {iteration} not found at {checkpoint_path}")
    
    model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden'])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def test_tactical_scenarios(model, mcts, game, model_name, device='cpu'):
    """Test model on tactical scenarios with LEGAL positions and net-only comparison"""
    print(f"\n{'='*60}")
    print(f"TACTICAL TESTS - {model_name}")
    print('='*60)
    
    results = {}
    
    # Test 1: Find immediate win (LEGAL state via moves)
    print("\n1. Finding immediate vertical win (column 3)...")
    # Moves: P1 plays 3, P2 plays 0, P1 plays 3, P2 plays 0, P1 plays 3, P2 plays 1
    # Result: P1 has 3 in column 3 (rows 5,4,3), P1 to move, can win with column 3
    state, player = state_from_moves(game, [3, 0, 3, 0, 3, 1], starting_player=1)
    
    # Network-only evaluation
    net_policy, net_value = net_policy_value(model, game, state, device)
    net_win_prob = net_policy[3]
    
    # MCTS evaluation
    action_probs = mcts.search(state)
    mcts_win_prob = action_probs[3]
    
    print(f"   Network: Win move (col 3) prob = {net_win_prob:.3f} {'✅' if net_win_prob > 0.3 else '❌'}")
    print(f"   MCTS:    Win move (col 3) prob = {mcts_win_prob:.3f} {'✅' if mcts_win_prob > 0.5 else '❌'}")
    results['find_win_net'] = net_win_prob
    results['find_win_mcts'] = mcts_win_prob
    
    # Test 2: Block opponent threat (LEGAL state via moves)
    print("\n2. Blocking opponent vertical threat (column 5)...")
    # Moves: P1 plays 0, P2 plays 5, P1 plays 1, P2 plays 5, P1 plays 2, P2 plays 5
    # Result: P2 has 3 in column 5 (rows 5,4,3), P1 to move, MUST block column 5
    state, player = state_from_moves(game, [0, 5, 1, 5, 2, 5], starting_player=1)
    
    # Network-only evaluation
    net_policy, net_value = net_policy_value(model, game, state, device)
    net_block_prob = net_policy[5]
    
    # MCTS evaluation
    action_probs = mcts.search(state)
    mcts_block_prob = action_probs[5]
    
    print(f"   Network: Block move (col 5) prob = {net_block_prob:.3f} {'✅' if net_block_prob > 0.2 else '❌'}")
    print(f"   MCTS:    Block move (col 5) prob = {mcts_block_prob:.3f} {'✅' if mcts_block_prob > 0.3 else '❌'}")
    results['block_threat_net'] = net_block_prob
    results['block_threat_mcts'] = mcts_block_prob
    
    # Test 3: Empty board value
    print("\n3. Empty board value (should be ~0)...")
    state = game.get_initial_state()
    net_policy, net_value = net_policy_value(model, game, state, device)
    print(f"   Network value: {net_value:.3f} {'✅ PASS' if abs(net_value) < 0.5 else '❌ FAIL'}")
    results['empty_board_value'] = net_value
    
    # Test 4: Center preference
    print("\n4. Center opening preference...")
    action_probs = mcts.search(state)
    center_prob = action_probs[3]
    net_center_prob = net_policy[3]
    print(f"   Network: Center (col 3) prob = {net_center_prob:.3f}")
    print(f"   MCTS:    Center (col 3) prob = {center_prob:.3f} {'✅ PASS' if center_prob > 0.15 else '❌ FAIL'}")
    results['center_preference'] = center_prob
    results['center_preference_net'] = net_center_prob
    
    return results

def play_game_vs_random(model, mcts, game, model_as_player1=True, analyze_tactics=False, block_ok_if_win=True, device='cpu'):
    """Play one game against random opponent with optional tactical analysis"""
    state = game.get_initial_state()
    player = 1
    last_action = None
    tactics_stats = new_tactics_stats() if analyze_tactics else None
    
    while True:
        # Check terminal AFTER a move was made
        if last_action is not None:
            value, is_terminal = game.get_value_and_terminated(state, last_action)
            if is_terminal:
                # value=1 means the player who just moved won
                # last_action was made by -player (we already flipped player)
                winner = -player if value == 1 else 0
                return (winner, tactics_stats) if analyze_tactics else winner
        
        valid_moves = game.get_valid_moves(state)
        if not np.any(valid_moves):
            return (0, tactics_stats) if analyze_tactics else 0  # Draw
        
        # Select action
        if (player == 1 and model_as_player1) or (player == -1 and not model_as_player1):
            # Model's turn - MUST use canonical perspective
            neutral_state = game.change_perspective(state, player)
            
            # --- Tactical analysis (model turn only) ---
            if analyze_tactics:
                tactics_stats['turns_analyzed'] += 1
                win_moves = immediate_winning_moves(game, state, player)
                opp_win_moves = immediate_winning_moves(game, state, -player)
                
                # Precompute network probs only if needed (special events)
                need_probs = (len(win_moves) > 0) or (len(opp_win_moves) > 0)
                net_p = None
                if need_probs:
                    net_p, _ = policy_probs_from_net(game, model, neutral_state, device)
                
                if len(win_moves) > 0:
                    tactics_stats['had_immediate_win'] += 1
                
                if len(opp_win_moves) > 0:
                    tactics_stats['had_to_block'] += 1
            
            # MCTS search
            action_probs = mcts.search(neutral_state)
            action_probs = action_probs * valid_moves
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
                action = np.argmax(action_probs)  # Deterministic (argmax)
            else:
                action = np.random.choice(np.where(valid_moves)[0])
            
            # --- Record tactical outcomes ---
            if analyze_tactics:
                # Immediate win conversion
                if len(win_moves) > 0 and action in win_moves:
                    tactics_stats['took_immediate_win'] += 1
                    if net_p is not None:
                        update_running_avg(tactics_stats, 'avg_net_prob_on_win_moves', 
                                         'n_win_prob_samples', float(net_p[action]))
                    mcts_prob = action_probs[action] if action_probs.sum() > 0 else 0.0
                    update_running_avg(tactics_stats, 'avg_mcts_prob_on_win_moves', 
                                     'n_win_prob_samples', float(mcts_prob))
                
                # Block rate (winning supersedes blocking if block_ok_if_win)
                if len(opp_win_moves) > 0:
                    blocked = action in opp_win_moves
                    # If we win instead of blocking, count as OK
                    if block_ok_if_win and len(win_moves) > 0 and action in win_moves:
                        blocked = True
                    
                    if blocked:
                        tactics_stats['did_block'] += 1
                        if net_p is not None:
                            update_running_avg(tactics_stats, 'avg_net_prob_on_blocks', 
                                             'n_block_prob_samples', float(net_p[action]))
                        mcts_prob = action_probs[action] if action_probs.sum() > 0 else 0.0
                        update_running_avg(tactics_stats, 'avg_mcts_prob_on_blocks', 
                                         'n_block_prob_samples', float(mcts_prob))
        else:
            # Random opponent
            action = np.random.choice(np.where(valid_moves)[0])
        
        state = game.get_next_state(state, action, player)
        last_action = action
        player = -player

def test_vs_random(model, mcts, game, model_name, num_games=100, analyze_tactics=True, device='cpu'):
    """Test model against random opponent with statistical confidence and tactical analysis"""
    print(f"\n{'='*60}")
    print(f"VS RANDOM - {model_name} ({num_games} games)")
    print('='*60)
    
    wins_as_p1 = 0
    wins_as_p2 = 0
    draws = 0
    
    # Aggregate tactics stats
    agg_tactics = new_tactics_stats() if analyze_tactics else None
    
    # Play as player 1
    print(f"\nPlaying as Player 1 ({num_games//2} games)...")
    for _ in tqdm(range(num_games // 2), ncols=60):
        if analyze_tactics:
            result, tactics = play_game_vs_random(model, mcts, game, model_as_player1=True, 
                                                 analyze_tactics=True, device=device)
            # Merge tactics stats
            for key in agg_tactics:
                if key.startswith('avg_'):
                    continue  # Handle separately
                elif key.startswith('n_'):
                    agg_tactics[key] += tactics[key]
                else:
                    agg_tactics[key] += tactics[key]
        else:
            result = play_game_vs_random(model, mcts, game, model_as_player1=True)
        
        if result == 1:  # Model won
            wins_as_p1 += 1
        elif result == 0:
            draws += 1
    
    # Play as player 2
    print(f"Playing as Player 2 ({num_games//2} games)...")
    for _ in tqdm(range(num_games // 2), ncols=60):
        if analyze_tactics:
            result, tactics = play_game_vs_random(model, mcts, game, model_as_player1=False, 
                                                 analyze_tactics=True, device=device)
            # Merge tactics stats
            for key in agg_tactics:
                if key.startswith('avg_'):
                    continue
                elif key.startswith('n_'):
                    agg_tactics[key] += tactics[key]
                else:
                    agg_tactics[key] += tactics[key]
        else:
            result = play_game_vs_random(model, mcts, game, model_as_player1=False)
        
        if result == -1:  # Model won
            wins_as_p2 += 1
        elif result == 0:
            draws += 1
    
    # Recalculate averages from aggregated samples
    if analyze_tactics and agg_tactics['n_win_prob_samples'] > 0:
        # Averages were computed incrementally, already correct
        pass
    
    total_wins = wins_as_p1 + wins_as_p2
    losses = num_games - total_wins - draws
    
    # Calculate confidence interval
    win_rate, ci_low, ci_high = wilson_interval(total_wins, num_games)
    
    print(f"\nResults: {total_wins}/{num_games} wins ({win_rate*100:.1f}%), {draws} draws, {losses} losses")
    print(f"95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    print(f"As P1: {wins_as_p1}/{num_games//2} wins, As P2: {wins_as_p2}/{num_games//2} wins")
    
    if analyze_tactics:
        print_tactics_summary(model_name, agg_tactics)
    
    return {
        'wins': total_wins,
        'draws': draws,
        'losses': losses,
        'win_rate': win_rate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'wins_as_p1': wins_as_p1,
        'wins_as_p2': wins_as_p2
    }

def test_vs_random_p1_only(model, mcts, game, model_name, num_games=10, analyze_tactics=True, device='cpu'):
    """Test model against random opponent (as P1 only) - faster evaluation"""
    print(f"\n{'='*60}")
    print(f"VS RANDOM (P1 ONLY) - {model_name} ({num_games} games)")
    print('='*60)
    
    wins = 0
    draws = 0
    
    # Aggregate tactics stats
    agg_tactics = new_tactics_stats() if analyze_tactics else None
    
    # Play as player 1 only
    print(f"\nPlaying as Player 1 ({num_games} games)...")
    for _ in tqdm(range(num_games), ncols=60):
        if analyze_tactics:
            result, tactics = play_game_vs_random(model, mcts, game, model_as_player1=True, 
                                                 analyze_tactics=True, device=device)
            # Merge tactics stats
            for key in agg_tactics:
                if key.startswith('avg_'):
                    continue
                elif key.startswith('n_'):
                    agg_tactics[key] += tactics[key]
                else:
                    agg_tactics[key] += tactics[key]
        else:
            result = play_game_vs_random(model, mcts, game, model_as_player1=True)
        
        if result == 1:  # Model won
            wins += 1
        elif result == 0:
            draws += 1
    
    losses = num_games - wins - draws
    
    # Calculate confidence interval
    win_rate, ci_low, ci_high = wilson_interval(wins, num_games)
    
    print(f"\nResults: {wins}/{num_games} wins ({win_rate*100:.1f}%), {draws} draws, {losses} losses")
    print(f"95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    
    if analyze_tactics:
        print_tactics_summary(model_name, agg_tactics)
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': win_rate,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

def play_head_to_head(model1, mcts1, model2, mcts2, game, model1_as_p1=True, 
                      analyze_tactics=False, block_ok_if_win=True, device='cpu'):
    """Play one game head-to-head with optional tactical analysis for both models"""
    state = game.get_initial_state()
    player = 1
    last_action = None
    
    # Separate stats for each model
    tactics_m1 = new_tactics_stats() if analyze_tactics else None
    tactics_m2 = new_tactics_stats() if analyze_tactics else None
    
    while True:
        # Check terminal AFTER a move was made
        if last_action is not None:
            value, is_terminal = game.get_value_and_terminated(state, last_action)
            if is_terminal:
                # value=1 means the player who just moved won
                winner = -player if value == 1 else 0
                return (winner, tactics_m1, tactics_m2) if analyze_tactics else winner
        
        valid_moves = game.get_valid_moves(state)
        if not np.any(valid_moves):
            return (0, tactics_m1, tactics_m2) if analyze_tactics else 0  # Draw
        
        # Select action - MUST use canonical perspective for MCTS
        is_model1_turn = (player == 1 and model1_as_p1) or (player == -1 and not model1_as_p1)
        
        if is_model1_turn:
            current_model = model1
            current_mcts = mcts1
            current_tactics = tactics_m1
        else:
            current_model = model2
            current_mcts = mcts2
            current_tactics = tactics_m2
        
        neutral_state = game.change_perspective(state, player)
        
        # --- Tactical analysis ---
        if analyze_tactics:
            current_tactics['turns_analyzed'] += 1
            win_moves = immediate_winning_moves(game, state, player)
            opp_win_moves = immediate_winning_moves(game, state, -player)
            
            need_probs = (len(win_moves) > 0) or (len(opp_win_moves) > 0)
            net_p = None
            if need_probs:
                net_p, _ = policy_probs_from_net(game, current_model, neutral_state, device)
            
            if len(win_moves) > 0:
                current_tactics['had_immediate_win'] += 1
            
            if len(opp_win_moves) > 0:
                current_tactics['had_to_block'] += 1
        
        # MCTS search
        action_probs = current_mcts.search(neutral_state)
        
        action_probs = action_probs * valid_moves
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(np.where(valid_moves)[0])
        
        # --- Record tactical outcomes ---
        if analyze_tactics:
            # Immediate win conversion
            if len(win_moves) > 0 and action in win_moves:
                current_tactics['took_immediate_win'] += 1
                if net_p is not None:
                    update_running_avg(current_tactics, 'avg_net_prob_on_win_moves',
                                     'n_win_prob_samples', float(net_p[action]))
                mcts_prob = action_probs[action] if action_probs.sum() > 0 else 0.0
                update_running_avg(current_tactics, 'avg_mcts_prob_on_win_moves',
                                 'n_win_prob_samples', float(mcts_prob))
            
            # Block rate
            if len(opp_win_moves) > 0:
                blocked = action in opp_win_moves
                if block_ok_if_win and len(win_moves) > 0 and action in win_moves:
                    blocked = True
                
                if blocked:
                    current_tactics['did_block'] += 1
                    if net_p is not None:
                        update_running_avg(current_tactics, 'avg_net_prob_on_blocks',
                                         'n_block_prob_samples', float(net_p[action]))
                    mcts_prob = action_probs[action] if action_probs.sum() > 0 else 0.0
                    update_running_avg(current_tactics, 'avg_mcts_prob_on_blocks',
                                     'n_block_prob_samples', float(mcts_prob))
        
        state = game.get_next_state(state, action, player)
        last_action = action
        player = -player

def test_head_to_head(model1, mcts1, model2, mcts2, game, name1, name2, num_games=100, 
                      analyze_tactics=True, device='cpu'):
    """Test two models head-to-head with detailed statistics and tactical analysis"""
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD: {name1} vs {name2} ({num_games} games)")
    print('='*60)
    
    model1_wins_as_p1 = 0
    model1_wins_as_p2 = 0
    model2_wins_as_p1 = 0
    model2_wins_as_p2 = 0
    draws_as_p1 = 0
    draws_as_p2 = 0
    
    # Aggregate tactics for both models
    agg_tactics_m1 = new_tactics_stats() if analyze_tactics else None
    agg_tactics_m2 = new_tactics_stats() if analyze_tactics else None
    
    # Play half games as P1, half as P2
    print(f"\n{name1} as Player 1 ({num_games//2} games)...")
    for _ in tqdm(range(num_games // 2), ncols=60):
        if analyze_tactics:
            result, tac1, tac2 = play_head_to_head(model1, mcts1, model2, mcts2, game, 
                                                   model1_as_p1=True, analyze_tactics=True, device=device)
            # Merge tactics
            for key in agg_tactics_m1:
                if not key.startswith('avg_') and not key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
                elif key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
        else:
            result = play_head_to_head(model1, mcts1, model2, mcts2, game, model1_as_p1=True)
        
        if result == 1:
            model1_wins_as_p1 += 1
        elif result == -1:
            model2_wins_as_p2 += 1
        else:
            draws_as_p1 += 1
    
    print(f"{name1} as Player 2 ({num_games//2} games)...")
    for _ in tqdm(range(num_games // 2), ncols=60):
        if analyze_tactics:
            result, tac1, tac2 = play_head_to_head(model1, mcts1, model2, mcts2, game, 
                                                   model1_as_p1=False, analyze_tactics=True, device=device)
            # Merge tactics
            for key in agg_tactics_m1:
                if not key.startswith('avg_') and not key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
                elif key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
        else:
            result = play_head_to_head(model1, mcts1, model2, mcts2, game, model1_as_p1=False)
        
        if result == -1:
            model1_wins_as_p2 += 1
        elif result == 1:
            model2_wins_as_p1 += 1
        else:
            draws_as_p2 += 1
    
    model1_wins = model1_wins_as_p1 + model1_wins_as_p2
    model2_wins = model2_wins_as_p1 + model2_wins_as_p2
    draws = draws_as_p1 + draws_as_p2
    
    # Calculate score (wins + 0.5*draws)
    model1_score = model1_wins + 0.5 * draws
    model2_score = model2_wins + 0.5 * draws
    score_rate = model1_score / num_games
    
    # Confidence interval for score
    _, ci_low, ci_high = wilson_interval(int(model1_score * 2), num_games * 2)
    ci_low /= 2
    ci_high /= 2
    
    # Elo estimate
    try:
        elo_diff = elo_diff_estimate(score_rate)
        elo_str = f"{elo_diff:+.0f}" if abs(elo_diff) < 1000 else (">>+400" if elo_diff > 0 else "<<-400")
    except:
        elo_str = "N/A"
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   {name1}: {model1_wins}/{num_games} wins ({model1_wins/num_games*100:.1f}%)")
    print(f"     - As P1: {model1_wins_as_p1}/{num_games//2}, As P2: {model1_wins_as_p2}/{num_games//2}")
    print(f"   {name2}: {model2_wins}/{num_games} wins ({model2_wins/num_games*100:.1f}%)")
    print(f"     - As P1: {model2_wins_as_p1}/{num_games//2}, As P2: {model2_wins_as_p2}/{num_games//2}")
    print(f"   Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
    print(f"\n   {name1} score: {model1_score:.1f}/{num_games} ({score_rate*100:.1f}%)")
    print(f"   95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    print(f"   Estimated Elo difference: {elo_str}")
    
    if model1_wins > model2_wins:
        print(f"\n🏆 WINNER: {name1}")
    elif model2_wins > model1_wins:
        print(f"\n🏆 WINNER: {name2}")
    else:
        print(f"\n🤝 TIE (by wins)")
    
    # Print tactical summaries for both models
    if analyze_tactics:
        print_tactics_summary(name1, agg_tactics_m1)
        print_tactics_summary(name2, agg_tactics_m2)
    
    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'model1_score': model1_score,
        'score_rate': score_rate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'elo_diff': elo_str
    }

def test_head_to_head_custom(model1, mcts1, model2, mcts2, game, name1, name2, 
                             games_as_p1=10, games_as_p2=10,
                             analyze_tactics=True, device='cpu'):
    """Test two models head-to-head with custom game counts per side"""
    num_games = games_as_p1 + games_as_p2
    print(f"\n{'='*60}")
    print(f"HEAD-TO-HEAD: {name1} vs {name2} ({games_as_p1}+{games_as_p2}={num_games} games)")
    print('='*60)
    
    model1_wins_as_p1 = 0
    model1_wins_as_p2 = 0
    model2_wins_as_p1 = 0
    model2_wins_as_p2 = 0
    draws_as_p1 = 0
    draws_as_p2 = 0
    
    # Aggregate tactics for both models
    agg_tactics_m1 = new_tactics_stats() if analyze_tactics else None
    agg_tactics_m2 = new_tactics_stats() if analyze_tactics else None
    
    # Play games with model1 as P1
    print(f"\n{name1} as Player 1 ({games_as_p1} games)...")
    for _ in tqdm(range(games_as_p1), ncols=60):
        if analyze_tactics:
            result, tac1, tac2 = play_head_to_head(model1, mcts1, model2, mcts2, game, 
                                                   model1_as_p1=True, analyze_tactics=True, device=device)
            # Merge tactics
            for key in agg_tactics_m1:
                if not key.startswith('avg_') and not key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
                elif key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
        else:
            result = play_head_to_head(model1, mcts1, model2, mcts2, game, model1_as_p1=True)
        
        if result == 1:
            model1_wins_as_p1 += 1
        elif result == -1:
            model2_wins_as_p2 += 1
        else:
            draws_as_p1 += 1
    
    # Play games with model1 as P2
    print(f"{name1} as Player 2 ({games_as_p2} games)...")
    for _ in tqdm(range(games_as_p2), ncols=60):
        if analyze_tactics:
            result, tac1, tac2 = play_head_to_head(model1, mcts1, model2, mcts2, game, 
                                                   model1_as_p1=False, analyze_tactics=True, device=device)
            # Merge tactics
            for key in agg_tactics_m1:
                if not key.startswith('avg_') and not key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
                elif key.startswith('n_'):
                    agg_tactics_m1[key] += tac1[key]
                    agg_tactics_m2[key] += tac2[key]
        else:
            result = play_head_to_head(model1, mcts1, model2, mcts2, game, model1_as_p1=False)
        
        if result == -1:
            model1_wins_as_p2 += 1
        elif result == 1:
            model2_wins_as_p1 += 1
        else:
            draws_as_p2 += 1
    
    model1_wins = model1_wins_as_p1 + model1_wins_as_p2
    model2_wins = model2_wins_as_p1 + model2_wins_as_p2
    draws = draws_as_p1 + draws_as_p2
    
    # Calculate score (wins + 0.5*draws)
    model1_score = model1_wins + 0.5 * draws
    model2_score = model2_wins + 0.5 * draws
    score_rate = model1_score / num_games
    
    # Confidence interval for score
    _, ci_low, ci_high = wilson_interval(int(model1_score * 2), num_games * 2)
    ci_low /= 2
    ci_high /= 2
    
    # Elo estimate
    try:
        elo_diff = elo_diff_estimate(score_rate)
        elo_str = f"{elo_diff:+.0f}" if abs(elo_diff) < 1000 else (">>+400" if elo_diff > 0 else "<<-400")
    except:
        elo_str = "N/A"
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   {name1}: {model1_wins}/{num_games} wins ({model1_wins/num_games*100:.1f}%)")
    print(f"     - As P1: {model1_wins_as_p1}/{games_as_p1}, As P2: {model1_wins_as_p2}/{games_as_p2}")
    print(f"   {name2}: {model2_wins}/{num_games} wins ({model2_wins/num_games*100:.1f}%)")
    print(f"     - As P1: {model2_wins_as_p1}/{games_as_p2}, As P2: {model2_wins_as_p2}/{games_as_p1}")
    print(f"   Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
    print(f"\n   {name1} score: {model1_score:.1f}/{num_games} ({score_rate*100:.1f}%)")
    print(f"   95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    print(f"   Estimated Elo difference: {elo_str}")
    
    if model1_wins > model2_wins:
        print(f"\n🏆 WINNER: {name1}")
    elif model2_wins > model1_wins:
        print(f"\n🏆 WINNER: {name2}")
    else:
        print(f"\n🤝 TIE (by wins)")
    
    # Print tactical summaries for both models
    if analyze_tactics:
        print_tactics_summary(name1, agg_tactics_m1)
        print_tactics_summary(name2, agg_tactics_m2)
    
    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'model1_score': model1_score,
        'score_rate': score_rate,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'elo_diff': elo_str
    }

def main():
    parser = argparse.ArgumentParser(description='Compare two AlphaZero model iterations')
    parser.add_argument('--iter1', type=int, default=5, help='First model iteration')
    parser.add_argument('--iter2', type=int, default=13, help='Second model iteration')
    parser.add_argument('--vs_random_games', type=int, default=100, help='Games vs random per model (legacy)')
    parser.add_argument('--h2h_games', type=int, default=100, help='Head-to-head games (legacy)')
    parser.add_argument('--eval_num_searches', '--mcts_searches', type=int, default=70, 
                        dest='eval_num_searches', help='MCTS searches for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--analyze_tactics', action='store_true', default=True, 
                        help='Enable in-game tactical analysis (win conversion, blocking)')
    parser.add_argument('--no_analyze_tactics', action='store_false', dest='analyze_tactics',
                        help='Disable tactical analysis for faster evaluation')
    
    # New eval plan arguments
    parser.add_argument('--random_games', type=int, default=10, help='Games vs random for iter2')
    parser.add_argument('--random_only_p1', type=int, default=1, help='1=only as P1, 0=both sides')
    parser.add_argument('--h2h_p1_games', type=int, default=10, help='H2H games with iter2 as P1')
    parser.add_argument('--h2h_p2_games', type=int, default=10, help='H2H games with iter2 as P2')
    parser.add_argument('--log_path', type=str, default=None, help='Optional log path (for external logging)')
    
    args = parser.parse_args()
    
    iter1 = args.iter1
    iter2 = args.iter2
    
    # Setup logging to file (if not already handled externally)
    if args.log_path:
        # External logging - just print header
        print(f"📝 Log path: {args.log_path}")
        log_path = args.log_path
        log_file = None
    else:
        # Setup internal logging
        log_path, log_file = setup_logging(iter1, iter2)
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine eval plan: use new args if provided, else legacy
    use_new_plan = (args.random_games != 100 or args.h2h_p1_games != 100)
    
    print("="*60)
    print(f"COMPARING MODEL {iter1} vs MODEL {iter2}")
    print(f"Seed: {args.seed}, Device: {args.device}")
    print(f"MCTS searches: {args.eval_num_searches} (NO DIRICHLET NOISE)")
    if use_new_plan:
        total_games = args.random_games + args.h2h_p1_games + args.h2h_p2_games
        print(f"Eval plan: {args.random_games} vs random (P1 only={bool(args.random_only_p1)}) + {args.h2h_p1_games}+{args.h2h_p2_games} h2h = {total_games} total")
    print("="*60)
    
    game = ConnectFour()
    
    # Load models
    print(f"\n📥 Loading models...")
    model1 = load_model(iter1, game)
    model2 = load_model(iter2, game)
    model1 = model1.to(args.device)
    model2 = model2.to(args.device)
    print("✅ Models loaded")
    
    # Create MCTS with NO DIRICHLET NOISE for fair evaluation
    eval_args = {
        **MCTS_CONFIG, 
        'num_searches': args.eval_num_searches,
        'dirichlet_epsilon': 0.0  # CRITICAL: Disable noise for deterministic eval
    }
    mcts1 = MCTS(game, eval_args, model1)
    mcts2 = MCTS(game, eval_args, model2)
    
    if use_new_plan:
        # New eval plan: reduced games for auto-eval
        # Skip static tactical tests for speed
        
        # Test: Model 2 vs Random (P1 only if requested)
        if args.random_only_p1:
            random2 = test_vs_random_p1_only(model2, mcts2, game, f"Model {iter2}", 
                                            num_games=args.random_games, analyze_tactics=args.analyze_tactics,
                                            device=args.device)
        else:
            random2 = test_vs_random(model2, mcts2, game, f"Model {iter2}", 
                                    num_games=args.random_games, analyze_tactics=args.analyze_tactics,
                                    device=args.device)
        
        # Test: Head-to-head with custom game counts
        h2h = test_head_to_head_custom(model1, mcts1, model2, mcts2, game, 
                                      f"Model {iter1}", f"Model {iter2}", 
                                      games_as_p1=args.h2h_p1_games, games_as_p2=args.h2h_p2_games,
                                      analyze_tactics=args.analyze_tactics, device=args.device)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nModel {iter2} vs Random:")
        print(f"  Win rate: {random2['win_rate']*100:.1f}% ±[{random2['ci_low']*100:.1f}%, {random2['ci_high']*100:.1f}%]")
        print(f"\nHead-to-head ({args.h2h_p1_games + args.h2h_p2_games} games):")
        print(f"  Model {iter1}: {h2h['model1_wins']} wins, score {h2h['model1_score']:.1f}/{args.h2h_p1_games + args.h2h_p2_games}")
        model2_score = h2h['model2_wins'] + 0.5 * h2h['draws']
        print(f"  Model {iter2}: {h2h['model2_wins']} wins, score {model2_score:.1f}/{args.h2h_p1_games + args.h2h_p2_games}")
        print(f"  Elo diff estimate: {h2h['elo_diff']}")
        
    else:
        # Legacy full eval plan
        # Test 1: Tactical scenarios for model 1
        tactical1 = test_tactical_scenarios(model1, mcts1, game, f"Model {iter1}", args.device)
        
        # Test 2: Tactical scenarios for model 2
        tactical2 = test_tactical_scenarios(model2, mcts2, game, f"Model {iter2}", args.device)
        
        # Test 3: Model 1 vs Random
        random1 = test_vs_random(model1, mcts1, game, f"Model {iter1}", 
                                num_games=args.vs_random_games, analyze_tactics=args.analyze_tactics, 
                                device=args.device)
        
        # Test 4: Model 2 vs Random
        random2 = test_vs_random(model2, mcts2, game, f"Model {iter2}", 
                                num_games=args.vs_random_games, analyze_tactics=args.analyze_tactics, 
                                device=args.device)
        
        # Test 5: Head-to-head
        h2h = test_head_to_head(model1, mcts1, model2, mcts2, game, 
                                f"Model {iter1}", f"Model {iter2}", num_games=args.h2h_games,
                                analyze_tactics=args.analyze_tactics, device=args.device)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nModel {iter1}:")
        print(f"  Tactical (Net/MCTS): Win={tactical1['find_win_net']:.3f}/{tactical1['find_win_mcts']:.3f}, Block={tactical1['block_threat_net']:.3f}/{tactical1['block_threat_mcts']:.3f}")
        print(f"  vs Random: {random1['win_rate']*100:.1f}% ±[{random1['ci_low']*100:.1f}%, {random1['ci_high']*100:.1f}%]")
        print(f"\nModel {iter2}:")
        print(f"  Tactical (Net/MCTS): Win={tactical2['find_win_net']:.3f}/{tactical2['find_win_mcts']:.3f}, Block={tactical2['block_threat_net']:.3f}/{tactical2['block_threat_mcts']:.3f}")
        print(f"  vs Random: {random2['win_rate']*100:.1f}% ±[{random2['ci_low']*100:.1f}%, {random2['ci_high']*100:.1f}%]")
        print(f"\nHead-to-head ({args.h2h_games} games):")
        print(f"  Model {iter1}: {h2h['model1_wins']} wins, score {h2h['model1_score']:.1f}/{args.h2h_games}")
        model2_score = h2h['model2_wins'] + 0.5 * h2h['draws']
        print(f"  Model {iter2}: {h2h['model2_wins']} wins, score {model2_score:.1f}/{args.h2h_games}")
        print(f"  Elo diff estimate: {h2h['elo_diff']}")
        
        # Key insights
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        net_improvement_win = tactical2['find_win_net'] - tactical1['find_win_net']
        net_improvement_block = tactical2['block_threat_net'] - tactical1['block_threat_net']
        print(f"Network learning (Net-only policy):")
        print(f"  Win detection: {net_improvement_win:+.3f}")
        print(f"  Block detection: {net_improvement_block:+.3f}")
        if abs(net_improvement_win) < 0.05 and abs(net_improvement_block) < 0.05:
            print(f"  → Network skills similar; differences likely from MCTS/noise")
        elif net_improvement_win > 0.1 or net_improvement_block > 0.1:
            print(f"  → Model {iter2} network learned better tactics!")
        else:
            print(f"  → Model {iter1} network has slightly better raw tactics")
    
    # Close log file (if we opened it)
    if log_file is not None:
        print(f"\n✅ Evaluation complete. Log saved to: {log_path}")
        log_file.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
