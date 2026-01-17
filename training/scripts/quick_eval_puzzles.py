#!/usr/bin/env python3
"""Fast evaluation for Connect4 training.

Runs:
  1) Quick head-to-head (iter2 vs iter1): 5 games with iter2 as P1, 5 games with iter2 as P2
  2) 50 tactical puzzles (must-win / must-block), ordered easy->hard

Outputs:
  - A timestamped log in logs/evaluations/
  - A JSONL puzzle result file in logs/evaluations/

Designed to be quick enough to run during training without stopping self-play.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# repo root assumed to be parent of this file's folder
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import MODEL_CONFIG, MCTS_CONFIG


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(tag: str):
    log_dir = BASE_DIR / 'logs' / 'evaluations'
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{tag}_{ts}.log'
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)
    print(f"📝 Logging to: {log_path}")
    return log_dir, log_path, log_file


def load_model(iteration: int, game: ConnectFour, device: str):
    ckpt = BASE_DIR / f"checkpoints/connect4/model_{iteration}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden']).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def softmax_policy_from_net(game, model, state, device: str):
    with torch.no_grad():
        enc = game.get_encoded_state(state)
        x = torch.tensor(enc, dtype=torch.float32, device=device).unsqueeze(0)
        p_logits, v = model(x)
        p = torch.softmax(p_logits, dim=1).squeeze(0).detach().cpu().numpy()
        return p, float(v.item())


def masked_normalize(policy: np.ndarray, valid: np.ndarray):
    p = policy * valid
    s = float(p.sum())
    if s > 0:
        return p / s
    # fallback uniform over valid
    idx = np.where(valid > 0)[0]
    out = np.zeros_like(policy, dtype=np.float32)
    out[idx] = 1.0 / len(idx)
    return out


def choose_argmax(policy: np.ndarray, valid: np.ndarray):
    p = masked_normalize(policy, valid)
    return int(np.argmax(p)), p


def play_one_h2h(game, mcts_p1, mcts_p2, iter2_is_player1: bool):
    """Return result from iter2 perspective: +1 win, 0 draw, -1 loss."""
    state = game.get_initial_state()
    player = 1

    while True:
        valid = game.get_valid_moves(state)
        if not np.any(valid):
            return 0

        # if previous player just won
        if game.check_win(state, -player):
            winner = -player
            iter2_player = 1 if iter2_is_player1 else -1
            return 1 if winner == iter2_player else -1

        neutral = game.change_perspective(state, player)  # canonical for current-to-move

        if (player == 1 and iter2_is_player1) or (player == -1 and not iter2_is_player1):
            # iter2 acts
            probs = mcts_p2.search(neutral, add_noise=False)
        else:
            # iter1 acts
            probs = mcts_p1.search(neutral, add_noise=False)

        # choose move
        valid2 = game.get_valid_moves(state)
        action, _ = choose_argmax(probs, valid2)
        state = game.get_next_state(state, action, player)
        player = -player


def run_h2h(game, mcts1, mcts2, games_as_p1: int, games_as_p2: int, seed: int):
    rng = np.random.default_rng(seed)
    # shuffle order a bit
    plan = ([True] * games_as_p1) + ([False] * games_as_p2)
    rng.shuffle(plan)

    w = d = l = 0
    for i, iter2_is_p1 in enumerate(plan, 1):
        r = play_one_h2h(game, mcts1, mcts2, iter2_is_player1=iter2_is_p1)
        if r == 1:
            w += 1
        elif r == 0:
            d += 1
        else:
            l += 1
    return {"wins": w, "draws": d, "losses": l, "games": len(plan)}


def immediate_winning_moves(game, state, player):
    valid = game.get_valid_moves(state)
    wins = []
    for a in np.where(valid)[0]:
        s2 = game.get_next_state(state, int(a), player)
        if game.check_win(s2, player):
            wins.append(int(a))
    return wins


def find_forced_win_moves(game, state, player, depth=3):
    """Find moves that lead to forced win within depth plies (player's moves only).
    Returns list of moves that guarantee a win assuming best defense."""
    valid = game.get_valid_moves(state)
    winning_moves = []
    
    for a in np.where(valid)[0]:
        if can_force_win(game, state, int(a), player, depth):
            winning_moves.append(int(a))
    
    return winning_moves


def can_force_win(game, state, action, player, depth):
    """Check if making 'action' leads to a forced win within depth plies."""
    # Make the move
    new_state = game.get_next_state(state, action, player)
    
    # Immediate win?
    if game.check_win(new_state, player):
        return True
    
    if depth <= 1:
        return False
    
    # Opponent's turn - they will try all moves to avoid loss
    opp = -player
    opp_state = game.change_perspective(new_state, opp)
    valid_opp = game.get_valid_moves(opp_state)
    
    # If opponent has no moves, it's a draw (not a win)
    if not np.any(valid_opp):
        return False
    
    # For every opponent move, we must have a winning response
    for opp_a in np.where(valid_opp)[0]:
        after_opp = game.get_next_state(new_state, int(opp_a), opp)
        
        # If opponent can win, this line fails
        if game.check_win(after_opp, opp):
            return False
        
        # Now it's our turn again - can we still win from here?
        our_state = game.change_perspective(after_opp, player)
        valid_ours = game.get_valid_moves(our_state)
        
        # Try to find at least one winning continuation
        has_winning_response = False
        for our_a in np.where(valid_ours)[0]:
            if can_force_win(game, our_state, int(our_a), player, depth - 1):
                has_winning_response = True
                break
        
        # If we have no winning response to this opponent move, fail
        if not has_winning_response:
            return False
    
    # All opponent moves have been countered
    return True


def find_critical_defense_moves(game, state, player, depth=2):
    """Find moves that are necessary to avoid losing within depth plies.
    Returns list of defensive moves that prevent opponent's forced win."""
    opp = -player
    valid = game.get_valid_moves(state)
    critical_moves = []
    
    # Check what happens if opponent moves next (they're about to win)
    opp_state = game.change_perspective(state, opp)
    opp_winning_moves = find_forced_win_moves(game, opp_state, opp, depth)
    
    if not opp_winning_moves:
        return []
    
    # Find our moves that block all opponent winning threats
    for a in np.where(valid)[0]:
        after_our_move = game.get_next_state(state, int(a), player)
        opp_state_after = game.change_perspective(after_our_move, opp)
        
        # Check if opponent can still force win after our move
        opp_still_wins = find_forced_win_moves(game, opp_state_after, opp, depth)
        
        if not opp_still_wins:
            critical_moves.append(int(a))
    
    return critical_moves


def make_state_from_moves(game, moves, starting_player=1):
    """Create a legal state by applying moves alternately. Returns canonical state for next-to-move."""
    state = game.get_initial_state()
    player = starting_player
    last_action = None
    for a in moves:
        last_action = int(a)
        state = game.get_next_state(state, last_action, player)
        # stop if terminal
        if game.check_win(state, player):
            break
        player = -player
    # next player to move is "player" if last move ended game, but puzzles assume nonterminal.
    # we return canonical for next-to-move:
    next_player = -player if (last_action is not None and not game.check_win(state, player)) else player
    return game.change_perspective(state, next_player)


def generate_puzzles(game, n_total: int, seed: int):
    """Generate harder tactical puzzles requiring multi-move lookahead."""
    rng = np.random.default_rng(seed)

    puzzles = []

    # --- Harder curated puzzles requiring 2-3 move lookahead ---
    curated = [
        # 2-move forced win setups
        {"name": "fork_setup", "type": "WIN_2", "moves": [3, 0, 3, 1, 2, 0, 4, 1], "depth": 2},
        {"name": "forced_sequence", "type": "WIN_2", "moves": [3, 2, 3, 2, 3, 5, 4, 2], "depth": 2},
        # Must block opponent's 2-move threat
        {"name": "defend_fork", "type": "DEFEND_2", "moves": [3, 0, 2, 0, 4, 1, 5, 0], "depth": 2},
        # 3-move combinations
        {"name": "deep_win", "type": "WIN_3", "moves": [3, 1, 3, 2, 2, 1, 4, 2, 2, 4], "depth": 3},
    ]

    for c in curated:
        s = make_state_from_moves(game, c["moves"], starting_player=1)
        depth = c.get("depth", 2)
        
        if "WIN" in c["type"]:
            correct = find_forced_win_moves(game, s, player=1, depth=depth)
        else:  # DEFEND
            correct = find_critical_defense_moves(game, s, player=1, depth=depth)
        
        if correct:
            puzzles.append({
                "id": f"curated::{c['name']}",
                "type": c["type"],
                "state": s.astype(np.float32),
                "correct_moves": correct,
                "difficulty": depth * 10  # Weight by depth
            })

    # --- Mine harder tactical positions ---
    # Focus on deeper positions (15-30 moves) that require lookahead
    attempts = 0
    max_attempts = 30000
    
    while len(puzzles) < n_total and attempts < max_attempts:
        attempts += 1

        # Create deeper random positions
        state = game.get_initial_state()
        player = 1
        n_moves = int(rng.integers(15, 32))  # Deeper positions
        ok = True

        for _ in range(n_moves):
            valid = np.where(game.get_valid_moves(state) > 0)[0]
            if len(valid) == 0:
                ok = False
                break
            a = int(rng.choice(valid))
            state = game.get_next_state(state, a, player)
            if game.check_win(state, player):
                ok = False
                break
            player = -player

        if not ok:
            continue

        canon = game.change_perspective(state, player)

        # Try to find 2-3 move tactics (more valuable than 1-move)
        # First check if there's an immediate win (skip these - too easy)
        immediate_wins = immediate_winning_moves(game, canon, player=1)
        if immediate_wins:
            continue  # Skip 1-move puzzles
        
        immediate_blocks = immediate_winning_moves(game, canon, player=-1)
        if immediate_blocks:
            continue  # Skip simple blocks

        # Look for 2-move forced wins
        win2_moves = find_forced_win_moves(game, canon, player=1, depth=2)
        if win2_moves and len(puzzles) < n_total:
            puzzles.append({
                "id": f"mined::win2::{len(puzzles)}",
                "type": "WIN_2",
                "state": canon.astype(np.float32),
                "correct_moves": win2_moves,
                "difficulty": 20 + n_moves
            })
            continue

        # Look for critical 2-move defenses
        def2_moves = find_critical_defense_moves(game, canon, player=1, depth=2)
        if def2_moves and len(puzzles) < n_total:
            puzzles.append({
                "id": f"mined::def2::{len(puzzles)}",
                "type": "DEFEND_2",
                "state": canon.astype(np.float32),
                "correct_moves": def2_moves,
                "difficulty": 25 + n_moves
            })
            continue
        
        # Look for 3-move forced wins (hardest)
        if len(puzzles) < n_total * 0.3:  # Only generate some 3-move puzzles
            win3_moves = find_forced_win_moves(game, canon, player=1, depth=3)
            if win3_moves:
                puzzles.append({
                    "id": f"mined::win3::{len(puzzles)}",
                    "type": "WIN_3",
                    "state": canon.astype(np.float32),
                    "correct_moves": win3_moves,
                    "difficulty": 40 + n_moves
                })
                continue

    # Sort easy->hard
    puzzles.sort(key=lambda x: x["difficulty"])
    return puzzles[:n_total]


def eval_puzzles(game, model, mcts, puzzles, device: str, label: str, jsonl_path: Path):
    correct_net = 0
    correct_mcts = 0

    with jsonl_path.open('a') as f:
        for idx, pz in enumerate(puzzles, 1):
            s = pz["state"]
            valid = game.get_valid_moves(s)
            correct = set(pz["correct_moves"])

            # net-only
            net_policy, net_value = softmax_policy_from_net(game, model, s, device)
            net_action, net_p = choose_argmax(net_policy, valid)

            # mcts
            mcts_policy = mcts.search(s, add_noise=False)
            mcts_action, mcts_p = choose_argmax(mcts_policy, valid)

            net_ok = (net_action in correct)
            mcts_ok = (mcts_action in correct)
            correct_net += int(net_ok)
            correct_mcts += int(mcts_ok)

            # log short line
            top3_net = list(np.argsort(-net_p)[:3])
            top3_mcts = list(np.argsort(-mcts_p)[:3])

            print(f"[{label}] Puzzle {idx:02d}/{len(puzzles)} {pz['type']} id={pz['id']} correct={sorted(correct)}")
            print(f"  NET : pick={net_action} ok={net_ok} v={net_value:+.3f} top3={[(int(a), float(net_p[a])) for a in top3_net]}")
            print(f"  MCTS: pick={mcts_action} ok={mcts_ok} top3={[(int(a), float(mcts_p[a])) for a in top3_mcts]}")

            rec = {
                "label": label,
                "puzzle_id": pz["id"],
                "type": pz["type"],
                "difficulty": pz["difficulty"],
                "correct_moves": sorted(list(correct)),
                "net": {
                    "action": net_action,
                    "ok": net_ok,
                    "value": net_value,
                    "probs": [float(x) for x in net_p]
                },
                "mcts": {
                    "action": mcts_action,
                    "ok": mcts_ok,
                    "probs": [float(x) for x in mcts_p]
                }
            }
            f.write(json.dumps(rec) + "\n")
            print()  # Empty line after each puzzle

    n = len(puzzles)
    return {
        "puzzles": n,
        "net_acc": 0.0 if n == 0 else correct_net / n,
        "mcts_acc": 0.0 if n == 0 else correct_mcts / n
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iter1', type=int, required=True)
    ap.add_argument('--iter2', type=int, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--mcts_searches', type=int, default=70)
    ap.add_argument('--num_puzzles', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    tag = f"quick_eval_{args.iter1}_vs_{args.iter2}"
    log_dir, log_path, log_file = setup_logging(tag)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    game = ConnectFour()

    print("="*70)
    print(f"FAST EVAL: {args.iter1} vs {args.iter2}")
    print(f"Device={args.device} | MCTS searches={args.mcts_searches} | puzzles={args.num_puzzles}")
    print("="*70)

    # load models
    model1 = load_model(args.iter1, game, args.device)
    model2 = load_model(args.iter2, game, args.device)

    eval_args = {**MCTS_CONFIG, 'num_searches': args.mcts_searches, 'dirichlet_epsilon': 0.0}
    mcts1 = MCTS(game, eval_args, model1)
    mcts2 = MCTS(game, eval_args, model2)

    # puzzles
    print("\n" + "="*70)
    print("TACTICAL PUZZLES")
    print("="*70)
    puzzles = generate_puzzles(game, n_total=args.num_puzzles, seed=args.seed)
    jsonl_path = log_dir / f"puzzles_{args.iter1}_vs_{args.iter2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"🧩 Writing puzzle JSONL to: {jsonl_path}")

    res1 = eval_puzzles(game, model1, mcts1, puzzles, args.device, label=f"iter{args.iter1}", jsonl_path=jsonl_path)
    res2 = eval_puzzles(game, model2, mcts2, puzzles, args.device, label=f"iter{args.iter2}", jsonl_path=jsonl_path)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Puzzles iter{args.iter1}: NET acc={res1['net_acc']:.1%} | MCTS acc={res1['mcts_acc']:.1%} | N={res1['puzzles']}")
    print(f"Puzzles iter{args.iter2}: NET acc={res2['net_acc']:.1%} | MCTS acc={res2['mcts_acc']:.1%} | N={res2['puzzles']}")

    log_file.close()


if __name__ == '__main__':
    main()
