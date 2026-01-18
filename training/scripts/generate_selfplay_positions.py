#!/usr/bin/env python3
"""
Generate positions using AlphaZero model self-play, then label with solver.

This creates a better training distribution than random positions:
- Positions are tactically interesting (what strong play produces)
- Mid/late-game states that solve faster
- AlphaZero generates → Solver labels → NN learns perfect policy/value
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.engine.zobrist import get_zobrist_hasher
from alpha_zero_light.data.generate_positions import save_positions_npz


def load_model(checkpoint_path="checkpoints/connect4/model_195.pt"):
    """Load AlphaZero model for position generation."""
    game = ConnectFour()
    model = ResNet(game, num_res_blocks=20, num_hidden=256)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return game, model, device


def generate_positions_from_selfplay(
    model,
    game,
    num_positions: int,
    mcts_searches: int = 50,
    temperature: float = 0.8,
    min_moves: int = 10,
    max_moves: int = 25,
    checkpoint_interval: int = 500,
    output_path: str = 'data/selfplay_positions.npz',
    verbose: bool = False,
    random_opening_moves: int = 0
):
    """
    Generate positions using AlphaZero self-play with checkpointing.
    
    Args:
        model: Trained AlphaZero model
        game: Connect4 game instance
        num_positions: Number of positions to generate
        mcts_searches: MCTS simulations per move
        temperature: Sampling temperature (0.8 = focused, 1.0 = more variety)
        min_moves: Minimum moves before saving position
        max_moves: Maximum moves before starting new game
        checkpoint_interval: Save checkpoint every N positions
        output_path: Where to save final/checkpoint data
        verbose: Print first 3 games for inspection
        random_opening_moves: Randomize first N moves for diversity (0 = use model)
    
    Returns:
        List of (state, player, hash) tuples
    """
    # MCTS args (dict-like for compatibility)
    args = {
        'C': 2.0,
        'num_searches': mcts_searches,
        'dirichlet_epsilon': 0.0,  # No noise
        'dirichlet_alpha': 0.3,
        'mcts_batch_size': 1,
    }
    
    mcts = MCTS(game, args, model)
    zobrist = get_zobrist_hasher()
    
    # Try to resume from checkpoint
    checkpoint_path = output_path.replace('.npz', '_checkpoint.npz')
    positions = []
    seen_hashes = set()
    
    if Path(checkpoint_path).exists():
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = np.load(checkpoint_path)
        for i in range(len(checkpoint_data['states'])):
            state = checkpoint_data['states'][i]
            player = int(checkpoint_data['players'][i])
            hash_val = int(checkpoint_data['hashes'][i])
            positions.append((state, player, hash_val))
            seen_hashes.add(hash_val)
        print(f"✓ Resumed from checkpoint with {len(positions):,} positions")
    
    games_played = 0
    last_checkpoint = len(positions)
    
    # Helper to print board
    def print_board(state, title=""):
        if title:
            print(f"\n{title}")
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in range(6):
            print("  " + " ".join(symbols[state[row, col]] for col in range(7)))
    
    with tqdm(initial=len(positions), total=num_positions, desc="Generating positions via self-play") as pbar:
        while len(positions) < num_positions:
            # Play one game
            state = game.get_initial_state()
            player = 1
            move_count = 0
            game_history = []  # Track ALL moves (col indices) for solver
            game_vis_history = []  # Track for visualization only
            
            # Verbose mode: show first 3 games
            show_game = verbose and games_played < 3
            if show_game:
                print(f"\n{'='*60}")
                print(f"SAMPLE GAME {games_played + 1}")
                print(f"{'='*60}")
            
            # Randomly choose number of opening moves (0-4) for THIS game
            if random_opening_moves > 0:
                num_opening_random = np.random.randint(0, random_opening_moves + 1)
            else:
                num_opening_random = 0
            
            if show_game:
                if num_opening_random == 0:
                    print(f"Opening: Center start (strong theory)")
                else:
                    print(f"Opening: {num_opening_random} random moves")
            
            while True:
                # Special case: if 0 random moves, start with center column
                if move_count == 0 and num_opening_random == 0 and random_opening_moves > 0:
                    # Force center opening (column 3, 0-indexed)
                    action = 3
                    valid_moves = game.get_valid_moves(state)
                    mcts_probs = valid_moves / valid_moves.sum()  # Uniform for display
                # Random opening moves for diversity
                elif move_count < num_opening_random:
                    # Play completely random valid move
                    valid_moves = game.get_valid_moves(state)
                    valid_indices = np.where(valid_moves)[0]
                    action = np.random.choice(valid_indices)
                    mcts_probs = valid_moves / valid_moves.sum()  # Uniform for display
                else:
                    # Get MCTS policy with configurable temperature
                    mcts_probs = mcts.search(state, add_noise=False, temperature=temperature)
                    
                    # Sample move
                    valid_moves = game.get_valid_moves(state)
                    action = np.random.choice(
                        game.action_size,
                        p=mcts_probs * valid_moves
                    )
                
                # Track move for solver (1-indexed column)
                game_history.append(action + 1)  # Solver uses 1-7, not 0-6
                
                if show_game:
                    game_vis_history.append((state.copy(), player, action, mcts_probs))
                
                # Make move
                state = game.get_next_state(state, action, player)
                move_count += 1
                
                # Check termination
                value, terminated = game.get_value_and_terminated(state, action)
                
                # Save position if in mid/late-game range
                if min_moves <= move_count <= max_moves and not terminated:
                    hash_val = zobrist.hash_position(state, player)
                    
                    if hash_val not in seen_hashes:
                        seen_hashes.add(hash_val)
                        # Change perspective to make it from player-to-move's view
                        state_perspective = game.change_perspective(state, player)
                        
                        # Store position WITH move history (for solver)
                        move_sequence = game_history[:move_count]  # All moves so far
                        positions.append((state_perspective.copy(), player, hash_val, move_sequence))
                        pbar.update(1)
                        
                            # Save with move sequences
                            states = np.array([p[0] for p in positions], dtype=np.float32)
                            players = np.array([p[1] for p in positions], dtype=np.int8)
                            hashes = np.array([p[2] for p in positions], dtype=np.uint64)
                            # Store move sequences as CONCATENATED strings (for solver)
                            move_seqs = np.array([''.join(map(str, p[3])) for p in positions], dtype=object)
                            
                            np.savez_compressed(
                                checkpoint_path,
                                states=states,
                                players=players,
                                hashes=hashes,
                                move_sequences=move_seqs
                            )
                            last_checkpoint = len(positions)
                            pbar.set_postfix({'checkpoint': f'{len(positions):,}'})
                        
                        if len(positions) >= num_positions:
                            break
                
                if terminated or move_count >= max_moves:
                    break
                
                player = -player
            
            # Print game if verbose
            if show_game:
                for i, (s, p, a, probs) in enumerate(game_vis_history[:10]):  # First 10 moves
                    print(f"\nMove {i+1}: Player {'X' if p == 1 else 'O'} plays column {a}")
                    print(f"Policy: {' '.join(f'{probs[i]:.2f}' for i in range(7))}")
                    next_s = game.get_next_state(s, a, p)
                    print_board(next_s)
                
                result = "WIN" if value > 0 else "DRAW"
                print(f"\nGame ended: {result} after {move_count} moves")
                print(f"Saved positions from moves {min_moves}-{min(move_count, max_moves)}")
                print(f"Move sequence: {''.join(map(str, game_history[:15]))}...")
            
            games_played += 1
    
    print(f"\n✓ Generated {len(positions):,} positions from {games_played:,} self-play games")
    print(f"✓ Average {len(positions)/games_played:.1f} positions per game")
    
    return positions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate positions via AlphaZero self-play')
    parser.add_argument('--num-positions', type=int, default=2000,
                       help='Number of positions to generate')
    parser.add_argument('--output', default='data/selfplay_positions_2k.npz',
                       help='Output file path')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                       help='Save checkpoint every N positions')
    parser.add_argument('--mcts-searches', type=int, default=50,
                       help='MCTS simulations per move')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (0.8=focused, 1.0=variety)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show first 3 games for quality inspection')
    parser.add_argument('--random-opening-moves', type=int, default=0,
                       help='Max random opening moves per game (0-N, default 0). Each game randomly chooses 0 to N moves. If 0 moves chosen, starts with center column.')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*15 + "ALPHAZERO SELF-PLAY POSITION GENERATION")
    print("="*70)
    print(f"\nGenerating {args.num_positions:,} positions")
    print(f"Checkpoint every {args.checkpoint_interval} positions")
    print(f"Temperature: {args.temperature} ({'focused' if args.temperature < 0.9 else 'exploratory'})")
    if args.random_opening_moves > 0:
        print(f"Opening diversity: 0-{args.random_opening_moves} random moves per game")
        print(f"  (0 moves → center start, 1-{args.random_opening_moves} → random)")
    else:
        print(f"Opening: Model-based from start")
    print(f"Verbose mode: {'ON (showing first 3 games)' if args.verbose else 'OFF'}")
    print("\nStrategy: Use model_195 to generate tactically interesting positions")
    print("Then label with solver for perfect supervision\n")
    
    # Load model
    print("Loading AlphaZero model...")
    game, model, device = load_model()
    print(f"✓ Loaded model_195 on {device}")
    
    # Generate positions
    print(f"\nGenerating positions via self-play...")
    print("This uses your trained model, so positions will be high-quality!")
    
    positions = generate_positions_from_selfplay(
        model=model,
        game=game,
        num_positions=args.num_positions,
        mcts_searches=args.mcts_searches,
        temperature=args.temperature,
        min_moves=12,      # Mid-game start
        max_moves=25,      # Late-game end
        checkpoint_interval=args.checkpoint_interval,
        output_path=args.output,
        verbose=args.verbose,
        random_opening_moves=args.random_opening_moves
    )
    
    # Save final
    states = np.array([p[0] for p in positions], dtype=np.float32)
    players = np.array([p[1] for p in positions], dtype=np.int8)
    hashes = np.array([p[2] for p in positions], dtype=np.uint64)
    move_seqs = np.array([''.join(map(str, p[3])) for p in positions], dtype=object)
    
    np.savez_compressed(
        args.output,
        states=states,
        players=players,
        hashes=hashes,
        move_sequences=move_seqs
    )
    
    # Clean up checkpoint
    checkpoint_path = args.output.replace('.npz', '_checkpoint.npz')
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print(f"✓ Cleaned up checkpoint file")
    
    print("\n" + "="*70)
    print("POSITION GENERATION COMPLETE")
    print("="*70)
    print(f"✓ Saved {len(positions):,} positions to {args.output}")
    print(f"\nNext step: Label with solver")
    print(f"Command: python training/scripts/label_dataset.py \\")
    print(f"         --input {args.output} \\")
    print(f"         --output {args.output.replace('.npz', '_labeled.npz')} \\")
    print(f"         --workers 6")
