#!/usr/bin/env python3
"""
Batch label positions with Connect4 solver.

Labels positions in parallel for supervised training.
Solver is used OFFLINE only - final engine uses NN, not solver!
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from alpha_zero_light.data.solver_interface import Connect4Solver
from alpha_zero_light.data.generate_positions import load_positions_npz


def label_single_position(args):
    """Label a single position (for multiprocessing)."""
    idx, state, player, hash_val, move_sequence = args
    
    # Create solver instance (each process gets its own)
    solver = Connect4Solver()
    
    try:
        # Use the stored move sequence directly (already in solver format!)
        wdl, score, _ = solver.solve_position(move_sequence, timeout=120)
        return (idx, wdl, score, None)
    except Exception as e:
        print(f"Warning: Failed to solve position {idx}: {e}")
        return (idx, 0, 0, str(e))


def label_positions_parallel(
    positions_file: str,
    output_file: str,
    num_workers: int = 4,
    max_positions: int = None
):
    """
    Label positions in parallel using multiprocessing.
    
    Args:
        positions_file: NPZ file with unlabeled positions
        output_file: NPZ file to save labeled positions
        num_workers: Number of parallel workers
        max_positions: Maximum positions to label (for testing)
    """
    print(f"Loading positions from {positions_file}...")
    data = np.load(positions_file, allow_pickle=True)
    
    # Build positions list with move sequences
    positions = []
    for i in range(len(data['states'])):
        state = data['states'][i]
        player = int(data['players'][i])
        hash_val = int(data['hashes'][i])
        move_seq = str(data['move_sequences'][i])  # Already in solver format
        positions.append((state, player, hash_val, move_seq))
    
    if max_positions:
        positions = positions[:max_positions]
    
    print(f"Labeling {len(positions):,} positions with {num_workers} workers...")
    print(f"Estimated time: {len(positions) * 30 / num_workers / 3600:.1f} hours")
    print("(Solver used OFFLINE only - runtime engine won't call solver!)\n")
    
    # Prepare arguments for parallel processing
    args_list = [
        (idx, pos[0], pos[1], pos[2], pos[3])
        for idx, pos in enumerate(positions)
    ]
    
    # Label in parallel
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(label_single_position, args_list),
            total=len(positions),
            desc="Labeling positions"
        ))
    
    elapsed = time.time() - start_time
    
    # Extract labels
    wdl_labels = np.zeros(len(positions), dtype=np.int8)
    scores = np.zeros(len(positions), dtype=np.int16)
    errors = []
    
    for idx, wdl, score, error in results:
        wdl_labels[idx] = wdl
        scores[idx] = score
        if error:
            errors.append((idx, error))
    
    # Save labeled dataset
    print(f"\nSaving labeled dataset to {output_file}...")
    
    states = np.array([pos[0] for pos in positions], dtype=np.float32)
    players = np.array([pos[1] for pos in positions], dtype=np.int8)
    hashes = np.array([pos[2] for pos in positions], dtype=np.uint64)
    move_seqs = np.array([pos[3] for pos in positions], dtype=object)
    
    np.savez_compressed(
        output_file,
        states=states,
        players=players,
        hashes=hashes,
        move_sequences=move_seqs,
        wdl_labels=wdl_labels,
        scores=scores
    )
    
    # Statistics
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"Total positions: {len(positions):,}")
    print(f"Successfully labeled: {len(positions) - len(errors):,}")
    print(f"Errors: {len(errors)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Throughput: {len(positions)/elapsed*3600:.0f} positions/hour")
    
    # WDL distribution
    wins = np.sum(wdl_labels == 1)
    draws = np.sum(wdl_labels == 0)
    losses = np.sum(wdl_labels == -1)
    
    print(f"\nWDL Distribution:")
    print(f"  Wins:   {wins:,} ({wins/len(positions)*100:.1f}%)")
    print(f"  Draws:  {draws:,} ({draws/len(positions)*100:.1f}%)")
    print(f"  Losses: {losses:,} ({losses/len(positions)*100:.1f}%)")
    
    print(f"\nLabeled dataset saved to: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
    
    if errors:
        print(f"\nWarning: {len(errors)} positions failed to label")
        print("First few errors:")
        for idx, error in errors[:5]:
            print(f"  Position {idx}: {error}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Label positions with solver')
    parser.add_argument('--input', default='data/training_positions_10k.npz',
                       help='Input positions file')
    parser.add_argument('--output', default='data/labeled_positions_10k.npz',
                       help='Output labeled file')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--max', type=int, default=None,
                       help='Max positions to label (for testing)')
    
    args = parser.parse_args()
    
    label_positions_parallel(
        positions_file=args.input,
        output_file=args.output,
        num_workers=args.workers,
        max_positions=args.max
    )
