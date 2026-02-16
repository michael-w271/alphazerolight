#!/usr/bin/env python3
"""
Batch label positions with Connect4 solver (with RAM-efficient batch processing).

Labels positions in parallel for supervised training.
Solver is used OFFLINE only - final engine uses NN, not solver!
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from alpha_zero_light.data.solver_interface import Connect4Solver


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


def label_positions_batched(
    positions_file: str,
    output_file: str,
    num_workers: int = 4,
    batch_size: int = 5000,
    max_positions: int = None
):
    """
    Label positions in batches to reduce RAM usage.
    
    Args:
        positions_file: NPZ file with unlabeled positions
        output_file: NPZ file to save labeled positions
        num_workers: Number of parallel workers
        batch_size: Number of positions per batch (lower = less RAM)
        max_positions: Maximum positions to label (for testing)
    """
    # CRITICAL: Open NPZ file but DON'T load arrays yet (memory mapped)
    print(f"Opening positions file {positions_file}...")
    data = np.load(positions_file, allow_pickle=True, mmap_mode='r')
    
    total_positions = len(data['states'])
    if max_positions:
        total_positions = min(total_positions, max_positions)
    
    num_batches = (total_positions + batch_size - 1) // batch_size
    
    print(f"Labeling {total_positions:,} positions with {num_workers} workers...")
    print(f"Processing in {num_batches} batches of {batch_size:,} to reduce RAM")
    print(f"Estimated time: {total_positions * 30 / num_workers / 3600:.1f} hours")
    print("(Solver used OFFLINE only - runtime engine won't call solver!)\n")
    
    # Process in batches - STREAMING from disk!
    all_wdl_labels = []
    all_scores = []
    all_errors = []
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_positions)
        batch_len = end_idx - start_idx
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Positions {start_idx:,} to {end_idx:,}")
        print(f"{'='*60}")
        
        # Load ONLY this batch into RAM
        print(f"Loading batch {batch_idx + 1} from disk...")
        batch_states = data['states'][start_idx:end_idx]
        batch_players = data['players'][start_idx:end_idx]
        batch_hashes = data['hashes'][start_idx:end_idx]
        batch_move_seqs = data['move_sequences'][start_idx:end_idx]
        
        # Prepare batch arguments
        args_list = []
        for i in range(batch_len):
            args_list.append((
                start_idx + i,
                batch_states[i],
                int(batch_players[i]),
                int(batch_hashes[i]),
                str(batch_move_seqs[i])
            ))
        
        # Label batch in parallel
        print(f"Labeling batch {batch_idx + 1}...")
        with mp.Pool(processes=num_workers) as pool:
            batch_results = list(tqdm(
                pool.imap(label_single_position, args_list),
                total=len(args_list),
                desc=f"Batch {batch_idx + 1}/{num_batches}"
            ))
        
        # Extract batch results
        batch_wdl = np.zeros(batch_len, dtype=np.int8)
        batch_scores = np.zeros(batch_len, dtype=np.int16)
        
        for idx, wdl, score, error in batch_results:
            local_idx = idx - start_idx
            batch_wdl[local_idx] = wdl
            batch_scores[local_idx] = score
            if error:
                all_errors.append((idx, error))
        
        all_wdl_labels.append(batch_wdl)
        all_scores.append(batch_scores)
        
        # Clear batch from memory immediately!
        del batch_states, batch_players, batch_hashes, batch_move_seqs, args_list, batch_results
        print(f"✓ Batch {batch_idx + 1} complete, memory cleared")
    
    # Concatenate all batches
    print(f"\nConcatenating {num_batches} batches...")
    wdl_labels = np.concatenate(all_wdl_labels)
    scores = np.concatenate(all_scores)
    elapsed = time.time() - start_time
    
    # NOW load full dataset for final save (we need all data)
    print(f"Loading full dataset for final save...")
    data = np.load(positions_file, allow_pickle=True)  # Load into RAM now
    states = data['states'][:total_positions]
    players = data['players'][:total_positions]
    hashes = data['hashes'][:total_positions]
    move_seqs = data['move_sequences'][:total_positions]
    
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
    print(f"Total positions: {total_positions:,}")
    print(f"Successfully labeled: {total_positions - len(all_errors):,}")
    print(f"Errors: {len(all_errors)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Throughput: {total_positions/elapsed*3600:.0f} positions/hour")
    
    # WDL distribution
    wins = np.sum(wdl_labels == 1)
    draws = np.sum(wdl_labels == 0)
    losses = np.sum(wdl_labels == -1)
    
    print(f"\nWDL Distribution:")
    print(f"  Wins:   {wins:,} ({wins/total_positions*100:.1f}%)")
    print(f"  Draws:  {draws:,} ({draws/total_positions*100:.1f}%)")
    print(f"  Losses: {losses:,} ({losses/total_positions*100:.1f}%)")
    
    print(f"\nLabeled dataset saved to: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
    
    if all_errors:
        print(f"\nWarning: {len(all_errors)} positions failed to label")
        print("First few errors:")
        for idx, error in all_errors[:5]:
            print(f"  Position {idx}: {error}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Label positions with solver (batched)')
    parser.add_argument('--input', required=True,
                       help='Input positions file')
    parser.add_argument('--output', required=True,
                       help='Output labeled file')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Positions per batch (lower = less RAM)')
    parser.add_argument('--max', type=int, default=None,
                       help='Max positions to label (for testing)')
    
    args = parser.parse_args()
    
    label_positions_batched(
        positions_file=args.input,
        output_file=args.output,
        num_workers=args.workers,
        batch_size=args.batch_size,
        max_positions=args.max
    )
