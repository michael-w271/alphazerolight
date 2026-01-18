#!/usr/bin/env python3
"""
Fix corrupted selfplay_42k.npz by removing commas from move sequences.

Bug: Checkpoint save used ','.join() creating "3,2,3,3"
Fix: Remove commas to create "3233" format that solver expects
"""

import numpy as np
from pathlib import Path

print("="*70)
print("FIXING CORRUPTED MOVE SEQUENCES")
print("="*70)

# Load corrupted file
print("\nLoading corrupted dataset: data/selfplay_42k.npz...")
data = np.load('data/selfplay_42k.npz', allow_pickle=True)

print(f"✓ Loaded {len(data['states']):,} positions")

# Extract all data
states = data['states']
players = data['players']
hashes = data['hashes']
move_sequences = data['move_sequences']

print(f"\nBefore fix:")
print(f"  Sample move sequence: \"{move_sequences[0]}\"")
print(f"  Length: {len(str(move_sequences[0]))}")

# Fix move sequences by removing commas
print(f"\nRemoving commas from {len(move_sequences):,} move sequences...")
fixed_move_sequences = np.array([
    str(seq).replace(',', '')  # Remove all commas
    for seq in move_sequences
], dtype=object)

print(f"\nAfter fix:")
print(f"  Sample move sequence: \"{fixed_move_sequences[0]}\"")
print(f"  Length: {len(str(fixed_move_sequences[0]))}")

# Verify format matches working dataset
print(f"\nVerifying fix...")
print(f"  Sample sequences:")
for i in [0, 100, 1000, 6907]:
    orig = str(move_sequences[i])
    fixed = str(fixed_move_sequences[i])
    print(f"    {i}: \"{orig}\" → \"{fixed}\"")

# Save fixed dataset
output_file = 'data/selfplay_42k_fixed.npz'
print(f"\nSaving fixed dataset to {output_file}...")

np.savez_compressed(
    output_file,
    states=states,
    players=players,
    hashes=hashes,
    move_sequences=fixed_move_sequences
)

print(f"\n{'='*70}")
print("FIX COMPLETE!")
print(f"{'='*70}")
print(f"✓ Fixed {len(states):,} positions")
print(f"✓ Saved to: {output_file}")
print(f"✓ File size: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
print(f"\nReady for labeling!")
print(f"\nNext command:")
print(f"  python training/scripts/label_dataset_batched.py \\")
print(f"    --input {output_file} \\")
print(f"    --output data/selfplay_42k_labeled.npz \\")
print(f"    --workers 6 --batch-size 5000")
