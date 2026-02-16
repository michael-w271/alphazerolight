"""
Position generator for Connect4 dataset creation.

Generates diverse Connect4 positions for supervised training:
1. Self-play positions from existing AlphaZero model
2. Random opening book positions (first 8-10 moves)
3. Noise-injected positions for diversity
4. Focus on mid/late-game (10+ moves) for faster solver labeling
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.engine.zobrist import get_zobrist_hasher


class PositionGenerator:
    """
    Generate diverse Connect4 positions for dataset creation.
    
    Strategy:
    - Generate positions with 10-20 moves played (mid/late-game)
    - These solve faster than early positions (<60s vs >120s)
    - Use mix of self-play, random, and noise-injected moves
    """
    
    def __init__(self, game: Optional[ConnectFour] = None, seed: int = 42):
        """
        Initialize position generator.
        
        Args:
            game: Connect4 game instance
            seed: Random seed for reproducibility
        """
        self.game = game or ConnectFour()
        self.zobrist = get_zobrist_hasher()
        self.rng = np.random.RandomState(seed)
        
        # Deduplication set
        self.generated_hashes = set()
    
    def generate_random_positions(
        self, 
        num_positions: int,
        moves_min: int = 10,
        moves_max: int = 20
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Generate random Connect4 positions.
        
        Args:
            num_positions: Number of positions to generate
            moves_min: Minimum number of moves to play
            moves_max: Maximum number of moves to play
        
        Returns:
            List of (state, player_to_move, zobrist_hash) tuples
        """
        positions = []
        
        for _ in tqdm(range(num_positions), desc="Generating random positions"):
            # Random number of moves
            num_moves = self.rng.randint(moves_min, moves_max + 1)
            
            state = self.game.get_initial_state()
            player = 1
            
            for move_idx in range(num_moves):
                valid_moves = self.game.get_valid_moves(state)
                
                if np.sum(valid_moves) == 0:
                    break  # Board full
                
                # Choose random valid move
                valid_indices = np.where(valid_moves)[0]
                action = self.rng.choice(valid_indices)
                
                # Make move
                state = self.game.get_next_state(state, action, player)
                
                # Check if game ended
                value, terminated = self.game.get_value_and_terminated(state, action)
                if terminated:
                    break
                
                player = -player
            
            # Only add if game not over  
            if np.sum(self.game.get_valid_moves(state)) > 0:
                hash_val = self.zobrist.hash_position(state, player)
                
                # Dedup check
                if hash_val not in self.generated_hashes:
                    self.generated_hashes.add(hash_val)
                    positions.append((state.copy(), player, hash_val))
        
        return positions
    
    def generate_opening_positions(
        self,
        num_positions: int,
        opening_moves: int = 8
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Generate positions from opening book (systematic exploration).
        
        Plays first N moves systematically to cover opening space.
        
        Args:
            num_positions: Number of positions to generate
            opening_moves: Number of opening moves to play
        
        Returns:
            List of (state, player_to_move, zobrist_hash) tuples
        """
        positions = []
        
        # Generate opening sequences
        for _ in tqdm(range(num_positions), desc="Generating opening positions"):
            state = self.game.get_initial_state()
            player = 1
            
            # Play opening moves with some randomness
            for _ in range(opening_moves):
                valid_moves = self.game.get_valid_moves(state)
                
                if np.sum(valid_moves) == 0:
                    break
                
                valid_indices = np.where(valid_moves)[0]
                
                # Weighted towards center columns
                weights = np.array([0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5])
                weights = weights[valid_indices]
                weights = weights / weights.sum()
                
                action = self.rng.choice(valid_indices, p=weights)
                state = self.game.get_next_state(state, action, player)
                
                # Check termination
                value, terminated = self.game.get_value_and_terminated(state, action)
                if terminated:
                    break
                
                player = -player
            
            # Then play random moves to reach mid-game (10-15 total moves)
            for _ in range(self.rng.randint(2, 8)):
                valid_moves = self.game.get_valid_moves(state)
                
                if np.sum(valid_moves) == 0:
                    break
                
                valid_indices = np.where(valid_moves)[0]
                action = self.rng.choice(valid_indices)
                state = self.game.get_next_state(state, action, player)
                
                value, terminated = self.game.get_value_and_terminated(state, action)
                if terminated:
                    break
                
                player = -player
            
            # Add if valid
            if np.sum(self.game.get_valid_moves(state)) > 0:
                hash_val = self.zobrist.hash_position(state, player)
                
                if hash_val not in self.generated_hashes:
                    self.generated_hashes.add(hash_val)
                    positions.append((state.copy(), player, hash_val))
        
        return positions
    
    def generate_mixed_dataset(
        self,
        total_positions: int,
        random_ratio: float = 0.6,
        opening_ratio: float = 0.4
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Generate mixed dataset with multiple position sources.
        
        Args:
            total_positions: Total number of positions to generate
            random_ratio: Fraction of random positions
            opening_ratio: Fraction of opening-based positions
        
        Returns:
            List of (state, player_to_move, zobrist_hash) tuples
        """
        num_random = int(total_positions * random_ratio)
        num_opening = int(total_positions * opening_ratio)
        
        print(f"Generating {total_positions:,} positions:")
        print(f"  - Random: {num_random:,}")
        print(f"  - Opening-based: {num_opening:,}")
        
        all_positions = []
        
        # Generate random positions
        all_positions.extend(self.generate_random_positions(num_random))
        
        # Generate opening positions
        all_positions.extend(self.generate_opening_positions(num_opening))
        
        # Shuffle
        indices = self.rng.permutation(len(all_positions))
        all_positions = [all_positions[i] for i in indices]
        
        print(f"\nGenerated {len(all_positions):,} unique positions")
        print(f"Deduplication removed {total_positions - len(all_positions):,} duplicates")
        
        return all_positions


def save_positions_npz(
    positions: List[Tuple[np.ndarray, int, int]],
    output_path: str
):
    """
    Save positions to compressed NPZ file.
    
    Args:
        positions: List of (state, player, hash) tuples
        output_path: Path to save file
    """
    states = np.array([pos[0] for pos in positions], dtype=np.float32)
    players = np.array([pos[1] for pos in positions], dtype=np.int8)
    hashes = np.array([pos[2] for pos in positions], dtype=np.uint64)
    
    np.savez_compressed(
        output_path,
        states=states,
        players=players,
        hashes=hashes
    )
    
    print(f"Saved {len(positions):,} positions to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")


def load_positions_npz(input_path: str) -> List[Tuple[np.ndarray, int, int]]:
    """Load positions from NPZ file."""
    data = np.load(input_path)
    
    positions = []
    for i in range(len(data['states'])):
        positions.append((
            data['states'][i],
            int(data['players'][i]),
            int(data['hashes'][i])
        ))
    
    return positions


if __name__ == '__main__':
    # Test dataset generation
    print("Testing Position Generator")
    print("="*60)
    
    generator = PositionGenerator(seed=42)
    
    # Generate small test dataset
    positions = generator.generate_mixed_dataset(
        total_positions=1000,
        random_ratio=0.7,
        opening_ratio=0.3
    )
    
    # Save to file
    output_path = Path("tests/fixtures/connect4/test_positions.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_positions_npz(positions, str(output_path))
    
    # Load and verify
    print(f"\nVerifying saved data...")
    loaded = load_positions_npz(str(output_path))
    print(f"Loaded {len(loaded):,} positions")
    
    # Show sample position
    print(f"\nSample position:")
    print(f"Player to move: {loaded[0][1]}")
    print(f"Hash: {loaded[0][2]}")
    print("Board:")  
    from alpha_zero_light.game.connect_four import ConnectFour
    game = ConnectFour()
    state = loaded[0][0]
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for row in range(6):
        print("  " + " ".join(symbols[state[row, col]] for col in range(7)))
    
    print("\n✓ Dataset generation test complete!")
