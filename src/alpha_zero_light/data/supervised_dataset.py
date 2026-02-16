"""
Supervised dataset loader for WDL-labeled Connect4 positions.

Loads solver-labeled positions for supervised training of WDL value head.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SupervisedWDLDataset(Dataset):
    """
    Dataset for supervised WDL training from solver labels.
    
    Loads NPZ files with:
    - states: (N, 6, 7) board positions
    - wdl_labels: (N,) solver labels {-1, 0, 1}
    - players: (N,) player to move
    """
    
    def __init__(self, npz_path, augment=True):
        """
        Args:
            npz_path: Path to labeled NPZ file
            augment: Apply horizontal flip augmentation
        """
        data = np.load(npz_path, allow_pickle=True)
        
        self.states = data['states'].astype(np.float32)
        self.players = data['players'].astype(np.int8)
        self.wdl_labels = data['wdl_labels'].astype(np.int8)
        
        self.augment = augment
        
        print(f"Loaded {len(self.states):,} positions from {npz_path}")
        
        # WDL distribution
        wins = np.sum(self.wdl_labels == 1)
        draws = np.sum(self.wdl_labels == 0)
        losses = np.sum(self.wdl_labels == -1)
        print(f"  Wins:   {wins:,} ({wins/len(self.states)*100:.1f}%)")
        print(f"  Draws:  {draws:,} ({draws/len(self.states)*100:.1f}%)")
        print(f"  Losses: {losses:,} ({losses/len(self.states)*100:.1f}%)")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        # Get state (6x7 board)
        state = self.states[idx]
        player = self.players[idx]
        wdl_label = self.wdl_labels[idx]
        
        # Apply horizontal flip augmentation (50% chance)
        if self.augment and np.random.rand() < 0.5:
            state = np.fliplr(state)
        
        # Encode state with 3 channels for ResNet
        # Channel 0: Current player pieces
        # Channel 1: Opponent pieces  
        # Channel 2: Valid moves
        encoded = np.zeros((3, 6, 7), dtype=np.float32)
        encoded[0] = (state == player).astype(np.float32)
        encoded[1] = (state == -player).astype(np.float32)
        encoded[2] = (state[0] == 0).astype(np.float32)  # Top row empty = valid

        
        # Convert WDL label to class index: {-1, 0, 1} → {2, 1, 0}
        # WIN (1) → class 0
        # DRAW (0) → class 1
        # LOSS (-1) → class 2
        wdl_class = (1 - wdl_label) // 1  # Maps: 1→0, 0→1, -1→2
        
        return {
            'state': torch.from_numpy(encoded),
            'wdl_label': torch.tensor(wdl_class, dtype=torch.long)
        }


def create_dataloaders(
    train_path,
    val_path=None,
    batch_size=256,
    num_workers=4,
    val_split=0.2
):
    """
    Create train and validation dataloaders.
    
    Args:
        train_path: Path to training NPZ
        val_path: Optional path to separate validation NPZ
        batch_size: Batch size
        num_workers: DataLoader workers
        val_split: Validation split if val_path not provided
    
    Returns:
        (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = SupervisedWDLDataset(train_path, augment=True)
    
    if val_path:
        # Separate validation file
        val_dataset = SupervisedWDLDataset(val_path, augment=False)
        train_dataset = full_dataset
    else:
        # Split from training data
        n_train = int(len(full_dataset) * (1 - val_split))
        n_val = len(full_dataset) - n_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Split dataset: {n_train:,} train, {n_val:,} val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    dataset = SupervisedWDLDataset('data/selfplay_42k_labeled.npz')
    
    print(f"\nDataset test:")
    print(f"  Length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"  Sample state shape: {sample['state'].shape}")
    print(f"  Sample WDL label: {sample['wdl_label']}")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        'data/selfplay_42k_labeled.npz',
        batch_size=64,
        val_split=0.2
    )
    
    print(f"\nDataLoader test:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"  Batch state shape: {batch['state'].shape}")
    print(f"  Batch WDL shape: {batch['wdl_label'].shape}")
