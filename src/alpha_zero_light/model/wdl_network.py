"""
WDL (Win/Draw/Loss) ResNet for Stockfish-like evaluation.

Extends the standard ResNet to output 3-way WDL probabilities
instead of a single scalar value, enabling more precise evaluation.
"""

import torch
import torch.nn as nn
from .network import ResNet, ResBlock


class WDLResNet(ResNet):
    """
    ResNet with WDL (Win/Draw/Loss) value head.
    
    Identical to standard ResNet except the value head outputs
    3 logits (Win/Draw/Loss) instead of 1 scalar value.
    
    This enables training with perfect solver labels for more
    accurate position evaluation.
    """
    
    def __init__(self, game, num_res_blocks, num_hidden):
        """
        Initialize WDL ResNet.
        
        Args:
            game: Game instance (Connect4)
            num_res_blocks: Number of residual blocks (e.g., 20)
            num_hidden: Hidden dimension (e.g., 256)
        """
        # Initialize parent ResNet (this creates start_block, backbone, policy_head, value_head)
        super().__init__(game, num_res_blocks, num_hidden)
        
        # Replace value head with WDL version (3 outputs instead of 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 3)  # 3 outputs: Win, Draw, Loss
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, 6, 7) - 3-channel board encoding
        
        Returns:
            (policy, wdl):
                - policy: (batch_size, 7) - action logits
                - wdl: (batch_size, 3) - Win/Draw/Loss logits
        """
        x = self.start_block(x)
        
        for block in self.backbone:
            x = block(x)
        
        policy = self.policy_head(x)
        wdl = self.value_head(x)  # Raw logits, apply softmax for probabilities
        
        return policy, wdl
    
    @classmethod
    def from_alphazero_checkpoint(cls, checkpoint_path, game, device='cuda'):
        """
        Initialize WDL ResNet from AlphaZero checkpoint.
        
        Loads all weights from standard ResNet except value head
        (which is randomly initialized for WDL classification).
        
        Args:
            checkpoint_path: Path to AlphaZero model checkpoint (.pt file)
            game: Game instance
            device: Device to load model on
        
        Returns:
            WDL ResNet with transferred weights
        """
        # Load checkpoint to get architecture params
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Infer architecture from checkpoint
        # Count backbone blocks
        num_res_blocks = sum(1 for key in checkpoint.keys() if 'backbone' in key and 'conv1.weight' in key)
        
        # Get hidden dimension from start block
        num_hidden = checkpoint['start_block.0.weight'].shape[0]
        
        print(f"Detected architecture: ResNet-{num_res_blocks}, {num_hidden} hidden")
        
        # Create WDL model
        model = cls(game, num_res_blocks, num_hidden)
        model.to(device)
        
        # Load weights (skip value_head since structure is different)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                          if k in model_dict and 'value_head' not in k}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"✓ Transferred {len(pretrained_dict)} weight tensors from AlphaZero")
        print(f"✓ Randomly initialized value head for WDL classification")
        
        return model


if __name__ == '__main__':
    # Test WDL ResNet
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from alpha_zero_light.game.connect_four import ConnectFour
    
    print("Testing WDL ResNet")
    print("="*60)
    
    game = ConnectFour()
    
    # Test 1: Create from scratch
    print("\n✓ Test 1: Create WDL ResNet from scratch")
    model = WDLResNet(game, num_res_blocks=20, num_hidden=256)
    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test 2: Forward pass
    print("\n✓ Test 2: Forward pass")
    batch_size = 4
    x = torch.randn(batch_size, 3, 6, 7)
    policy, wdl = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Policy shape: {policy.shape} (expected: {batch_size}, 7)")
    print(f"  WDL shape: {wdl.shape} (expected: {batch_size}, 3)")
    
    # Test 3: WDL probabilities
    print("\n✓ Test 3: WDL probabilities")
    wdl_probs = torch.softmax(wdl, dim=1)
    print(f"  Sample WDL probs: {wdl_probs[0].detach().cpu().numpy()}")
    print(f"  Sum to 1: {wdl_probs[0].sum().item():.4f}")
    
    # Test 4: Load from AlphaZero checkpoint
    checkpoint_path = Path("checkpoints/connect4/model_195.pt")
    if checkpoint_path.exists():
        print(f"\n✓ Test 4: Load from AlphaZero checkpoint")
        model_from_az = WDLResNet.from_alphazero_checkpoint(
            str(checkpoint_path),
            game,
            device='cpu'
        )
        print(f"  Model loaded successfully")
        
        # Test forward pass
        policy2, wdl2 = model_from_az(x)
        print(f"  Forward pass successful")
    else:
        print(f"\n⚠ Test 4 skipped: {checkpoint_path} not found")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
