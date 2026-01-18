"""
WDL (Win/Draw/Loss) Evaluator for Alpha-Beta Engine.

Provides neural network position evaluation using WDL classification
trained on perfect solver labels. Converts WDL probabilities to scalar
scores for alpha-beta search.
"""

import torch
import numpy as np
from pathlib import Path


def wdl_to_score(wdl_probs):
    """
    Convert WDL probabilities to evaluation score.
    
    Args:
        wdl_probs: [p_win, p_draw, p_loss]
    
    Returns:
        Score in range [-1, 1]
    """
    p_win, p_draw, p_loss = wdl_probs
    # Expected value: win=1, draw=0, loss=-1
    return p_win - p_loss


class WDLEvaluator:
    """
    Neural network evaluator using WDL classification.
    
    Loads a trained WDL model and provides position evaluation
    for alpha-beta search. Supports both single and batch inference.
    """
    
    def __init__(self, model_path, game, device='cuda'):
        """
        Initialize WDL evaluator.
        
        Args:
            model_path: Path to trained WDL model checkpoint (.pt file)
            game: Game instance (Connect4)
            device: Device for inference ('cuda' or 'cpu')
        """
        from alpha_zero_light.model.wdl_network import WDLResNet
        
        self.game = game
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading WDL model from {model_path}...")
        
        # Detect architecture from checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        num_res_blocks = sum(1 for key in checkpoint.keys() if 'backbone' in key and 'conv1.weight' in key)
        num_hidden = checkpoint.get('start_block.0.weight', torch.zeros(256, 1, 1, 1)).shape[0]
        
        # Create model
        self.model = WDLResNet(game, num_res_blocks, num_hidden)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded WDL model (ResNet-{num_res_blocks}, {num_hidden} hidden) on {self.device}")
    
    def evaluate(self, state, player):
        """
        Evaluate position from player's perspective.
        
        Args:
            state: Board state (6x7 numpy array)
            player: Player to move (1 or -1)
        
        Returns:
            eval_score: Scalar evaluation in range [-1, 1]
        """
        # Encode state
        encoded = np.zeros((1, 3, 6, 7), dtype=np.float32)
        encoded[0, 0] = (state == player).astype(np.float32)
        encoded[0, 1] = (state == -player).astype(np.float32)
        encoded[0, 2] = (state[0] == 0).astype(np.float32)  # Valid moves
        
        # Get NN prediction
        with torch.no_grad():
            state_tensor = torch.from_numpy(encoded).to(self.device)
            _, wdl_logits = self.model(state_tensor)
            
            # Convert to probabilities
            wdl_probs = torch.softmax(wdl_logits, dim=1)[0].cpu().numpy()
        
        # Convert WDL to score
        eval_score = wdl_to_score(wdl_probs)
        
        return eval_score
    
    def evaluate_batch(self, states, players):
        """
        Batch evaluation for multiple positions.
        
        Args:
            states: List of board states
            players: List of players to move
        
        Returns:
            eval_scores: Numpy array of evaluations
        """
        batch_size = len(states)
        encoded = np.zeros((batch_size, 3, 6, 7), dtype=np.float32)
        
        for i, (state, player) in enumerate(zip(states, players)):
            encoded[i, 0] = (state == player).astype(np.float32)
            encoded[i, 1] = (state == -player).astype(np.float32)
            encoded[i, 2] = (state[0] == 0).astype(np.float32)
        
        # Batch inference
        with torch.no_grad():
            state_tensor = torch.from_numpy(encoded).to(self.device)
            _, wdl_logits = self.model(state_tensor)
            
            wdl_probs = torch.softmax(wdl_logits, dim=1).cpu().numpy()
        
        # Convert all to scores
        eval_scores = np.array([wdl_to_score(probs) for probs in wdl_probs])
        
        return eval_scores
    
    def get_wdl_probs(self, state, player):
        """
        Get raw WDL probabilities for a position.
        
        Args:
            state: Board state
            player: Player to move
        
        Returns:
            (p_win, p_draw, p_loss): WDL probabilities
        """
        # Encode state
        encoded = np.zeros((1, 3, 6, 7), dtype=np.float32)
        encoded[0, 0] = (state == player).astype(np.float32)
        encoded[0, 1] = (state == -player).astype(np.float32)
        encoded[0, 2] = (state[0] == 0).astype(np.float32)
        
        # Get NN prediction
        with torch.no_grad():
            state_tensor = torch.from_numpy(encoded).to(self.device)
            _, wdl_logits = self.model(state_tensor)
            
            wdl_probs = torch.softmax(wdl_logits, dim=1)[0].cpu().numpy()
        
        return tuple(wdl_probs)
    
    def get_policy_priors(self, state, player):
        """
        Get policy priors from model (for move ordering).
        
        Args:
            state: Board state
            player: Player to move
        
        Returns:
            policy_probs: (7,) array of policy probabilities
        """
        # Encode state
        encoded = np.zeros((1, 3, 6, 7), dtype=np.float32)
        encoded[0, 0] = (state == player).astype(np.float32)
        encoded[0, 1] = (state == -player).astype(np.float32)
        encoded[0, 2] = (state[0] == 0).astype(np.float32)
        
        # Get policy
        with torch.no_grad():
            state_tensor = torch.from_numpy(encoded).to(self.device)
            policy_logits, _ = self.model(state_tensor)
            
            policy_probs = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
        
        # Mask invalid moves
        valid_moves = self.game.get_valid_moves(state)
        policy_probs *= valid_moves
        policy_sum = np.sum(policy_probs)
        if policy_sum > 0:
            policy_probs /= policy_sum
        
        return policy_probs


if __name__ == '__main__':
    # Test WDL evaluator
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from alpha_zero_light.game.connect_four import ConnectFour
    
    print("Testing WDL Evaluator")
    print("="*60)
    
    game = ConnectFour()
    
    # Check for model
    model_path = Path('checkpoints/connect4/model_wdl_best.pt')
    if not model_path.exists():
        print(f"⚠ Model not found: {model_path}")
        print("Run training first: python training/scripts/train_supervised.py")
        sys.exit(1)
    
    # Test 1: Load evaluator
    print("\n✓ Test 1: Load evaluator")
    evaluator = WDLEvaluator(model_path, game, device='cuda')
    
    # Test 2: Evaluate initial position
    print("\n✓ Test 2: Evaluate initial position")
    initial_state = game.get_initial_state()
    score = evaluator.evaluate(initial_state, player=1)
    print(f"  Initial position score: {score:.4f} (should be near 0)")
    
    # Test 3: Evaluate winning position
    print("\n✓ Test 3: Evaluate winning position")
    state = game.get_initial_state()
    state[5, 0] = 1
    state[5, 1] = 1
    state[5, 2] = 1
    # X X X _ _ _ _ (player 1 can win at column 3)
    score = evaluator.evaluate(state, player=1)
    wdl = evaluator.get_wdl_probs(state, player=1)
    print(f"  Win-in-1 score: {score:.4f} (should be high)")
    print(f"  WDL: Win={wdl[0]:.3f}, Draw={wdl[1]:.3f}, Loss={wdl[2]:.3f}")
    
    # Test 4: Batch evaluation
    print("\n✓ Test 4: Batch evaluation")
    states = [initial_state, state]
    players = [1, 1]
    scores = evaluator.evaluate_batch(states, players)
    print(f"  Batch scores: {scores}")
    assert len(scores) == 2
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
