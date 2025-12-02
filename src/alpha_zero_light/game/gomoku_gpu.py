import torch
import torch.nn.functional as F
import numpy as np

class GomokuGPU:
    """
    Gomoku implementation using PyTorch tensors for massive parallelism.
    All states and operations are kept on the GPU.
    """
    def __init__(self, board_size=9, win_length=5, device='cuda'):
        self.board_size = board_size
        self.win_length = win_length
        self.device = torch.device(device)
        self.action_size = board_size * board_size
        
        # Create convolution kernels for win detection
        self.kernels = self._create_win_kernels().to(self.device)

    def _create_win_kernels(self):
        """Create 5x5 convolution kernels for detecting wins"""
        kernels = []
        
        # Horizontal
        k_h = torch.zeros((5, 5))
        k_h[2, :] = 1
        kernels.append(k_h)
        
        # Vertical
        k_v = torch.zeros((5, 5))
        k_v[:, 2] = 1
        kernels.append(k_v)
        
        # Diagonal (down-right)
        k_d1 = torch.eye(5)
        kernels.append(k_d1)
        
        # Diagonal (up-right)
        k_d2 = torch.flip(torch.eye(5), [1])
        kernels.append(k_d2)
        
        # Stack kernels: (4, 1, 5, 5) for Conv2d
        return torch.stack(kernels).unsqueeze(1)

    def get_initial_state(self, batch_size):
        """Return batch of empty boards: (batch_size, 1, H, W)"""
        return torch.zeros((batch_size, 1, self.board_size, self.board_size), 
                          device=self.device, dtype=torch.float32)

    def get_next_state(self, states, actions, players):
        """
        Update states with actions.
        states: (B, 1, H, W)
        actions: (B,) indices
        players: (B,) or scalar, 1 or -1
        """
        B = states.shape[0]
        
        # Convert actions to coordinates
        rows = actions // self.board_size
        cols = actions % self.board_size
        
        # Create batch indices
        batch_indices = torch.arange(B, device=self.device)
        
        # Update board
        # We clone to avoid modifying the original tensor in place if needed, 
        # but for efficiency we might want in-place. Let's do in-place for now.
        states[batch_indices, 0, rows, cols] = players
        
        return states

    def get_valid_moves(self, states):
        """Return mask of valid moves: (B, action_size)"""
        # 1 where empty (0), 0 where occupied
        return (states.view(states.shape[0], -1) == 0).float()

    def check_win(self, states, player):
        """
        Check if player has won in any of the states.
        states: (B, 1, H, W)
        player: scalar 1 or -1 (or tensor (B,))
        Returns: (B,) boolean tensor
        """
        # Handle batched player input
        if isinstance(player, torch.Tensor) and player.ndim > 0:
            # player is (B,), need to broadcast to (B, 1, H, W)
            player = player.view(-1, 1, 1, 1)
        
        # Filter board for current player positions (1 for player, 0 otherwise)
        player_board = (states == player).float()
        
        # Convolve with win kernels
        # Padding=0 because we only care about full 5-in-a-row fits
        # We need to handle edges correctly, but valid 5-in-a-row fits inside board
        # Conv2d output: (B, 4, H-4, W-4)
        conv = F.conv2d(player_board, self.kernels, padding=0)
        
        # Check if any value reaches 5 (win_length)
        # Max over kernels (dim 1) and spatial dims (2, 3)
        max_vals = conv.view(states.shape[0], -1).max(dim=1).values
        
        # Floating point safety (though we use 0/1, so it should be exact)
        return max_vals >= self.win_length - 0.1

    def get_value_and_terminated(self, states, last_actions, last_player):
        """
        Check wins and draws.
        Returns: 
            values: (B,) 1 if player won, 0 otherwise
            terminated: (B,) boolean
        """
        # Check if the player who just moved won
        has_won = self.check_win(states, last_player)
        
        # Check draws (board full)
        # Sum of absolute values of board should be size*size if full
        # Or just check if valid moves sum is 0
        board_full = (states.abs().sum(dim=(1, 2, 3)) >= self.action_size - 0.1)
        
        terminated = has_won | board_full
        
        # Value is 1 if won, 0 if draw (or not terminated)
        # Note: If not terminated, value doesn't matter much, usually 0
        values = has_won.float()
        
        return values, terminated

    def get_encoded_state(self, states):
        """
        Encode state for neural network.
        Input: (B, 1, H, W) with 1, -1, 0
        Output: (B, 3, H, W) - [P1, P2, Color] or similar
        AlphaZero usually uses: [My Stones, Opponent Stones, To Play]
        """
        # Assuming states are from perspective of player to move?
        # Or we handle perspective change separately.
        # Let's assume standard encoding:
        # Channel 0: Player 1 stones (1)
        # Channel 1: Player -1 stones (-1)
        # Channel 2: All 1s (if P1 to move) or 0s? Usually handled by canonical form.
        
        # For simplicity here:
        # 0: My stones
        # 1: Opponent stones
        # 2: To play (all 1)
        
        # This requires knowing who is to play. 
        # If we assume canonical form (always 1 to move), then:
        
        p1_stones = (states == 1).float()
        p2_stones = (states == -1).float()
        to_play = torch.ones_like(states) # Canonical: always 1 to move
        
        return torch.cat([p1_stones, p2_stones, to_play], dim=1)

    def change_perspective(self, states, player):
        """
        Flip board values so that 'player' becomes 1.
        states: (B, 1, H, W)
        player: scalar or (B,)
        """
        if isinstance(player, torch.Tensor):
            player = player.view(-1, 1, 1, 1)
        return states * player
