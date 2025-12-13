import numpy as np
import torch
from tqdm import tqdm

class Evaluator:
    """Evaluate model strength by playing against a baseline opponent"""
    
    def __init__(self, game, model, mcts):
        self.game = game
        self.model = model
        self.mcts = mcts
    
    def play_game(self, model_player=1):
        """
        Play a single game. Model plays as model_player, random plays as opponent.
        Returns: 1 if model wins, -1 if random wins, 0 if draw
        """
        # Check if game uses tensors (has .device) or numpy
        uses_tensors = hasattr(self.game, 'device')
        
        if uses_tensors:
            state = self.game.get_initial_state(1).squeeze(0)  # Get single game state
        else:
            state = self.game.get_initial_state()  # Numpy-based game
        
        player = 1
        
        while True:
            if player == model_player:
                # Model's turn - use MCTS
                if uses_tensors:
                    neutral_state = self.game.change_perspective(state.unsqueeze(0), player).squeeze(0)
                else:
                    neutral_state = self.game.change_perspective(state, player)
                action_probs = self.mcts.search(neutral_state)
                action = np.argmax(action_probs)
            else:
                # Random player's turn
                if uses_tensors:
                    valid_moves = self.game.get_valid_moves(state.unsqueeze(0))
                    if hasattr(valid_moves, 'cpu'):
                        valid_moves = valid_moves.cpu().numpy().flatten()
                else:
                    valid_moves = self.game.get_valid_moves(state)
                action = np.random.choice(self.game.action_size, p=valid_moves / np.sum(valid_moves))
            
            # Get next state
            if uses_tensors:
                state = self.game.get_next_state(state.unsqueeze(0), 
                                                torch.tensor([action], device=self.game.device),
                                                torch.tensor([player], device=self.game.device, dtype=torch.float32)).squeeze(0)
                value, is_terminal = self.game.get_value_and_terminated(state.unsqueeze(0), 
                                                                        torch.tensor([action], device=self.game.device))
            else:
                state = self.game.get_next_state(state, action, player)
                value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            # Convert to Python types
            if hasattr(value, 'item'):
                value = value.item()
            if hasattr(is_terminal, 'item'):
                is_terminal = is_terminal.item()
            
            if is_terminal:
                if value == 1:
                    # Current player won
                    return 1 if player == model_player else -1
                else:
                    # Draw
                    return 0
            
            player = self.game.get_opponent(player)
    
    def evaluate(self, num_games=20, verbose=True):
        """
        Evaluate model by playing num_games against random player.
        Returns: dict with wins, losses, draws, win_rate
        """
        wins = 0
        losses = 0
        draws = 0
        
        self.model.eval()
        
        iterator = tqdm(range(num_games), desc="Evaluation") if verbose else range(num_games)
        
        for i in iterator:
            # Alternate who plays first
            model_player = 1 if i % 2 == 0 else -1
            result = self.play_game(model_player)
            
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        
        win_rate = (wins + 0.5 * draws) / num_games
        
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'total': num_games,
            'win_rate': win_rate
        }
