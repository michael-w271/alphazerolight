import torch
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import PATHS, MODEL_CONFIG

def get_random_action(game, state):
    valid_moves = game.get_valid_moves(state)
    valid_indices = np.where(valid_moves == 1)[0]
    return np.random.choice(valid_indices)

def play_game(game, model, args, device):
    state = game.get_initial_state()
    mcts = MCTS(game, args, model)
    
    # Store game history: list of {board: list, player: int, action: int}
    history = []
    
    player = 1
    while True:
        # Record state before move
        state_list = state.tolist()
        
        if player == 1:
            # AI Move
            mcts_probs = mcts.search(state)
            action = np.argmax(mcts_probs)
        else:
            # Random Opponent Move
            action = get_random_action(game, state)
            
        history.append({
            'board': state_list,
            'player': player,
            'action': int(action)
        })
        
        state = game.get_next_state(state, action, player)
        
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            # Record final state
            history.append({
                'board': state.tolist(),
                'player': game.get_opponent(player), # Next player (who won't move)
                'action': None,
                'winner': value if player == 1 else -value # value is relative to current player
            })
            return history, value if player == 1 else -value
            
        player = game.get_opponent(player)

def main():
    print("Generating evolution replay...")
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = Path(PATHS.checkpoints)
    # Sort by iteration number
    checkpoints = sorted(checkpoint_dir.glob('model_*.pt'), key=lambda x: int(x.stem.split('_')[1]))
    
    evolution_data = []
    
    args = {
        'C': 2,
        'num_searches': 60
    }
    
    for cp in checkpoints:
        iteration = int(cp.stem.split('_')[1])
        print(f"Processing iteration {iteration}...")
        
        model = ResNet(game, 
                      num_res_blocks=MODEL_CONFIG['num_res_blocks'], 
                      num_hidden=MODEL_CONFIG['num_hidden']).to(device)
        model.load_state_dict(torch.load(cp, map_location=device))
        model.eval()
        
        # Play one game as Player 1 (X) vs Random
        game_history, result = play_game(game, model, args, device)
        
        evolution_data.append({
            'iteration': iteration,
            'checkpoint': cp.name,
            'moves': game_history,
            'result': result # 1 (AI won), -1 (AI lost), 0 (Draw)
        })
        
    output_path = Path('docs/connect4_evolution_replay.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evolution_data, f)
        
    print(f"Saved evolution replay to {output_path}")

if __name__ == "__main__":
    main()
