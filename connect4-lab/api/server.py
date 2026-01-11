#!/usr/bin/env python3
"""
Flask API server for Connect4 AI predictions using model_120.pt
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import sys
from pathlib import Path

import sys
from pathlib import Path

# Add parent directory's src to path to import alpha_zero_light
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS

app = Flask(__name__)
CORS(app)  # Enable CORS for React dev server

# Global model and MCTS
game = None
model = None
mcts = None
device = None

def load_model():
    """Load the Connect4 model on startup"""
    global game, model, mcts, device
    
    print("Loading Connect4 model...")
    checkpoint_path = Path("/mnt/ssd2pro/alpha-zero-checkpoints/connect4/model_120.pt")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found at {checkpoint_path}")
    
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(game, num_res_blocks=10, num_hidden=128).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Create MCTS with moderate search count for responsive UI
    args = {
        'C': 2.0,
        'num_searches': 100,  # Balance between speed and quality
        'dirichlet_epsilon': 0.0,  # No exploration in play mode
        'dirichlet_alpha': 0.3,
        'mcts_batch_size': 1,
    }
    mcts = MCTS(game, args, model)
    
    print(f"✅ Model loaded successfully on {device}")
    return True

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Get AI prediction for current board state
    
    Request JSON:
    {
        "board": [[0,0,0,0,0,0,0], ...],  # 6x7 array
        "player": 2  # AI player (1 or 2)
    }
    
    Response JSON:
    {
        "q_values": [0.1, 0.2, ...],  # float[7] - estimated value per column
        "policy": [0.05, 0.15, ...],   # float[7] - move probabilities
        "legal_mask": [true, true, ...], # bool[7] - which columns have space
        "chosen_column": 3,             # int - best move
        "value": 0.45                   # float - position evaluation
    }
    """
    try:
        data = request.json
        board_list = data.get('board')
        player = data.get('player', 2)
        
        if not board_list or len(board_list) != 6 or len(board_list[0]) != 7:
            return jsonify({'error': 'Invalid board format'}), 400
        
        # Convert to numpy array
        board = np.array(board_list, dtype=np.float32)
        
        # Get AI's perspective (change perspective to AI player)
        # Board values: 0=empty, 1=player1, 2=player2
        # Convert to AlphaZero format: 0=empty, 1=current, -1=opponent
        ai_board = np.zeros((6, 7), dtype=np.float32)
        for i in range(6):
            for j in range(7):
                if board[i][j] == player:
                    ai_board[i][j] = 1  # AI's pieces
                elif board[i][j] != 0:
                    ai_board[i][j] = -1  # Opponent's pieces
        
        # Get MCTS prediction
        action_probs = mcts.search(ai_board, add_noise=False)
        
        # Get valid moves
        valid_moves = game.get_valid_moves(ai_board)
        legal_mask = [bool(vm) for vm in valid_moves]
        
        # Get Q-values (visit counts serve as Q-values proxy)
        q_values = action_probs.tolist()
        
        # Find best move
        masked_probs = action_probs * valid_moves
        if np.sum(masked_probs) > 0:
            chosen_column = int(np.argmax(masked_probs))
        else:
            # Fallback: random valid move
            valid_cols = np.where(valid_moves)[0]
            chosen_column = int(valid_cols[0]) if len(valid_cols) > 0 else 3
        
        # Get value estimate from network
        with torch.no_grad():
            encoded = game.get_encoded_state(ai_board)
            state_tensor = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
            policy_logits, value_tensor = model(state_tensor)
            value = float(value_tensor.item())
        
        return jsonify({
            'q_values': q_values,
            'policy': action_probs.tolist(),
            'legal_mask': legal_mask,
            'chosen_column': chosen_column,
            'value': value
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        load_model()
        print("\n" + "="*60)
        print("🚀 Connect4 API Server Starting")
        print("="*60)
        print("API Endpoints:")
        print("  GET  /api/health  - Health check")
        print("  POST /api/predict - Get AI prediction")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
