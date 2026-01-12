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

import threading
import time
import os
import re

# Watcher config
CHECKPOINT_DIR = Path("/mnt/ssd2pro/alpha-zero-light/checkpoints/connect4")
CHECK_INTERVAL_SECONDS = 15
current_iteration = 0
current_model_name = "None"
last_loaded_time = 0
auto_reload_enabled = True  # Default to auto
target_model_file = None  # If manually selected

def find_latest_model():
    """Find the model with the highest iteration number"""
    if not CHECKPOINT_DIR.exists():
        return None, 0
    
    max_iter = -1
    best_model = None
    
    # Scan for model_X.pt files
    pattern = re.compile(r"model_(\d+).pt")
    try:
        for f in os.listdir(CHECKPOINT_DIR):
            match = pattern.match(f)
            if match:
                iteration = int(match.group(1))
                if iteration > max_iter:
                    max_iter = iteration
                    best_model = CHECKPOINT_DIR / f
    except Exception as e:
        print(f"Scanning error: {e}")
                
    return best_model, max_iter

def get_available_models():
    """Get list of all available models"""
    models = []
    if not CHECKPOINT_DIR.exists():
        return models
        
    pattern = re.compile(r"model_(\d+).pt")
    try:
        for f in os.listdir(CHECKPOINT_DIR):
            match = pattern.match(f)
            if match:
                iteration = int(match.group(1))
                models.append({
                    'filename': f,
                    'iteration': iteration,
                    'path': str(CHECKPOINT_DIR / f)
                })
    except Exception as e:
        print(f"Scanning error: {e}")
    
    # Sort by iteration (descending)
    models.sort(key=lambda x: x['iteration'], reverse=True)
    return models

def load_model(specific_filename=None):
    """Load the Connect4 model (specific filename or latest auto-detected)"""
    global game, model, mcts, device, current_iteration, current_model_name, last_loaded_time
    
    checkpoint_path = None
    iter_num = 0
    
    if specific_filename:
        checkpoint_path = CHECKPOINT_DIR / specific_filename
        match = re.search(r"model_(\d+).pt", specific_filename)
        if match:
            iter_num = int(match.group(1))
    else:
        checkpoint_path, iter_num = find_latest_model()
    
    if not checkpoint_path or not checkpoint_path.exists():
        print(f"⚠️ Model not found: {checkpoint_path}")
        return False
        
    print(f"🔄 Loading Connect4 model: {checkpoint_path.name} (Iteration {iter_num})...")
    
    try:
        if game is None:
            game = ConnectFour()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model structure (ResNet-20, 256 hidden for new training)
        # Fallback to smaller model if loading fails (backward compatibility)
        try:
            # TRY NEW MAX CONFIG FIRST
            temp_model = ResNet(game, num_res_blocks=20, num_hidden=256).to(device)
            state_dict = torch.load(checkpoint_path, map_location=device)
            temp_model.load_state_dict(state_dict)
            model = temp_model
        except Exception as e:
            print(f"⚠️ Failed to load as Max Config (ResNet-20): {e}")
            print("Trying legacy config (ResNet-10)...")
            temp_model = ResNet(game, num_res_blocks=10, num_hidden=128).to(device)
            state_dict = torch.load(checkpoint_path, map_location=device)
            temp_model.load_state_dict(state_dict)
            model = temp_model
            
        model.eval()
        
        # Dynamic MCTS searches based on iteration to reflect true strength curve
        # Iter 0-25: 50 searches (Fast/Weak)
        # Iter 25-50: 100 searches
        # Iter 50-75: 200 searches
        # Iter 100+: 400 searches (or more for inference)
        
        if iter_num < 25:
            inference_searches = 50
        elif iter_num < 50:
            inference_searches = 100
        elif iter_num < 75:
            inference_searches = 200
        else:
            inference_searches = 400  # Full strength for mature models
            
        # Create MCTS with dynamic search count
        args = {
            'C': 2.0,
            'num_searches': inference_searches, 
            'dirichlet_epsilon': 0.0,  # No exploration in play mode
            'dirichlet_alpha': 0.3,
            'mcts_batch_size': 1,
        }
        mcts = MCTS(game, args, model)
        
        current_iteration = iter_num
        current_model_name = checkpoint_path.name
        last_loaded_time = time.time()
        print(f"✅ Loaded {checkpoint_path.name} successfully on {device} (MCTS: {inference_searches})")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def model_watcher():
    """Background thread to auto-reload newer models"""
    global auto_reload_enabled
    while True:
        try:
            if auto_reload_enabled:
                _, latest_iter = find_latest_model()
                # If we have a new model OR we haven't loaded anything yet
                if latest_iter > current_iteration or (current_iteration == 0 and latest_iter > 0):
                    print(f"✨ New model detected (Iter {latest_iter} > {current_iteration})")
                    load_model()
        except Exception as e:
            print(f"Watcher error: {e}")
        
        time.sleep(CHECK_INTERVAL_SECONDS)

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models"""
    return jsonify({
        'models': get_available_models(),
        'current_model': current_model_name,
        'current_iteration': current_iteration,
        'auto_reload': auto_reload_enabled
    })

@app.route('/api/model', methods=['POST'])
def select_model():
    """Select a specific model or switch to auto mode"""
    global auto_reload_enabled
    data = request.json
    mode = data.get('mode', 'manual')  # 'auto' or 'manual'
    filename = data.get('filename')
    
    if mode == 'auto':
        auto_reload_enabled = True
        # Immediate check
        load_model()
        return jsonify({
            'status': 'success', 
            'message': 'Switched to Auto-Reload mode',
            'current_model': current_model_name
        })
    else:
        if not filename:
            return jsonify({'error': 'Filename required for manual mode'}), 400
            
        auto_reload_enabled = False
        success = load_model(specific_filename=filename)
        if success:
            return jsonify({
                'status': 'success', 
                'message': f'Loaded {filename}',
                'current_model': current_model_name
            })
        else:
            return jsonify({'error': f'Failed to load {filename}'}), 404

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'current_model': current_model_name,
        'current_iteration': current_iteration,
        'auto_reload': auto_reload_enabled,
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
        mcts_searches = data.get('mcts_searches')  # Optional override
        
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
        
        # Update MCTS searches if requested
        if mcts_searches is not None:
             mcts.args['num_searches'] = int(mcts_searches)
        
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
            'value': value,
            'model_iteration': current_iteration,  # Report which model made the move
            'model_name': current_model_name,
            'mcts_searches': mcts.args['num_searches']
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
        
        print(f"   Watching for new models in: {CHECKPOINT_DIR}")
        print(f"   Auto-reload interval: {CHECKPOINT_DIR}s")
        
        # Start watcher thread
        watcher = threading.Thread(target=model_watcher, daemon=True)
        watcher.start()
        
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
