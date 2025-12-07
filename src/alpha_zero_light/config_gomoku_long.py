"""
Configuration for long Gomoku 9x9 training (overnight)
Based on working 30min config
"""

import torch

# Training Configuration
TRAINING_CONFIG = {
    'num_iterations': 5,             # Quick diagnostic test
    'num_self_play_iterations': 512,
    'num_epochs': 8,
    'batch_size': 256,
    'temperature': 1.0,              # High early, will decay
    'dirichlet_epsilon': 0.25,       # 25% noise for exploration
    'dirichlet_alpha': 0.3,          # Noise concentration
    'eval_frequency': 10,
    'num_eval_games': 20,
    'num_sampling_moves': 10,        # Use temperature for first 10 moves
}

# MCTS Configuration
MCTS_CONFIG = {
    'num_searches': 150,             # Increased from 100 (better search)
    'C': 2,
    'num_parallel_games': 512,
    'num_sampling_moves': 10,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'temperature': 1.0,              # Will be overridden per move
}

# Model Configuration  
MODEL_CONFIG = {
    'num_res_blocks': 12,            # Medium size (balanced speed/capacity)
    'num_hidden': 384,
    'learning_rate': 0.002,
    'weight_decay': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Paths
PATHS = {
    'checkpoints': 'checkpoints/gomoku_fixed',  # New checkpoints for fixed version
    'logs': 'logs/gomoku_fixed',
}
