"""
Configuration file for AlphaZero Light - Gomoku 9x9 (Overnight Training)
Optimized for 6 hours of training with ~75% GPU usage
HIGH GPU UTILIZATION VERSION
"""

# Training Configuration
TRAINING_CONFIG = {
    # Number of training iterations
    'num_iterations': 30,
    
    # Self-play games per iteration
    'num_self_play_iterations': 2048,
    
    # Training epochs per iteration
    'num_epochs': 10,
    
    # Batch size for training (INCREASED for GPU utilization)
    'batch_size': 2048,
    
    # Temperature for exploration
    'temperature': 1.25,
    
    # Evaluation games per iteration
    'num_eval_games': 10,
    
    # Evaluate every N iterations
    'eval_frequency': 5,
}

# MCTS Configuration
MCTS_CONFIG = {
    # Exploration constant
    'C': 2,
    
    # Number of MCTS searches per move (INCREASED)
    'num_searches': 200,
}

# Model Configuration - INCREASED for GPU utilization
MODEL_CONFIG = {
    # Number of residual blocks (INCREASED: 4 -> 12)
    'num_res_blocks': 12,
    
    # Number of hidden channels (INCREASED: 64 -> 256)
    'num_hidden': 256,
    
    # Learning rate
    'learning_rate': 0.001,
    
    # Weight decay
    'weight_decay': 0.0001,
}

# Paths
PATHS = {
    'checkpoints': 'checkpoints/gomoku_9x9_overnight',
    'plots': 'docs/training_plots/gomoku_9x9_overnight',
    'logs': 'logs/gomoku_9x9_overnight',
    'mlruns': 'mlruns',
}
