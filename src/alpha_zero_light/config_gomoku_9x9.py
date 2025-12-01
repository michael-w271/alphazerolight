"""
Configuration file for AlphaZero Light - Gomoku 9x9 (Fast Training)
"""

# Training Configuration
TRAINING_CONFIG = {
    # Number of training iterations
    'num_iterations': 10,
    
    # Self-play games per iteration
    'num_self_play_iterations': 20,
    
    # Training epochs per iteration
    'num_epochs': 5,
    
    # Batch size for training
    'batch_size': 32,
    
    # Temperature for exploration
    'temperature': 1.25,
    
    # Evaluation games per iteration
    'num_eval_games': 5,
    
    # Evaluate every N iterations
    'eval_frequency': 1,
}

# MCTS Configuration
MCTS_CONFIG = {
    # Exploration constant
    'C': 2,
    
    # Number of MCTS searches per move
    'num_searches': 50,
}

# Model Configuration
MODEL_CONFIG = {
    # Number of residual blocks
    'num_res_blocks': 4,
    
    # Number of hidden channels
    'num_hidden': 64,
    
    # Learning rate
    'learning_rate': 0.001,
    
    # Weight decay
    'weight_decay': 0.0001,
}

# Paths
PATHS = {
    'checkpoints': 'checkpoints/gomoku_9x9',
    'plots': 'docs/training_plots/gomoku_9x9',
    'logs': 'logs/gomoku_9x9',
    'mlruns': 'mlruns',
}
