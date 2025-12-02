"""
Configuration for 1-hour GPU training test
Target: ~200W GPU usage (80% instead of 100%)
"""

# Training Configuration - 1 HOUR TEST
TRAINING_CONFIG = {
    # Small number of iterations for 1h test
    'num_iterations': 6,  # ~10 min per iteration = 1 hour
    
    # Self-play games per iteration
    'num_self_play_iterations': 200,
    
    # Training epochs per iteration
    'num_epochs': 10,
    
    # Batch size for training
    'batch_size': 512,
    
    # Temperature for exploration
    'temperature': 1.25,
    
    # Evaluation games per iteration
    'num_eval_games': 10,
    
    # Evaluate every N iterations
    'eval_frequency': 2,
}

# MCTS Configuration
MCTS_CONFIG = {
    # Exploration constant
    'C': 2,
    
    # Number of MCTS searches per move
    'num_searches': 150,  # Reduced from 200 for speed
}

# Model Configuration - Reduced for ~200W (20% less than max)
MODEL_CONFIG = {
    # Number of residual blocks (reduced from 20)
    'num_res_blocks': 15,
    
    # Number of hidden channels (reduced from 512)
    'num_hidden': 384,
    
    # Learning rate
    'learning_rate': 0.001,
    
    # Weight decay
    'weight_decay': 0.0001,
}

# Paths
PATHS = {
    'checkpoints': 'checkpoints/gomoku_9x9_gpu_test',
    'plots': 'docs/training_plots/gomoku_9x9_gpu_test',
    'logs': 'logs/gomoku_9x9_gpu_test',
}
