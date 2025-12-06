"""
Configuration for 30-minute GPU training run
Target: High throughput, fast initial learning
"""

# Training Configuration - 30 MINUTE RUN
TRAINING_CONFIG = {
    # 30 mins total. ~5-6 mins per iteration.
    'num_iterations': 6,
    
    # Self-play games per iteration
    # High parallelism: 2048 games in parallel (Increased for GPU utilization)
    'num_self_play_iterations': 2048,
    
    # Training epochs per iteration
    'num_epochs': 5,
    
    # Batch size for training
    'batch_size': 2048,
    
    # Temperature for exploration
    'temperature': 1.25,
    
    # Evaluation games per iteration
    'num_eval_games': 20,
    
    # Evaluate every iteration to see progress
    'eval_frequency': 1,
}

# MCTS Configuration
MCTS_CONFIG = {
    # Exploration constant
    'C': 2,
    
    # Number of MCTS searches per move
    # Lower count for speed during self-play
    'num_searches': 100,
}

# Model Configuration
MODEL_CONFIG = {
    # Number of residual blocks
    'num_res_blocks': 20,
    
    # Number of hidden channels
    'num_hidden': 256,
    
    # Learning rate
    'learning_rate': 0.002,
    
    # Weight decay
    'weight_decay': 0.0001,
}

# Paths
PATHS = {
    'checkpoints': 'checkpoints/gomoku_30min',
    'plots': 'docs/training_plots', # Save directly to website assets source
    'logs': 'logs/gomoku_30min',
}
