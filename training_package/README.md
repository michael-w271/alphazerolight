# Gomoku Training Package

This package contains all files needed for AlphaZero Gomoku 9x9 training.

## Files:

### Scripts
- `train_gomoku_long.py` - Main training script
- `eval_quick.py` - Quick evaluation vs random player
- `config_gomoku_long.py` - Training configuration

### Core Modules
- `training/` - Trainer and evaluator
- `mcts/` - Monte Carlo Tree Search (C++ backed)
- `model/` - ResNet neural network
- `game/` - Gomoku game logic (CPU and GPU versions)

## Current Configuration:
- **Model:** 12 res blocks, 384 hidden (~4M parameters)
- **MCTS:** 100 searches
- **Games:** 512 per iteration
- **Iterations:** 250
- **Checkpoints:** checkpoints/gomoku_medium/

## To Run:
```bash
OMP_NUM_THREADS=24 python train_gomoku_long.py
```

## To Evaluate:
```bash
python eval_quick.py
```

## Key Parameters to Tune:
1. **num_res_blocks** (12) - Model depth
2. **num_hidden** (384) - Model width  
3. **num_searches** (100) - MCTS depth
4. **num_self_play_iterations** (512) - Games per iteration
5. **learning_rate** (0.002) - Training speed
