# Gomoku (Five in a Row) Implementation

## Overview

Gomoku (五子棋) is a traditional board game where players take turns placing stones on a 15×15 grid. The first player to get 5 stones in a row (horizontally, vertically, or diagonally) wins.

## Implementation Details

### Game Rules
- **Board Size**: 15×15 grid (225 positions)
- **Players**: Black (⚫) starts, White (⚪) follows
- **Win Condition**: 5 consecutive stones in any direction
- **No Forbidden Moves**: This is the basic variant without Renju restrictions

### Files Created

#### Game Logic
- **`src/alpha_zero_light/game/gomoku.py`**: Core game implementation
  - Board representation and state management
  - Move validation
  - Win detection in all 4 directions (horizontal, vertical, 2 diagonals)
  - State encoding for neural network

#### Training
- **`src/alpha_zero_light/config_gomoku.py`**: Training configuration
  - 100 iterations
  - 100 self-play games per iteration
  - 200 MCTS searches per move
  - Batch size: 64
  - Evaluation every iteration
  
- **`scripts/train_gomoku.py`**: Training script
  - Loads Gomoku game and configuration
  - Initializes ResNet model (8 residual blocks, 128 hidden units)
  - Runs AlphaZero training loop
  - Saves checkpoints to `checkpoints/gomoku/`

#### UI Improvements
- **`src/alpha_zero_light/ui/app.py`**: Updated Streamlit interface
  - Game selector dropdown (TicTacToe vs Gomoku)
  - Dynamic board rendering (3×3 or 15×15)
  - Improved styling with wood-colored board for Gomoku
  - Unicode stones (⚫⚪) for better visual distinction
  - Responsive layout for larger board

#### Training Control Scripts
- **`pause_training.sh`**: Gracefully stop training
- **`resume_training.sh`**: Start or resume training in background
- **`check_training.sh`**: View current training status
- **`TRAINING_CONTROLS.md`**: Quick reference guide

#### Testing
- **`test_gomoku.py`**: Unit tests for game logic
  - Initial state validation
  - Move execution
  - Valid moves calculation
  - Win detection in all directions

### Training Performance

**Current Status** (as of 2025-12-01):
- **Speed**: ~60 seconds per self-play game
- **Iteration Time**: ~100 minutes (100 games)
- **GPU Usage**: 36% utilization, 69W power draw
- **VRAM**: ~400MB (plenty of room for parallel tasks)

**Computational Complexity**:
- Gomoku is significantly more complex than TicTacToe
- 225 positions vs 9 (25× larger state space)
- Games last 50-100+ moves vs max 9
- Each game requires thousands of neural network inferences


## Usage

### Start Training
```bash
./resume_training.sh
```

### Monitor Progress
```bash
tail -f artifacts/logs/training/training_log_v2.txt
```

### Check Status
```bash
./check_training.sh
```

### Pause Training
```bash
./pause_training.sh
```

### Play Against AI
```bash
bash run_app.sh
```
Then select "Gomoku" from the dropdown in the sidebar.

## Next Steps

1. **Complete First Iteration**: Let training run to completion
2. **Evaluate Performance**: Check win rate against random player
3. **Tune Hyperparameters**: Adjust MCTS searches, learning rate if needed
4. **Add Features**: 
   - AI vs AI spectator mode
   - Opening book for common patterns
   - Heatmap visualization for MCTS probabilities

## Technical Notes

### Why is Training Slow?

Gomoku requires significantly more computation than TicTacToe:

| Metric | TicTacToe | Gomoku | Ratio |
|--------|-----------|--------|-------|
| Board Size | 3×3 = 9 | 15×15 = 225 | 25× |
| Max Game Length | 9 moves | ~100 moves | 11× |
| MCTS Searches | 100 | 200 | 2× |
| **Total Complexity** | - | - | **~550×** |

This explains why each Gomoku game takes ~60 seconds vs ~0.1 seconds for TicTacToe.

### Model Architecture

- **Input**: 3-channel encoded board (player 1, empty, player 2)
- **Network**: ResNet with 8 residual blocks, 128 hidden channels
- **Outputs**: 
  - Policy head: 225-dimensional probability distribution
  - Value head: Scalar win/loss/draw prediction

## References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [Gomoku Rules](https://en.wikipedia.org/wiki/Gomoku)
- Project Repository: https://github.com/mbenz227/alphazerolight
