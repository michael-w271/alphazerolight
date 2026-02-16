# Walkthrough - v0.1 Tic-Tac-Toe AlphaZero

I have successfully implemented the v0.1 milestone of the AlphaZero Light project. This includes the core components for Tic-Tac-Toe, the Neural Network, MCTS, and the Training Loop.

## Changes
### Core Components
- **Game**: Implemented `TicTacToe` class with efficient numpy-based state management.
- **Model**: Implemented `ResNet` with Policy and Value heads.
- **MCTS**: Implemented `MCTS` with PUCT algorithm and parallel-friendly structure.
- **Training**: Implemented `AlphaZeroTrainer` for self-play and training.

### Scripts
- **run_train.py**: A script to execute the training loop.

## Verification Results
### Training Run
I executed a short training run with the following parameters:
- **Iterations**: 3
- **Self-Play Episodes per Iteration**: 10
- **Epochs per Iteration**: 4

The training completed successfully without errors.

### Generated Artifacts
The following model checkpoints were generated:
- `model_0.pt`, `optimizer_0.pt`
- `model_1.pt`, `optimizer_1.pt`
- `model_2.pt`, `optimizer_2.pt`

## Next Steps
- Implement the Evaluation script to test the strength of the trained models.
- Implement the Streamlit UI to visualize the game and play against the agent.
