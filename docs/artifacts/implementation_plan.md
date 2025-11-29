# Implementation Plan - v0.1 Tic-Tac-Toe AlphaZero

## Goal Description
Build a compact, scalable AlphaZero-style engine starting with Tic-Tac-Toe. This phase focuses on setting up the core architecture: Game interface, MCTS, Neural Network, and the Training Loop.

## User Review Required
- **Dependencies**: Confirming PyTorch (CUDA 12.4), MLflow, Streamlit.
- **Directory Structure**: Standard `src` layout.

## Proposed Changes

### Project Structure
#### [NEW] [README.md](file:///home/michael/.gemini/antigravity/scratch/alpha-zero-light/README.md)
#### [NEW] [project.json](file:///home/michael/.gemini/antigravity/scratch/alpha-zero-light/project.json)
#### [NEW] [requirements.txt](file:///home/michael/.gemini/antigravity/scratch/alpha-zero-light/requirements.txt)

### Source Code (`src/alpha_zero_light`)
#### [NEW] `game/game.py`: Abstract Base Class for Games.
#### [NEW] `game/tictactoe.py`: Tic-Tac-Toe implementation.
#### [NEW] `model/network.py`: PyTorch CNN model.
#### [NEW] `mcts/mcts.py`: Monte Carlo Tree Search implementation.
#### [NEW] `training/trainer.py`: Self-play and training loop.

## Verification Plan
### Automated Tests
- Unit tests for Game logic (Tic-Tac-Toe rules).
- Unit tests for MCTS (check if it finds winning moves in obvious states).
- Tensor shape checks for Neural Network.

### Manual Verification
- Run a short training loop (e.g., 10 iterations) and check if loss decreases.
- Play a game against the untrained/partially trained agent via CLI.
