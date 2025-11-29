# AlphaZero Light - Project Tasks

## v0.1: Tic-Tac-Toe AlphaZero
- [x] **Project Setup**
    - [x] Create project structure and files
    - [x] Setup virtual environment and install dependencies
- [x] **Game Implementation (Tic-Tac-Toe)**
    - [x] Implement `Game` interface
    - [x] Implement Tic-Tac-Toe logic (board, moves, win check)
    - [x] Implement state encoding for Neural Network
- [x] **Neural Network (Small CNN)**
    - [x] Design CNN architecture (Policy & Value heads)
    - [x] Implement PyTorch model
- [x] **MCTS Implementation**
    - [x] Implement Node class
    - [x] Implement MCTS logic (Select, Expand, Simulate/Evaluate, Backprop)
    - [x] Integrate with Neural Network
- [x] **Training Loop**
    - [x] Implement Self-Play data generation
    - [x] Implement Training pipeline (Loss calculation, Backprop)
    - [x] Basic evaluation (vs Random)

## v0.2: Connect4 AlphaZero
- [ ] **Game Extension**
    - [ ] Implement Connect4 logic
    - [ ] Update state encoding
- [ ] **Model & MCTS Updates**
    - [ ] Adjust NN architecture for larger board
    - [ ] Tune MCTS hyperparameters
- [ ] **Experiment Tracking**
    - [ ] Integrate MLflow

## v0.3: MiniChess (5x5) AlphaZero
- [ ] **Game Extension**
    - [ ] Implement 5x5 Chess logic
    - [ ] Implement Chess state encoding
- [ ] **Refinement**
    - [ ] Test NN and MCTS on Chess complexity

## v1.0: Full Chess AlphaZero-Light
- [ ] **Full Chess Implementation**
    - [ ] Standard 8x8 Chess logic
    - [ ] Residual Network Architecture
    - [ ] Advanced MCTS optimizations

## v1.1: UI + Visualization
- [ ] **Streamlit UI**
    - [ ] Interactive Board
    - [ ] MCTS Visualization (Heatmaps, Stats)

## v1.2: Documentation & Polish
- [ ] **Documentation**
    - [ ] Architecture Overview
    - [ ] Design Decisions
    - [ ] MCTS & NN Explanations
- [ ] **Code Cleanup**
    - [ ] Refactoring
    - [ ] Optimization
