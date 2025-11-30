# AlphaZero Tic-Tac-Toe: Comprehensive Training Documentation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [How AlphaZero Works](#how-alphazero-works)
3. [Game Rules: Tic-Tac-Toe](#game-rules-tic-tac-toe)
4. [The Three Pillars of the System](#the-three-pillars-of-the-system)
5. [The Training Process](#the-training-process)
6. [Training Progress and Improvements](#training-progress-and-improvements)
7. [Technical Details](#technical-details)
8. [How to Try It Yourself](#how-to-try-it-yourself)

---

## 🎯 Project Overview

This project implements **AlphaZero**, a revolutionary AI technology from DeepMind that learns to master games – **without ever being shown human strategies**. The AI learns only by playing against itself (self-play), continuously improving and developing its own, often surprising strategies.

We've implemented AlphaZero for **Tic-Tac-Toe** – a simple game that's perfect for understanding the fundamental principles.

### What makes this project special?

- ✅ **Fully self-taught**: No pre-programmed strategies
- ✅ **GPU-accelerated**: Training on an NVIDIA RTX 5080
- ✅ **Interactive visualization**: Play against the AI and see its thought processes
- ✅ **Traceable evolution**: Watch how the AI learns over 50 iterations

---

## 🤖 How AlphaZero Works

AlphaZero combines three revolutionary techniques:

### 1. **Deep Neural Network**
The AI's "brain". It consists of multiple layers of artificial neurons that learn to predict two things:

- **Policy**: Which move is most promising in this situation?
- **Value**: How good is this position? (Win probability)

**Architecture**:
```
Input: 3x3 board (3 channels: X pieces, Empty squares, O pieces)
    ↓
Start Block: Convolution → Batch Normalization → ReLU
    ↓
4x ResNet Blocks (Residual Connections for deep learning)
    ↓
    ├─→ Policy Head → 9 probabilities (for each square)
    └─→ Value Head → 1 number between -1 (loss) and +1 (win)
```

### 2. **Monte Carlo Tree Search (MCTS)**
An intelligent search method that simulates and evaluates moves:

1. **Selection**: Follow the most promising path in the search tree
2. **Expansion**: Add new possible moves
3. **Simulation**: Use the neural network for evaluation
4. **Backpropagation**: Update all nodes in the path with the result

**UCB Formula** (Upper Confidence Bound):
```
UCB = Q + C × √(N_parent / N_child) × P
```
- `Q`: Average evaluation (Exploitation)
- `C × √(...)`: Exploration bonus (Exploration)
- `P`: Prior probability from the neural network

### 3. **Self-Play**
The AI plays against itself to generate training data:

1. Both players use the same neural network
2. MCTS is used to find the best moves
3. Each position is stored along with the final game outcome
4. This data is used to train the neural network

---

## 🎮 Game Rules: Tic-Tac-Toe

Tic-Tac-Toe is a simple strategy game for two players on a 3×3 grid.

### Gameplay

1. **Player X** starts and places an X in an empty square
2. **Player O** places an O in another empty square
3. Players alternate placing their symbols
4. **Winner**: First to get three symbols in a row (horizontal, vertical, or diagonal)
5. **Draw**: When all 9 squares are filled without a winner

### Implementation of Rules

```python
# Board representation: 3x3 NumPy Array
# 0 = Empty, 1 = Player 1 (X), -1 = Player 2 (O)

def check_win(state, action):
    """Check if the last move resulted in a win"""
    row, col = action // 3, action % 3
    player = state[row, col]
    
    # Check row, column, and diagonals
    return (
        np.sum(state[row, :]) == player * 3 or  # Row
        np.sum(state[:, col]) == player * 3 or  # Column
        np.sum(np.diag(state)) == player * 3 or  # Diagonal ↘
        np.sum(np.diag(np.flip(state, axis=1))) == player * 3  # Diagonal ↙
    )
```

### Perspective Change

Since the neural network always thinks from Player 1's perspective, we need to "flip" the board when Player -1 is to move:

```python
def change_perspective(state, player):
    """Multiply all values by the current player"""
    return state * player
```

---

## 🏗️ The Three Pillars of the System

### 1. **Game Engine** (`TicTacToe`)

The game logic that provides the following functions:

| Function | Description | Example |
|----------|-------------|---------|
| `get_initial_state()` | Empty 3×3 board | `[[0,0,0], [0,0,0], [0,0,0]]` |
| `get_next_state(state, action, player)` | Executes a move | Place X in center: `action=4` |
| `get_valid_moves(state)` | Which squares are empty? | `[1,1,0,1,0,1,1,1,1]` |
| `get_value_and_terminated(state, action)` | Game over? Who won? | `(1, True)` = Win |
| `check_win(state, action)` | Checks win condition | 3 in a row? |

### 2. **Neural Network** (`ResNet`)

A **Residual Network** with the following components:

#### ResNet Architecture

```
Input (3×3×3):
  ├─ Channel 0: Positions of Player -1 (O)
  ├─ Channel 1: Empty squares
  └─ Channel 2: Positions of Player 1 (X)

↓ Start Block
├─ Conv2D (3→64 channels, Kernel 3×3)
├─ BatchNorm2D
└─ ReLU

↓ Backbone (4× ResBlocks)
├─ ResBlock 1
│   ├─ Conv → BN → ReLU
│   ├─ Conv → BN
│   └─ + Residual Connection → ReLU
├─ ResBlock 2, 3, 4 ...
└─ Output: 64 channels, 3×3

↓ Split into two heads

Policy Head:              Value Head:
├─ Conv (64→32)           ├─ Conv (64→3)
├─ BN → ReLU              ├─ BN → ReLU
├─ Flatten                ├─ Flatten
└─ Linear → 9 Outputs     └─ Linear → Tanh
   (Probability for          (Evaluation -1 to +1)
    each move)
```

#### Why Residual Connections?

Residual blocks solve the **Vanishing Gradient Problem** in deep networks:

```python
def forward(x):
    residual = x  # Save input
    x = conv1(x)
    x = bn1(x)
    x = relu(x)
    x = conv2(x)
    x = bn2(x)
    x += residual  # Add original input
    x = relu(x)
    return x
```

### 3. **Monte Carlo Tree Search** (`MCTS`)

MCTS builds a search tree to find the best decision:

#### The Search Process (200 Simulations)

```
                    Root (Current board)
                    /       |       \
              Move 0    Move 1    Move 2   ... (9 possible moves)
               /  \      /  \      /  \
          [Further moves expanded by priority]
```

Each node stores:
- `visit_count`: How often was this move examined?
- `value_sum`: Total evaluation of all simulations
- `prior`: Probability from the neural network

**Selecting the best move**:
After 200 simulations, MCTS chooses the move with the **most visits** (not the highest value!), because this is more robust.

---

## 🎓 The Training Process

### Step-by-Step Process of a Training Iteration

#### 1. **Self-Play Phase** (100 Games)

```python
for game in range(100):
    state = get_initial_state()  # Empty board
    memory = []  # Storage for training data
    
    while not game_over:
        # MCTS searches for best move (200 simulations)
        action_probs = mcts.search(state)
        
        # Store (state, probabilities, player)
        memory.append((state, action_probs, current_player))
        
        # Choose move with temperature sampling
        action = sample(action_probs, temperature=1.25)
        state = get_next_state(state, action, current_player)
        
        # Switch player
        current_player = -current_player
    
    # Evaluate all positions with final game result
    for (state, probs, player) in memory:
        outcome = final_result if player == winner else -final_result
        training_data.append((state, probs, outcome))
```

**Temperature Sampling**: Adds randomness for exploration
```python
temp_probs = probs ** (1 / temperature)
temp_probs /= sum(temp_probs)
action = random.choice(actions, p=temp_probs)
```

#### 2. **Training Phase** (10 Epochs)

```python
for epoch in range(10):
    shuffle(training_data)  # Important for SGD
    
    for batch in batches(training_data, size=64):
        states, target_policies, target_values = zip(*batch)
        
        # Forward Pass
        pred_policy, pred_value = neural_network(states)
        
        # Loss Calculation
        policy_loss = CrossEntropy(pred_policy, target_policies)
        value_loss = MSE(pred_value, target_values)
        total_loss = policy_loss + value_loss
        
        # Backward Pass (Gradient Descent)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

**Loss Functions**:
- **Policy Loss** (Cross-Entropy): Measures difference between MCTS probabilities and predictions
- **Value Loss** (Mean Squared Error): Measures difference between actual game outcome and prediction

#### 3. **Evaluation Phase** (Every 5 Iterations)

```python
def evaluate(num_games=20):
    wins, losses, draws = 0, 0, 0
    
    for game in range(num_games):
        # AI plays as X, random player as O
        result = play_game(ai_player=1, random_player=-1)
        
        if result == 1: wins += 1
        elif result == -1: losses += 1
        else: draws += 1
    
    win_rate = wins / num_games
    return {"win_rate": win_rate, "wins": wins, "losses": losses, "draws": draws}
```

---

## 📊 Training Progress and Improvements

### Today's Training Progress (November 30, 2025)

#### Iteration 0 (Untrained)
- **Total Loss**: 2.15
- **Self-Play**: 20 games, 60 MCTS simulations
- **Behavior**: Random moves, no strategy

#### Iteration 8 (First Training)
- **Total Loss**: 1.34 (-38%)
- **Win Rate**: 87.5% vs Random
- **Result**: Still too weak against humans

### 🚀 Improvements Implemented

#### 1. **Increased Self-Play Intensity**
```diff
- 'num_self_play_iterations': 20
+ 'num_self_play_iterations': 100  # 5× more data
```
**Effect**: More diverse positions for robust learning

#### 2. **Deeper MCTS Search**
```diff
- 'num_searches': 60
+ 'num_searches': 200  # 3.3× deeper search
```
**Effect**: Better tactical decisions

#### 3. **Longer Training per Iteration**
```diff
- 'num_epochs': 4
+ 'num_epochs': 10  # 2.5× more training steps
```
**Effect**: Better convergence, stronger learning

#### Iteration 49 (Final Model)
- **Total Loss**: 0.25 (-88% from start)
- **Policy Loss**: 0.25
- **Value Loss**: 0.00 (nearly perfect!)
- **Win Rate**: 82.5% (14W, 5D, 1L)
- **Behavior**: Plays strategically, forces draws when uncertain

### Visualization of Improvement

![Training Metrics](docs/training_plots/training_metrics.png)

**Observations**:
1. **Loss decreases continuously**: Shows successful learning
2. **Win rate stabilizes**: AI has reached a high level
3. **More draws**: AI plays more cautiously and avoids risks

---

## 🔧 Technical Details

### Hardware & Software Stack

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA GeForce RTX 5080 (CUDA Acceleration) |
| **Framework** | PyTorch 2.x |
| **Virtual Environment** | `azl` (Miniforge3) |
| **UI Framework** | Streamlit |
| **Visualization** | Matplotlib |

### Training Configuration

```python
TRAINING_CONFIG = {
    'num_iterations': 50,           # Number of training iterations
    'num_self_play_iterations': 100, # Games per iteration
    'num_epochs': 10,                # Training epochs
    'batch_size': 64,                # Batch size for SGD
    'temperature': 1.25,             # Exploration parameter
    'num_eval_games': 20,            # Evaluation games
    'eval_frequency': 5,             # Evaluate every 5 iterations
}

MCTS_CONFIG = {
    'C': 2,                          # Exploration constant (UCB)
    'num_searches': 200,             # MCTS simulations per move
}

MODEL_CONFIG = {
    'num_res_blocks': 4,             # Number of ResNet blocks
    'num_hidden': 64,                # Number of hidden channels
    'learning_rate': 0.001,          # Adam Optimizer LR
    'weight_decay': 0.0001,          # L2 regularization
}
```

### Training Time

**Total Duration**: ~16 minutes for 50 iterations on RTX 5080

Breakdown per iteration:
- Self-Play (100 games): ~12 seconds
- Training (10 epochs): ~5 seconds
- Evaluation (every 5 iter.): ~2 seconds

---

## 🎮 How to Try It Yourself

### Installation

```bash
# Clone repository
git clone https://github.com/mbenz227/alphazerolight
cd alpha-zero-light

# Create virtual environment
conda create -n azl python=3.10
conda activate azl

# Install dependencies
pip install -r requirements.txt
```

### Start Training

```bash
# Simple with preconfigured script
./run_training.sh

# Or manually
python scripts/run_train.py
```

### Start Game UI

```bash
# With script
./run_app.sh

# Or manually
streamlit run src/alpha_zero_light/ui/app.py
```

### UI Features

#### 1. **Play Game** Tab
- Play against the trained AI
- See MCTS distribution after each AI move
- Real-time feedback

#### 2. **Training Evolution** Tab
- View training metrics (plots)
- Replay function: See how the AI played in each iteration
- Step-by-step walkthrough of historical games

### Manage Checkpoints

```bash
# Delete old checkpoints
./clean_checkpoints.sh

# Start training from scratch
./run_training.sh
```

---

## 🧠 Why Does AlphaZero Work So Well?

### 1. **Self-Supervised Learning**
No human data needed – the AI learns from the consequences of its decisions.

### 2. **Policy + Value Dual Heads**
- **Policy**: Quick pre-selection of good moves
- **Value**: Direct position evaluation
- **Combination**: MCTS can search deeper and more efficiently

### 3. **Iterative Improvement**
Each iteration:
- Improves the neural network → better MCTS search
- Better MCTS search → better training data
- Better training data → better neural network
- **Positive feedback loop!**

### 4. **Monte Carlo Tree Search**
Balance between:
- **Exploitation**: Use what you already know
- **Exploration**: Try new, uncertain moves

---

## 📈 Next Steps

### Possible Extensions

1. **More Complex Games**
   - Connect Four
   - Othello / Reversi
   - Chess

2. **Improved Training**
   - Experience Replay Buffer (larger memory)
   - Prioritized Experience Replay
   - Distributed Training (multi-GPU)

3. **Enhanced Visualization**
   - 3D search tree visualization
   - Live training dashboard
   - Heatmaps for position evaluations

4. **AI vs AI Tournaments**
   - Different iterations play against each other
   - Elo rating system
   - Identify best strategies

---

## 📚 Further Resources

### Papers
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero)
- [A general reinforcement learning algorithm that masters chess, shogi, and Go](https://science.sciencemag.org/content/362/6419/1140) (AlphaZero)

### Code & Tutorials
- [Official AlphaZero Pseudocode](https://github.com/deepmind/alphazero)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MCTS Tutorial](https://www.youtube.com/watch?v=UXW2yZndl7U)

---

## 🙏 Acknowledgments

- **DeepMind**: For groundbreaking AlphaZero research
- **PyTorch Team**: For the excellent deep learning framework
- **Streamlit**: For simple UI development
- **Community**: For open-source implementations and tutorials

---

## 📝 Summary

This project demonstrates how modern AI systems can autonomously learn to master complex strategy games through **Self-Play**, **Deep Learning**, and **Monte Carlo Tree Search**.

**Key Takeaways**:
1. ✅ AI doesn't need human data – it can learn from self-play
2. ✅ Combination of neural networks and tree search is very powerful
3. ✅ Iterative training leads to continuous improvement
4. ✅ Even simple games offer deep insights into AI mechanisms

**Project Repository**: [github.com/mbenz227/alphazerolight](https://github.com/mbenz227/alphazerolight)

---

**Created**: November 30, 2025  
**Version**: 1.0  
**Author**: Michael
