# HOW ALPHAZERO LEARNS FROM WINS AND LOSSES

## Current Configuration ✅
- **MCTS Searches**: 80 (good for tactical depth)
- **Training Epochs**: 120 (ensures model learns patterns well)
- **Games per Iteration**: 400 (large dataset per iteration)
- **Value Loss Weight**: 2.0 (emphasizes learning win/loss outcomes)

## How Learning from Wins/Losses Works

### 1. Self-Play Game Collection
Every iteration, the model plays 400 games against itself:
- Games continue until someone wins or draw
- Each position before every move is stored
- Final outcome (win/loss/draw) is recorded

### 2. Outcome Assignment (THE CRITICAL PART)
When a game ends, **every position** from that game gets labeled:

```python
# Example: Player 1 wins
for each_position in game_history:
    if position_was_played_by_player_1:
        label = +1.0  # This position led to a WIN
    else:  # played by player -1
        label = -1.0  # This position led to a LOSS
```

**This is already correctly implemented!** ✅

### 3. Neural Network Training
The model learns TWO things simultaneously:

**A) Value Prediction** (Am I winning or losing?)
- Input: Board position
- Target: +1 if this player won, -1 if lost, 0 if draw
- Loss function: MSE between prediction and actual outcome
- **Value loss weight = 2.0** means we prioritize learning this!

**B) Policy Prediction** (What moves did MCTS prefer?)
- Input: Board position
- Target: MCTS visit counts (which moves were explored most)
- Loss function: Cross-entropy
- This teaches the model to mimic good MCTS decisions

### 4. Improvement Over Iterations

**Iteration 0-10**: 
- Model is random, learns basic patterns
- Starts recognizing immediate wins/losses
- High loss values

**Iteration 10-30**:
- Detects 1-move tactics (wins and blocks)
- Value predictions become more accurate
- MCTS + better policy = stronger play

**Iteration 30+**:
- Multi-move tactics emerge
- Opening theory develops
- Loss plateaus (convergence)

## Why 120 Epochs Helps

More epochs per iteration means:
- ✅ Model sees each game position 120 times
- ✅ Better gradient updates for rare positions (like winning moves)
- ✅ More stable learning
- ⚠️ Takes longer per iteration (~2-3x time)

## Key Metrics to Watch

### Training Loss
```
Total Loss = Policy Loss + (2.0 × Value Loss)
```
- Should decrease over iterations
- If stuck, model isn't learning

### Value Loss Specifically
- High value loss = poor win/loss prediction
- Low value loss = model knows who's winning
- Target: < 0.3 after 20-30 iterations

### Test Performance
- Win rate vs random opponent (should reach 95%+)
- Immediate win detection (should reach 90%+)
- Blocking threats (should reach 85%+)

## Current Status (Iteration 10)

From our tests:
- ✅ Empty board balanced (~0.08)
- ⚠️ Blocks threats (17% confidence - LEARNING)
- ❌ Misses wins (not confident yet)

**This is NORMAL for iteration 10!** The fixed MCTS means it's learning correctly now.

## What Makes It Improve?

1. **More Iterations**: Each iteration collects 400 NEW games with the BETTER model
2. **Bootstrapping**: Model plays itself, so it learns from its own improvements
3. **MCTS**: Searches 80 positions deep, finding good moves even when model is weak
4. **Correct Outcomes**: With fixed terminal bug, wins are labeled +1, losses -1

The system is self-reinforcing:
```
Better Model → Better MCTS → Better Training Data → Better Model
```

## Verification Script

To verify it's learning from wins/losses correctly, check:
1. Value predictions improve (loss decreases)
2. Tactical awareness emerges (win detection)
3. Model gets stronger vs random opponent
4. Training loss decreases steadily
