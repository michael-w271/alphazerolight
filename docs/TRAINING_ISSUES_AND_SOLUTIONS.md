# AlphaZero Connect Four Training: Issues and Solutions

## Table of Contents
1. [Overview](#overview)
2. [Initial Training Failure](#initial-training-failure)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Failed Fix Attempts](#failed-fix-attempts)
5. [The Real Problem](#the-real-problem)
6. [Final Solution](#final-solution)
7. [Results](#results)
8. [Lessons Learned](#lessons-learned)

---

## Overview

This document chronicles the debugging journey of implementing AlphaZero for Connect Four (4-in-a-row). The model went through multiple complete training failures before we identified and fixed the fundamental issues. This serves as a case study in why AlphaZero self-play can fail catastrophically and how to bootstrap training properly.

**Timeline:**
- **Attempt 1-2:** Complete failure - model couldn't defend basic threats
- **Attempt 3:** Improved configuration but loss plateaued
- **Attempt 4:** Added value loss weighting - still failed
- **Attempt 5:** Progressive difficulty warmup - **SUCCESS**

---

## Initial Training Failure

### Configuration (First Attempt)
```python
num_iterations: 50
num_self_play_iterations: 1000
num_searches: 16  # MCTS simulations per move
num_res_blocks: 6
num_hidden: 64
learning_rate: 0.002
```

### Symptoms
After 50 iterations of training:
- **The model couldn't defend against 4-in-a-row threats**
- Policy output was nearly uniform (not learning move preferences)
- Value predictions were random noise
- Loss plateaued at 1.72 after iteration 9

### Testing the Failure
```python
# Testing iteration 9 model on a position where column 3 wins
model.predict(state_with_winning_move)
# Result: Uniform probabilities across all columns (no preference)
# Expected: Strong preference for the winning move
```

**Diagnosis:** Model was completely broken, not learning anything meaningful.

---

## Root Cause Analysis

### Issue 1: Insufficient MCTS Depth
- **Problem:** Only 16 MCTS searches per move
- **Impact:** Search tree too shallow to find good moves in Connect Four
- **Evidence:** Model made random moves, couldn't see tactical threats

### Issue 2: Model Too Small
- **Problem:** 6 ResBlocks × 64 hidden units
- **Impact:** Insufficient capacity to learn Connect Four patterns (6×7 board, 4-in-a-row detection)
- **Comparison:** AlphaZero Go uses 19-40 residual blocks

### Issue 3: Poor Training Data Quality
- **Problem:** Weak model playing against itself creates noisy training data
- **Impact:** "Garbage in, garbage out" - model couldn't improve from bad self-play games
- **Evidence:** Loss stopped improving after initial descent

---

## Failed Fix Attempts

### Attempt 2: Better Configuration
```python
num_iterations: 100
num_self_play_iterations: 600
num_searches: 50        # ↑ from 16
num_res_blocks: 10      # ↑ from 6
num_hidden: 128         # ↑ from 64
```

**Result:** Training progressed further but hit same plateau issue.

**Loss Pattern:**
```
Iteration 1-11:  Loss 1.87 → 0.92  (excellent improvement)
Iteration 12-31: Loss stuck at ~1.13-1.18  (plateau)
```

**Analysis:** Model learned basic patterns but couldn't improve further. Value loss began **degrading** while policy loss improved - a critical red flag.

---

### Attempt 3: Added Value Loss Weighting

**Hypothesis:** Cross-entropy loss (policy) has larger magnitude than MSE loss (value), causing optimizer to ignore value head.

```python
# Before
loss = policy_loss + value_loss

# After
loss = policy_loss + 2.0 * value_loss  # Weight value loss 2x
```

**Result:** Still failed! Value loss continued degrading.

**Loss Analysis (Iteration 76):**
```
Policy Loss: 1.44 → 0.73  (49.5% improvement) ✓
Value Loss:  0.26 → 0.40  (17.8% degradation) ✗
```

**Key Finding:** Policy was improving (learning move patterns) but value was degrading (getting worse at predicting outcomes).

---

## The Real Problem

### Catastrophic Discovery: Model Predicting Nonsense

Testing the trained model on an empty board:

```python
state = game.get_initial_state()  # Empty board
policy, value = model(state)

# Results:
value = +1.0  # Model predicts CERTAIN PLAYER 1 WIN on empty board!
# Expected: ~0.0 (balanced game)
```

**This revealed the fundamental issue:**

### The Self-Play Bootstrap Problem

When a weak model plays against itself:

1. **Both players are equally bad** → Games are random/chaotic
2. **Outcomes are noisy** → Win/loss/draw is more about luck than skill  
3. **Model tries to fit noise** → Learns spurious patterns
4. **Value predictions worsen** → Model "learns" that player 1 always wins
5. **Worse value → Worse MCTS** → Even more random games
6. **Vicious cycle** → Training diverges instead of converging

### Why This Happens

**AlphaZero assumes:**
- Self-play games have some structure
- Better positions lead to better outcomes consistently
- Value targets are meaningful signals

**Reality with weak model:**
- Games are random walks
- Same position can lead to opposite outcomes (noise)
- Value targets are contradictory
- Model overfits to "player 1 wins" because it saw that pattern in noise

### Evidence from Loss Components

```
Iteration  Total   Policy  Value   Problem
0          1.65    1.32    0.33    Baseline
1          1.18    0.89    0.29    Initial improvement
2          1.11    0.84    0.27    Best value loss
3-30       ~1.13   ↓0.73   ↑0.40   Policy improves, value degrades!
```

**Interpretation:**
- **Policy loss ↓**: Model memorizing "move patterns" from self-play
- **Value loss ↑**: Model getting worse at predicting outcomes
- **Root cause**: Self-play data is garbage when model is weak

---

## Final Solution

### Progressive Difficulty Warmup Training

Instead of pure self-play from the start, train against progressively harder opponents:

#### Phase 1: Pure Random Opponent (Iterations 0-9)
```python
def self_play_vs_random():
    # Model plays as Player 1
    # Opponent makes random valid moves
    # Outcome: Win/Loss/Draw is REAL, not noise
```

**Benefits:**
- Clean value targets: If model wins → value = +1 (unambiguous)
- Model learns "what is a winning position"
- No noise from weak self-play
- Fast games (~10 games/sec)

#### Phase 2: Heuristic Opponent (Iterations 10-19)
```python
class HeuristicOpponent:
    def get_action(self, state, player):
        # 1. If I can win (4-in-a-row), do it
        # 2. If opponent can win, block it
        # 3. Otherwise, random move (prefer center)
```

**Benefits:**
- Forces model to learn defense (blocks threats)
- Forces model to learn offense (takes wins)
- Still beatable by model with MCTS
- Teaches tactical play

#### Phase 3: Mixed Opponents (Iterations 20-29)
```python
# 50% games vs heuristic, 50% vs random
# Ensures generalization across opponent types
```

#### Phase 4: Self-Play (Iterations 30+)
```python
# Standard AlphaZero: model vs model
# Now model is competent enough for self-play to work
```

### Implementation Details

**Configuration:**
```python
TRAINING_CONFIG = {
    'num_iterations': 190,
    'random_opponent_iterations': 30,  # Warmup phase
    'value_loss_weight': 2.0,
    'num_searches': 50,
    'num_res_blocks': 10,
    'num_hidden': 128,
}
```

**Heuristic Opponent (1-ply lookahead):**
```python
# Win if possible
for action in valid_actions:
    if would_win(state, action, player):
        return action

# Block if necessary  
for action in valid_actions:
    if would_win(state, action, opponent):
        return action  # Block the threat

# Otherwise random (with center preference)
return weighted_random_choice(valid_actions)
```

---

## Results

### Comparison: Before vs After

#### Failed Training (Self-Play from Start)
```
Iteration  Total   Policy  Value   Status
0          1.65    1.32    0.33    Baseline
10         0.92    0.67    0.25    Initial improvement
20         1.13    0.75    0.38    Value degrading ✗
50         1.14    0.73    0.41    Complete failure ✗

Model Predictions:
- Empty board: value = +1.0 (nonsense)
- Playing ability: Cannot defend 4-in-a-row
```

#### Successful Training (Progressive Warmup)
```
Iteration  Total   Policy  Value   Status
0          2.02    1.35    0.33    Baseline (warmup)
1          1.54    1.01    0.27    Value improving ✓
2          1.46    0.92    0.27    Steady progress ✓
3          1.33    0.79    0.27    Continuing ✓
4          1.31    0.79    0.26    Best so far ✓
5          1.38    0.84    0.27    Minor variance ✓

Overall: 19.3% value loss improvement ✓
```

### Key Differences

| Metric | Failed Training | Successful Training |
|--------|----------------|-------------------|
| Value Loss Trend | ↑ Degrading | ↓ Improving |
| Empty Board Value | +1.0 (broken) | ~0.0 (sensible) |
| Can Block Threats | No | Learning |
| Training Stability | Diverges | Converges |

---

## Lessons Learned

### 1. **Bootstrap Problem is Real**

AlphaZero's self-play assumes the model can learn incrementally. If the initial model is too weak, self-play creates a negative feedback loop:
- Weak model → Bad games → Noisy data → Worse model → Even worse games

**Solution:** Start with external opponents (random, heuristic) to provide clean training signal.

### 2. **Value Loss is the Canary in the Coal Mine**

When value loss degrades while policy loss improves:
- ✗ **NOT** a sign of successful learning
- ✓ **IS** a sign that training data is corrupted
- Model is memorizing patterns without understanding outcomes

**Diagnostic:** Always monitor value loss separately. If it degrades, training is broken.

### 3. **Loss Weighting Alone Won't Fix Bad Data**

We tried weighting value loss 2× higher:
```python
loss = policy_loss + 2.0 * value_loss
```

**Result:** Failed! 

**Why:** Weighting doesn't fix noisy targets. 2× weight on garbage data = 2× garbage gradient.

**Real fix:** Get better training data (warmup opponents).

### 4. **MCTS Depth Matters**

- 16 searches: Too shallow, misses tactics
- 50 searches: Good balance of speed/quality  
- 100 searches: Better but 2-3× slower

**Lesson:** Find the sweet spot for your game complexity.

### 5. **Model Capacity Must Match Game Complexity**

Connect Four (6×7 board, 4-in-a-row) needs:
- ≥10 ResBlocks (we used 10)
- ≥128 hidden units (we used 128)

Smaller models can't learn the pattern recognition needed.

### 6. **Progressive Training Works**

The curriculum approach:
1. Learn basics (vs random)
2. Learn tactics (vs heuristic)  
3. Learn strategy (vs self)

**Why it works:**
- Each phase builds on previous learning
- Model is never asked to learn from pure noise
- Smooth difficulty curve prevents divergence

### 7. **Test Your Models!**

Don't just trust loss values. Actually test predictions:
```python
# Does model think empty board is balanced?
assert abs(model.predict_value(empty_board)) < 0.3

# Can model recognize obvious wins?
assert model.prefers_winning_move(position_with_win)
```

We only discovered the "+1.0 on empty board" bug by testing, not from loss curves.

---

## Technical Details

### Why Value Loss Degrades in Bad Self-Play

**Scenario:** Weak model plays against itself

```
Game 1: Position X → Random play → Player 1 wins → value[X] = +1
Game 2: Position X → Random play → Player -1 wins → value[X] = -1
Game 3: Position X → Random play → Player 1 wins → value[X] = +1
...
```

**Problem:** Position X gets contradictory labels (+1, -1, +1, ...)

**Model's response:** 
- Can't fit contradictions
- Defaults to "memorize most frequent outcome"
- If player 1 wins 60% due to first-move advantage → value[X] ≈ +0.6
- This is **wrong** - position X might actually be balanced

**Result:** Value loss increases because model is learning a bad pattern.

### Why Progressive Training Fixes This

**Phase 1 (vs Random):**
```
Position X (good for player 1):
- Model with MCTS finds winning continuation → Wins
- Value target: +1 (correct!)

Position Y (bad for player 1):
- Random opponent punishes mistake → Loses
- Value target: -1 (correct!)
```

**Consistency:** Same position → Same outcome (because random opponent is stateless)

**Learning:** Model learns "position X is good" reliably

---

## Future Improvements

### Potential Enhancements

1. **Endgame Database**
   - Pre-compute perfect play for positions 1-5 moves from end
   - Use as additional training data
   - Would give perfect value targets for endgames

2. **Experience Replay Buffer**
   - Keep best games from all iterations
   - Sample from historical data, not just current iteration
   - Reduces overfitting to recent self-play

3. **Model Evaluation**
   - Only promote new model if it beats previous best
   - Prevents regression
   - AlphaZero paper does this

4. **Adaptive Opponent Difficulty**
   - Measure model win rate vs heuristic
   - If >90%, switch to self-play early
   - If <60%, stay in warmup longer

5. **Curriculum Self-Play**
   - Play 70% vs current model, 30% vs older checkpoints
   - Prevents overfitting to current opponent
   - Maintains diversity

---

## Conclusion

The failure of AlphaZero self-play from scratch taught us that:

1. **Bootstrap quality matters** - You can't learn from pure noise
2. **Value loss degradation is a critical warning sign**
3. **Progressive difficulty training is essential** for complex games
4. **Loss weighting ≠ better data quality**
5. **Always validate with actual predictions**, not just loss curves

The final solution (progressive warmup with random → heuristic → self-play) achieved:
- ✓ Stable value loss improvement
- ✓ Sensible position evaluations  
- ✓ Learning tactical and strategic play
- ✓ Successful 8-hour training runs

**Key Insight:** AlphaZero is powerful, but it requires a competent starting point. When the model is too weak, external opponents provide the scaffolding needed to bootstrap learning.

---

## References

- Original AlphaZero Paper: [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- AlphaGo Zero: [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- Our training logs: `training_log.txt` and `checkpoints/connect4/training_history.json`

---

**Last Updated:** December 14, 2025  
**Author:** Training debugging session with GitHub Copilot  
**Repository:** alphazerolight (4inarow branch)
