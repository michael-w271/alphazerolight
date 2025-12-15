# CORE FILES TO INVESTIGATE - AlphaZero Connect Four

## The two most critical files:

### 1. src/alpha_zero_light/training/trainer.py
**What it does:** Generates training data from self-play games
**Critical sections:**
- Lines 60-160: self_play_game() - Collects game positions and outcomes
- Lines 85-109: Self-play loop (model vs itself)
- Lines 110-157: vs opponent loop (model vs random/heuristic)
- Lines 98-107 & 146-155: Outcome assignment (THIS IS WHERE BUGS HIDE)

**Key logic:**
```python
# After game ends:
for hist_state, hist_probs, hist_player in game_memory:
    hist_outcome = value if hist_player == player else game.get_opponent_value(value)
    state_from_player_perspective = game.change_perspective(hist_state, hist_player)
    worker_memory.append((
        game.get_encoded_state(state_from_player_perspective),
        hist_probs,
        hist_outcome
    ))
```

### 2. src/alpha_zero_light/mcts/mcts.py
**What it does:** Searches game tree to find good moves
**Critical sections:**
- Lines 10-85: Node class (stores game state and statistics)
- Lines 48-77: Node.expand() - Creates child nodes
- Lines 86-190: MCTS.search() - Main search algorithm
- Lines 192-346: MCTS.search_batch() - Batch version

**Key logic:**
```python
# In Node.expand() - lines 64-77:
terminal_value, is_terminal = self.game.get_value_and_terminated(child_state, action)
child_state = self.game.change_perspective(child_state, player=-1)
if is_terminal and terminal_value != 0:
    terminal_value = -terminal_value
```

## Supporting files to check:

### 3. src/alpha_zero_light/game/connect_four.py
- Lines 100-165: check_win() - Win detection logic
- Lines 171-185: get_value_and_terminated() - Returns game outcome
- Lines 198-226: get_encoded_state() - Board encoding

### 4. src/alpha_zero_light/config_connect4.py
- Training hyperparameters (epochs, searches, temperature, etc.)

## Where bugs typically hide:

1. **Sign flips:** When converting between player perspectives
2. **State encoding:** Making sure current player = +1 channel
3. **Terminal value propagation:** Who won vs who's evaluating
4. **MCTS backpropagation:** Value sign flips going up the tree

## Quick diagnostic questions:

1. Is `game_memory` storing states from Player 1's perspective or current player's?
2. When we call `change_perspective(hist_state, hist_player)`, does it convert TO hist_player's view?
3. Does MCTS correctly flip values when backpropagating?
4. Are policy probabilities aligned with the correct state representation?

## My recommendation:

Focus on **trainer.py lines 85-157** and **mcts.py lines 48-85**.
These are where state perspective and outcome assignment happen.

Print DEBUG statements at every perspective change to trace:
- What player's perspective is the state in?
- What player made this move?
- What outcome should they get?
- What encoding channels represent what?
