# Connect Four Migration - Remaining Tasks

## Completed ✅
1. **Game Implementation** - Created `src/alpha_zero_light/game/connect_four.py`
2. **Configuration** - Created `src/alpha_zero_light/config_connect4.py`
3. **Game Module Exports** - Updated `src/alpha_zero_light/game/__init__.py`
4. **Training Scripts** - Created `scripts/train_connect4.py` and updated `scripts/run_train.py`
5. **Shell Scripts** - Updated all training control scripts (run, resume, pause, check, monitor)
6. **Utility Scripts** - Updated manage_training.py, update_dashboard.py, eval_quick.py, visualize_training.py, generate_evolution_replay.py
7. **Test Script** - Created `scripts/test_connect4.py` with comprehensive win detection tests
8. **Trainer/Evaluator** - Updated to support both numpy (ConnectFour) and tensor (GomokuGPU) games

## Remaining Tasks 🔄

### 9. UI Update (src/alpha_zero_light/ui/app.py)
**Status**: Backed up to `app_gomoku_backup.py`
**Actions Needed**:
- Remove Gomoku imports, import ConnectFour instead
- Update load_model() to handle ConnectFour (6x7, action_size=7)
- Change game selection to "TicTacToe" and "Connect Four" only
- Update board rendering to show 6 rows x 7 columns with column buttons
- Replace cell-click with column-click (pieces drop with gravity)
- Update AI move display to show 7 probability values (column choices)
- Update CSS for circular discs (red/yellow) in a 6x7 grid
- Remove Gomoku-specific styling

**Key Changes**:
```python
# Import
from alpha_zero_light.game.connect_four import ConnectFour

# load_model function
elif game_name == "Connect Four":
    game = ConnectFour()
    checkpoint_dir = Path(...) / "checkpoints" / "connect4"
    # Load model with MODEL_CONFIG from config_connect4

# Board rendering - use 7 column buttons
for col in range(7):
    if st.button(f"↓", key=f"col_{col}"):
        make_move(col)

# Display board as 6x7 grid with colored discs
```

### 10. Documentation Updates
**Files to Update**:

#### README.md
- Change "Gomoku" to "Connect Four" in features
- Update quick start: `python scripts/train_connect4.py`
- Update board size: "6x7 board, 4-in-a-row to win"
- Update UI command: still `streamlit run src/alpha_zero_light/ui/app.py`

#### TRAINING_CONTROLS.md
- Replace all "Gomoku" with "Connect Four"
- Update expected training speed (Connect Four is faster than Gomoku)
- Update log file examples to reference `training_log.txt`
- Update checkpoint paths to `checkpoints/connect4/`

#### docs/GOMOKU_IMPLEMENTATION.md
- Rename to `docs/CONNECT4_IMPLEMENTATION.md`
- Rewrite content:
  - Board: 6 rows × 7 columns
  - Win condition: 4 in a row (horizontal, vertical, diagonal)
  - Action space: 7 columns (0-6)
  - Gravity mechanic: pieces drop to lowest empty row
  - File structure: connect_four.py, config_connect4.py, train_connect4.py

#### docs/MODEL_STORAGE.md
- Replace example paths `checkpoints/gomoku_*` with `checkpoints/connect4`
- Update table entries showing Connect Four checkpoints
- Update example commands to use train_connect4.py

#### project.json
- Update `current_focus` to "Connect Four (v0.2)"
- Remove Gomoku milestone wording
- Update `ml_design` examples to reference 6x7 board

### 11. Website Updates

#### website/index.html
- Change hero title from "Gomoku" to "Connect Four"
- Update description: "Watch AI master the classic game of Connect Four (4-in-a-row) through self-play"
- Update any game rule references

#### website/dashboard.html
- Update metric descriptions to reference Connect Four
- Ensure it reads from `website/status.json` (already updated by update_dashboard.py)

#### website/guide.html
- Update rules section: 6x7 board, 4-in-a-row, column-drop mechanics
- Update encoding examples to show Connect Four states
- Update training guide references

#### website/status.json
- Reset to default state for fresh Connect Four training:
```json
{
  "current_iteration": 0,
  "total_iterations": 50,
  "current_loss": 0,
  "current_win_rate": 0,
  "eta": "Not started",
  "history": null
}
```

### 12. Cleanup Gomoku Files

**Files to Remove**:
```bash
rm src/alpha_zero_light/game/gomoku.py
rm src/alpha_zero_light/game/gomoku_9x9.py  
rm src/alpha_zero_light/game/gomoku_gpu.py
rm src/alpha_zero_light/config_gomoku*.py
```

**Verify No Remaining Imports**:
```bash
grep -r "gomoku" src/ --include="*.py" | grep -i import
grep -r "Gomoku" src/ --include="*.py" | grep -i "from\|import"
```

## Testing & Validation

### Unit Tests
```bash
# Test Connect Four game logic
python scripts/test_connect4.py
```

### Smoke Test Training
```bash
# Quick training test (1-2 iterations)
python scripts/train_connect4.py
# Should create checkpoints/connect4/model_0.pt, etc.
```

### UI Test
```bash
# Launch UI
streamlit run src/alpha_zero_light/ui/app.py
# Play a full Connect Four game
# Verify column buttons work
# Verify discs drop correctly
# Verify AI makes legal moves
```

### Evaluation Test
```bash
# After training checkpoint exists
python scripts/eval_quick.py
# Should load from checkpoints/connect4
# Should show win rate against random player
```

## Quick Command Reference

```bash
# Create checkpoint directories
mkdir -p checkpoints/connect4
mkdir -p logs/connect4  
mkdir -p docs/training_plots/connect4

# Start training
./resume_training.sh

# Monitor training
./monitor_training.sh

# Check status
./check_training.sh

# Pause training
./pause_training.sh

# Play against AI
streamlit run src/alpha_zero_light/ui/app.py

# Visualize metrics
python scripts/visualize_training.py

# Quick evaluation
python scripts/eval_quick.py
```

## Notes
- The trainer now automatically detects numpy vs tensor games via `hasattr(game, 'device')`
- Connect Four uses sequential self-play (simpler, CPU-friendly)
- GomokuGPU code remains for reference but is no longer used
- TicTacToe support is maintained alongside Connect Four
