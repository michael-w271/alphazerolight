# New Training Configuration - Improved Connect Four

## Changes Made

### 1. Extended Warmup Phase (30 → 100 iterations)
**Problem**: Previous warmup was too short, model never learned tactics properly  
**Solution**: 
- **Iterations 0-39 (40 iters)**: Pure random opponent - build fundamentals
- **Iterations 40-79 (40 iters)**: Heuristic opponent - learn tactics (win/block)
- **Iterations 80-99 (20 iters)**: Mixed opponents - generalize skills
- **Iterations 100-149 (50 iters)**: Self-play phase

### 2. Larger Model Capacity (10→15 blocks, 128→256 hidden)
**Problem**: Model too small to learn Connect Four patterns  
**Solution**: 
- ResNet blocks: 10 → **15** (+50%)
- Hidden units: 128 → **256** (+100%)
- Memory usage: ~72MB → ~350MB (plenty of headroom on RTX 5080)
- Training time: ~2-3x slower per iteration, but better quality

### 3. More MCTS Searches (50 → 100)
**Problem**: Shallow search couldn't find tactical moves  
**Solution**: 
- MCTS searches: 50 → **100** (deeper lookahead)
- Better training data quality (model learns from better games)
- ~2x slower self-play but worth it for tactics

### 4. Reduced Total Iterations (190 → 150)
**Reason**: Faster experimentation - focus warmup quality over total training time
- Previous: 30 warmup + 160 self-play
- New: 100 warmup + 50 self-play
- Can extend later if warmup succeeds

### 5. Fewer Games Per Iteration (600 → 400)
**Reason**: Faster iterations = quicker feedback on whether training is working
- Still enough samples per iteration (~4000-5000 training positions)

## Expected Outcomes

### GPU Impact
- **VRAM usage**: ~350MB / 16GB = 2.2% utilization ✓
- **Training speed**: ~2-3 minutes per iteration (vs ~1 min before)
- **Total training time**: ~6-7 hours for 150 iterations

### Performance Goals
By end of warmup (iteration 99):
- ✓ Empty board value prediction: |value| < 0.2
- ✓ Find immediate wins (4-in-a-row completions)
- ✓ Block opponent threats (4-in-a-row blocks)
- ✓ Prefer center columns in opening

By end of training (iteration 149):
- ✓ Beat heuristic opponent consistently
- ✓ Strategic positional play
- ✓ Value loss < 0.1 (stable)

## How to Run

```bash
# Clean old checkpoints
./clean_checkpoints.sh

# Start new training
./start_training.sh
```

## Monitoring Progress

Key metrics to watch:
1. **Value loss should NEVER degrade** (if it does, training is broken)
2. **Iteration 40**: First heuristic games - expect temporary loss increase (normal)
3. **Iteration 100**: Transition to self-play - watch for value loss stability
4. **Every 10 iterations**: Check training_history.json for trends

## Rollback Plan

If training fails:
- Checkpoints saved every iteration in `checkpoints/connect4/`
- Can load any specific iteration for testing
- Training history in `checkpoints/connect4/training_history.json`
