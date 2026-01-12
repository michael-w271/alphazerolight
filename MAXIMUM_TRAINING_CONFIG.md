# Maximum Strength Connect4 Training Configuration

## 🏆 Goal: Create Unbeatable AI

### Model Architecture (4x Larger!)
- **Network**: ResNet-20 (vs previous ResNet-10)
- **Hidden Units**: 256 (vs previous 128)
- **Parameters**: ~5 million (vs ~1.2 million)
- **File Size**: ~40MB per checkpoint (vs ~12MB)

### MCTS Configuration (4x Deeper!)
- **Searches per move**: 400 (vs previous 100)
- **Search depth**: ~8-12 ply (vs ~5-7)
- **Inference time**: ~400-800ms per move (vs ~100-200ms)
- **Tactical horizon**: Can see 6-8 moves ahead

### Training Strategy
**Opponent Mix** (More Diversity):
- 90% Self-play (vs 98.5%)
- 5% Aggressive opponent (vs 1%)
- 3% Strong 2-ply opponent (vs 0.5%)
- 2% Tactical puzzles (new!)

**Learning Rate Schedule**:
- Iterations 0-100: LR = 0.001
- Iterations 100-200: LR = 0.0005
- Iterations 200-300: LR = 0.0001

### Time Estimates
- **Per iteration**: ~25-40 minutes
  - Self-play (150 games): 10-15 min
  - MCTS (400 searches): 4x slower
  - Training (90 epochs): 15-25 min (larger model)

- **Total for 150 iterations**: ~3-4 days
- **Total for 300 iterations**: ~7-10 days

### Checkpoints
- **Location**: `/mnt/ssd2pro/alpha-zero-light/checkpoints/connect4/`
- **Frequency**: Every iteration
- **Size**: ~40MB × 150 = ~6GB total

## 🎯 Expected Performance Milestones

| Iteration | Expected Capability | Can Human Beat? |
|-----------|---------------------|-----------------|
| 50 | Strong tactics, basic strategy | Maybe (very hard) |
| 100 | Deep combinations, opening theory | Unlikely |
| 150 | Tournament-level play | Extremely unlikely |
| 200+ | Near-perfect play | Essentially unbeatable |

## 🔧 Using Models in React App

Any model can be loaded in the React app:

1. Edit `connect4-lab/api/server.py` line 40:
```python
checkpoint_path = Path("/mnt/ssd2pro/alpha-zero-checkpoints/connect4_max/model_100.pt")
```

2. Restart API:
```bash
cd connect4-lab/api
python server.py
```

3. Open game:
```bash
xdg-open /path/to/connect4-lab/index.html
```

## 📊 Comparison to Previous Training

| Metric | Previous (model_120) | Maximum (this run) |
|--------|---------------------|-------------------|
| Model size | ResNet-10, 128 hidden | ResNet-20, 256 hidden |
| Parameters | ~1.2M | ~5M |
| MCTS searches | 100 | 400 |
| Self-play % | 98.5% | 90% |
| Opponent diversity | 1.5% | 10% |
| Training time | ~24h (120 iter) | ~150-200h (150-300 iter) |
| Expected strength | Good (beatable) | Near-unbeatable |

## ⚠️ Important Notes

1. **Much slower training**: ~4x longer per iteration
2. **Higher GPU memory**: ~8-10GB VRAM (vs ~2-3GB)
3. **Larger files**: ~40MB per model (vs ~12MB)
4. **Worth it**: Should create genuinely strong AI

## 🚀 Launch Command

```bash
./launch_training_with_monitors.sh
```

Monitors will show:
- Window 1: Training progress (updates every 5s)
- Window 2: Evaluation results (every 10 iterations)

---

**Started**: 2026-01-12  
**Configuration**: Maximum Strength  
**Target**: Unbeatable AI
