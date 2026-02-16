# Connect4 AlphaZero Training V2 - Quick Reference

## 🚀 Quick Start (Recommended)

```bash
cd /mnt/ssd2pro/alpha-zero-light
./start_training_v2.sh
```

This script will:
- ✅ Check CUDA availability
- ✅ Create checkpoint directory
- ✅ Offer to start in tmux (recommended)
- ✅ Start training

## 📋 Training Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Iterations** | 150 | Reduced from 350 for faster evaluation |
| **Games/Iteration** | 150 | Self-play games per iteration |
| **Training Epochs** | 90 | Neural network training epochs |
| **Batch Size** | 512 | Training batch size |
| **MCTS Searches** | 100 | Searches per move |
| **Learning Rate** | 0.001 | Adam optimizer |
| **Model** | ResNet-10 | 128 hidden units, ~1.2M parameters |

## ⏱️ Estimated Timeline

- **Per Iteration**: ~12-15 minutes
- **Total Time**: ~30-40 hours
- **Checkpoint**: Every iteration (150 models saved)
- **Disk Space**: ~3GB total

## 📁 File Locations

- **Checkpoints**: `/mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2/model_*.pt`
- **Configuration**: `training_config_v2.json`
- **Logs**: `/mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2/training_history.json`

## 🎯 Manual Commands

### Option 1: Direct Run (blocks terminal)
```bash
cd /mnt/ssd2pro/alpha-zero-light
/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python scripts/train_connect4.py
```

### Option 2: tmux Session (recommended)
```bash
# Start new session
tmux new -s connect4_v2
cd /mnt/ssd2pro/alpha-zero-light
/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python scripts/train_connect4.py

# Detach: Ctrl+B then D
# Reattach: tmux attach -t connect4_v2
# Kill: tmux kill-session -t connect4_v2
```

### Option 3: Background with nohup
```bash
cd /mnt/ssd2pro/alpha-zero-light
nohup /mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python scripts/train_connect4.py > training_v2.log 2>&1 &

# Monitor: tail -f training_v2.log
# Find PID: ps aux | grep train_connect4
# Kill: kill <PID>
```

## 📊 Monitoring Progress

### View Training Log
```bash
tail -f /mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2/training_history.json
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Check Model Files
```bash
ls -lh /mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2/
```

## 🏆 Post-Training Evaluation

### Run Tournament vs model_120 (current champion)
```bash
cd /mnt/ssd2pro/alpha-zero-light
/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python model_tournament.py
```

Edit `model_tournament.py` to include:
- `model_50.pt` from v2
- `model_100.pt` from v2
- `model_150.pt` from v2
- `model_120.pt` (original champion for comparison)

### Play Against New Model
```bash
# Terminal play
/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python play_connect4.py

# Web UI (update model to new checkpoint)
cd connect4-lab/api
# Edit server.py line 40 to point to new model
/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python server.py
```

## 🎲 Expected Milestones

| Iteration | Expected Capability |
|-----------|---------------------|
| 20 | Basic tactics (block threats, connect pieces) |
| 50 | Opening theory development |
| 80 | Middle game patterns |
| 120 | Mature tactical play (champion level) |
| 150 | Peak performance |

## 🔧 Troubleshooting

**CUDA Out of Memory**:
- Reduce `batch_size` in `config_connect4.py` (512 → 256)

**Training Stuck**:
- Check GPU usage: `nvidia-smi`
- Check process: `ps aux | grep train_connect4`

**Slow Training**:
- Each iteration should take 12-15 minutes
- If slower, check GPU utilization
- Ensure no other processes using GPU

**Resume from Checkpoint**:
The trainer automatically resumes from the last saved checkpoint if it exists.

## 📈 Success Criteria

After 150 iterations, the model should:
- ✅ Block immediate threats  
- ✅ Find winning moves  
- ✅ Avoid obvious traps  
- ✅ Beat random opponent >95%  
- ✅ Show strategic opening play  
- ✅ Competitive with model_120 champion

## Full JSON Configuration

See: `training_config_v2.json` for complete hyperparameter details.
