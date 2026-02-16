# Gomoku Training Control Guide

## Quick Commands

### Check Training Status
```bash
./check_training.sh
```

### Pause Training
```bash
./pause_training.sh
```

### Resume/Start Training
```bash
./resume_training.sh
```

### Monitor Training Live
```bash
tail -f artifacts/logs/training/training_log_v2.txt
```

## GPU Usage While Training

**Yes, you can play games in parallel!** Your RTX 5080 has 16GB VRAM:
- Training uses ~400MB
- Plenty of room for Overcooked or other games
- Expect 10-20% slower training if GPU is heavily used

## Training Details

- **Speed**: ~60 seconds per self-play game
- **Iteration time**: ~100 minutes per iteration (100 games)
- **Progress**: Check `artifacts/logs/training/training_log_v2.txt` for live updates with emojis 🎮🧠📊

## Files Created

- `pause_training.sh` - Stop training gracefully
- `resume_training.sh` - Start/resume training in background
- `check_training.sh` - View current progress
- `artifacts/logs/training/training_log_v2.txt` - Live training output

## Note

Training will resume from the last saved checkpoint automatically when you run `resume_training.sh`.
