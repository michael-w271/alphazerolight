# Connect4 AI Lab - Quick Start Guide

## 🚀 Launch in 2 Steps

### Step 1: Start the API Server
```bash
cd /mnt/ssd2pro/alpha-zero-light/connect4-lab/api
/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python server.py
```

You should see:
```
✅ Model loaded successfully on cuda
🚀 Connect4 API Server Starting
API Endpoints:
  GET  /api/health  - Health check
  POST /api/predict - Get AI prediction
```

### Step 2: Open the Game
Open this file in your browser:
```
/mnt/ssd2pro/alpha-zero-light/connect4-lab/index.html
```

Or use command:
```bash
xdg-open /mnt/ssd2pro/alpha-zero-light/connect4-lab/index.html
```

## 🎮 How to Play

1. **Click any column** to drop your red piece
2. **AI responds** automatically with yellow piece
3. **Watch the analysis** - Q-values and top moves update in real-time
4. **Win** by connecting 4 pieces horizontally, vertically, or diagonally

## 🎨 Features

- **Dark Sci-Fi Dashboard** matching Tetris Agent Lab aesthetic
- **Live AI Analysis** - Q-values heatmap for all 7 columns
- **Move Rankings** - Top 5 moves ranked by AI confidence
- **Move History** - Complete game log
- **Smooth Animations** - Falling pieces and winning highlights
- **Responsive Design** - Works on desktop, tablet, and mobile

## 🧠 AI Model

- **Champion Model**: model_120.pt (60% tournament win rate)
- **MCTS Searches**: 100 per move
- **Inference Speed**: ~100-200ms per move on RTX 5080

## ⚙️ Controls

- **New Game** - Reset the board
- **Undo** - Take back last move (or last 2 if AI moved)
- **AI Toggle** - Turn AI opponent on/off

## 🎯 Understanding the Analysis

**Q-Values Heatmap**:
- **Green** = High value (AI prefers this move)
- **Red** = Low value (AI avoids this move)
- **Gold border** = Best move according to AI
- **Purple border** = AI's chosen move
- **Grayed out** = Column is full

**Move Rankings**:
- Shows top 5 columns ranked by Q-value
- Higher values = stronger moves
- Gold highlight = #1 ranked move

## 🔧 Troubleshooting

**"API Offline" in top right?**
- Make sure the Flask server is running (Step 1)
- Check firewall isn't blocking localhost:5000

**AI not moving?**
- Check browser console (F12) for errors
- Verify API is responding: http://localhost:5000/api/health

**Slow AI moves?**
- Normal! MCTS with 100 searches takes ~100-200ms
- Consider reducing num_searches in server.py for faster play

## 📁 Project Structure

```
connect4-lab/
├── index.html         # Complete app (open this!)
├── api/
│   ├── server.py      # Flask API
│   └── requirements.txt
└── README.md          # This file
```

## 🏆 Tournament Champion

This AI uses the tournament champion **model_120.pt** which achieved:
- 30 wins, 5 draws, 15 losses (60% win rate)
- Trained for 120 iterations with critical bug fixes
- See TRAINING_DOCUMENTATION.md for full technical details

Enjoy playing against the AI! 🎮
