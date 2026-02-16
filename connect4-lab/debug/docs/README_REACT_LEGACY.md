# Connect4 Lab - React App

## Setup Instructions

Since npm/node is not available on your system, here's how to use this app:

### Option 1: Install Node.js and run locally
1. Install Node.js from https://nodejs.org/
2. Run setup:
```bash
cd /mnt/ssd2pro/alpha-zero-light/connect4-lab/frontend
npm install
npm run dev
```

### Option 2: Use the provided standalone HTML
The app has been created with all components in a single HTML file that can run directly in a browser.

## Running the Application

### Start the Backend API:
```bash
cd /mnt/ssd2pro/alpha-zero-light/connect4-lab/api
/mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python server.py
```

### Open the Frontend:
- If using Vite dev server: http://localhost:5173
- If using standalone HTML: Open `connect4-lab/frontend/index.html` in browser

## Project Structure

```
connect4-lab/
├── api/
│   ├── server.py          # Flask API serving model_120.pt
│   └── requirements.txt   # Python dependencies
└── frontend/
    ├── src/
    │   ├── App.tsx        # Main application component
    │   ├── components/    # React components
    │   ├── lib/           # Game logic and API client
    │   └── styles/        # CSS styling
    ├── index.html         # Entry point
    └── package.json       # Node dependencies
```

## Features

- **Play Connect Four vs AI** (model_120.pt)
- **Real-time move analysis** with Q-values heatmap
- **Move ranking** showing top 3 moves with confidence
- **Move history log** with replayability
- **Dark sci-fi dashboard** matching Tetris Agent Lab aesthetic
- **Responsive design** (desktop, tablet, mobile)

## API Endpoints

- `GET /api/health` - Check API status
- `POST /api/predict` - Get AI prediction for board state

## Controls

- **Click column** to drop piece
- **New Game** - Reset board
- **Undo** - Take back last move
- **AI Toggle** - Enable/disable AI auto-play
