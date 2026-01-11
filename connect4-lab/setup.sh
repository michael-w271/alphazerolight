#!/bin/bash
# Setup script for Connect4 Lab React app

echo "🚀 Setting up Connect4 Lab..."

# Create React app with Vite
cd /mnt/ssd2pro/alpha-zero-light/connect4-lab
npm create vite@latest frontend -- --template react-ts

cd frontend

# Install dependencies
npm install

echo "✅ React app created at connect4-lab/frontend"
echo ""
echo "To run the app:"
echo "  cd connect4-lab/frontend && npm run dev"
echo ""
echo "To run the API:"
echo "  cd connect4-lab/api && python server.py"
