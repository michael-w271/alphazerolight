#!/bin/bash
# Live training monitor with auto-refresh

echo "🎮 Connect Four Training Monitor"
echo "================================"
echo ""

# Check if training is running
PID=$(pgrep -f "train_connect4.py")

if [ -z "$PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

echo "✅ Training is RUNNING (PID: $PID)"
echo ""

# Show GPU status
echo "📊 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "   GPU Usage: %s%%  |  VRAM: %s/%s MB  |  Temp: %s°C\n", $1, $2, $3, $4}'
echo ""

# Show latest training progress
echo "📈 Latest Progress:"
tail -n 25 training_log.txt | grep -E "(Iteration|Generated|Loss|Win Rate|complete)" | tail -n 10

echo ""
echo "💡 Run './check_training.sh' for updates"
echo "💡 Or: tail -f training_log.txt"
