#!/bin/bash
# Backup old checkpoints and start fresh training with fixed MCTS

echo "========================================================================"
echo "🔧 RESTARTING TRAINING WITH FIXED MCTS"
echo "========================================================================"
echo ""
echo "CRITICAL BUG FIXED:"
echo "  - Terminal state detection in MCTS (was checking after perspective flip)"
echo "  - This caused wins to be marked as losses and vice versa"
echo "  - All previous training (iterations 0-25) learned from corrupted data"
echo ""
echo "This script will:"
echo "  1. Backup current checkpoints to checkpoints_terminal_bug_backup/"
echo "  2. Clean current checkpoints/"
echo "  3. Clear training logs"
echo "  4. Start fresh training from iteration 0"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

cd "$(dirname "$0")"

# Backup old checkpoints
BACKUP_DIR="checkpoints_terminal_bug_backup_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "📦 Backing up old checkpoints..."
if [ -d "checkpoints/connect4" ]; then
    mkdir -p "$BACKUP_DIR"
    cp -r checkpoints/connect4 "$BACKUP_DIR/"
    echo "✅ Backed up to $BACKUP_DIR/"
else
    echo "⚠️  No checkpoints to backup"
fi

# Clean checkpoints
echo ""
echo "🧹 Cleaning current checkpoints..."
rm -rf checkpoints/connect4/*.pt
echo "✅ Checkpoints cleared"

# Backup old logs
echo ""
echo "📝 Backing up training log..."
if [ -f "training_log.txt" ]; then
    mv training_log.txt "training_log_terminal_bug_$(date +%Y%m%d_%H%M%S).txt"
    echo "✅ Log backed up"
fi

# Clear test results
echo ""
echo "🧪 Clearing old test results..."
if [ -f "model_test_results.json" ]; then
    mv model_test_results.json "model_test_results_old_$(date +%Y%m%d_%H%M%S).json"
    echo "✅ Test results backed up"
fi

echo ""
echo "========================================================================"
echo "✅ READY TO START FRESH TRAINING"
echo "========================================================================"
echo ""
echo "The MCTS bug has been fixed. New training will:"
echo "  ✓ Learn from correct win/loss signals"
echo "  ✓ Detect immediate wins properly"
echo "  ✓ Learn to block opponent threats"
echo "  ✓ Converge faster with correct gradients"
echo ""
echo "To start training, run:"
echo "  ./start_training.sh"
echo ""
echo "To monitor training with live testing:"
echo "  ./start_training_monitored.sh"
echo ""
