#!/bin/bash
# Quick runner script for WDL training and tournament

set -e

echo "=========================================="
echo "Phase 5-7 Execution Script"
echo "=========================================="
echo ""

# Step 1: Train WDL model
echo "Step 1: Training WDL model (100 epochs)..."
python3 training/scripts/train_supervised.py \
    --epochs 100 \
    --batch-size 256 \
    --lr 5e-5 \
    --weight-decay 5e-4 \
    --gradient-clip 1.0 \
    --early-stopping 10 \
    2>&1 | tee training_wdl_full.log

echo ""
echo "✓ Training complete!"
echo ""

# Step 2: Test WDL evaluator
echo "Step 2: Testing WDL evaluator..."
python3 src/alpha_zero_light/engine/wdl_evaluator.py

echo ""
echo "✓ Evaluator test complete!"
echo ""

# Step 3: Run tournament
echo "Step 3: Running tournament (100 games)..."
python3 tests/integration/alpha_beta_vs_mcts.py \
    --games 100 \
    --time 1000 \
    2>&1 | tee tournament_results.log

echo ""
echo "=========================================="
echo "All steps complete!"
echo "=========================================="
echo ""
echo "Check logs:"
echo "  - training_wdl_full.log"
echo "  - tournament_results.log"
