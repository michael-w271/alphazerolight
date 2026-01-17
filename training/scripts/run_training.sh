#!/bin/bash
source ../../env_config.sh

echo "Starting Training..."
$PYTHON_EXEC training/scripts/run_train.py
