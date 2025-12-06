#!/bin/bash
# Run the 30-minute training session

# Activate environment (if needed, though we assume we are in it)
# source activate azl

# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the training script with the new config
# We use the absolute path to the python executable in the azl environment
/home/michael/miniforge3/envs/azl/bin/python scripts/run_training_30min.py
