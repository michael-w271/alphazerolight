#!/bin/bash
source ./env_config.sh

echo "Starting Game UI..."
$PYTHON_EXEC -m streamlit run src/alpha_zero_light/ui/app.py
