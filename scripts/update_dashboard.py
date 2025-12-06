
import json
import time
import sys
import os
from pathlib import Path

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.config_gomoku_30min import PATHS as PATHS_30MIN
from alpha_zero_light.config_gomoku_9x9_overnight import PATHS as PATHS_OVERNIGHT
from alpha_zero_light.config_gomoku_9x9 import PATHS as PATHS_9X9

def get_latest_history():
    # Check all locations
    paths = [
        Path(PATHS_9X9['checkpoints']) / 'training_history.json',
        Path(PATHS_30MIN['checkpoints']) / 'training_history.json',
        Path(PATHS_OVERNIGHT['checkpoints']) / 'training_history.json'
    ]
    
    for p in paths:
        if p.exists():
            try:
                with open(p, 'r') as f:
                    return json.load(f), p.parent
            except:
                continue
    return None, None

def update_status():
    history, checkpoint_dir = get_latest_history()
    
    status = {
        'current_iteration': 0,
        'total_iterations': 0, # Unknown until we know which config
        'current_loss': 0,
        'current_win_rate': 0,
        'eta': 'Calculating...',
        'history': None
    }
    
    if history:
        status['history'] = history
        if history['iterations']:
            status['current_iteration'] = history['iterations'][-1] + 1
            status['current_loss'] = history['total_loss'][-1]
        
        if history['eval_win_rate']:
            status['current_win_rate'] = history['eval_win_rate'][-1]
            
        # Try to guess total iterations based on path
        if '30min' in str(checkpoint_dir):
            status['total_iterations'] = 6
        elif 'overnight' in str(checkpoint_dir):
            status['total_iterations'] = 30
        else:
            status['total_iterations'] = 200 # Standard 9x9 config
            
    # Write to website directory
    website_dir = Path(os.path.join(os.path.dirname(__file__), '../website'))
    website_dir.mkdir(exist_ok=True)
    
    with open(website_dir / 'status.json', 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"Updated status.json at {time.ctime()}")

if __name__ == "__main__":
    while True:
        update_status()
        time.sleep(5)
