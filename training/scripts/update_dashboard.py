
import json
import time
import sys
import os
from pathlib import Path

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from alpha_zero_light.config_connect4 import PATHS, TRAINING_CONFIG

def get_latest_history():
    # Check Connect Four checkpoint location
    history_path = Path(PATHS.checkpoints) / 'training_history.json'
    
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                return json.load(f), history_path.parent
        except:
            pass
    return None, None

def update_status():
    history, checkpoint_dir = get_latest_history()
    
    status = {
        'current_iteration': 0,
        'total_iterations': TRAINING_CONFIG['num_iterations'],
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
