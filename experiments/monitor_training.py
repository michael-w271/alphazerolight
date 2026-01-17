#!/usr/bin/env python3
"""
Rich-based Training Monitor for AlphaZero.
Combines high-level metrics with granular log streaming.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.ansi import AnsiDecoder
from rich.console import Group

class RichMonitor:
    def __init__(self, checkpoint_dir: str, log_file: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_file = log_file
        self.history_file = self.checkpoint_dir / 'training_history.json'
        
        # State
        self.metrics = {}
        self.log_lines = deque(maxlen=20)
        self.start_time = datetime.now()
        
        # Log reading
        self.log_f = None
        self.decoder = AnsiDecoder()
        
    def open_log(self):
        """Open log file and seek to end initially."""
        if os.path.exists(self.log_file):
            self.log_f = open(self.log_file, 'r')
            # Seek to end minus some bytes to show context immediately
            try:
                self.log_f.seek(0, 2)
                size = self.log_f.tell()
                self.log_f.seek(max(0, size - 2000), 0)
            except:
                pass

    def update_metrics(self):
        """Load latest metrics from JSON."""
        if not self.history_file.exists():
            return
            
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                
            entry = None
            # Handle dict-of-lists
            if isinstance(data, dict) and 'iterations' in data:
                if data['iterations']:
                    idx = -1
                    entry = {k: v[idx] for k, v in data.items() if isinstance(v, list) and v}
            # Handle list-of-dicts
            elif isinstance(data, list) and data:
                entry = data[-1]
            
            if entry:
                self.metrics = entry
                
        except Exception:
            pass # Don't crash on read errors

    def update_log(self):
        """Read new lines from log file."""
        if not self.log_f:
            self.open_log()
            return

        lines = self.log_f.readlines()
        for line in lines:
            # Handle carriage returns for progress bars (tqdm)
            # If line has \r, it's an update to the current line.
            # But rich.text.Text.from_ansi handles this reasonably well usually,
            # or we might want to just keep the raw line and let AnsiDecoder handle it.
            # The issue is \r usually overwrites.
            # Simple approach: Clean up empty lines and just strip \n
            clean_line = line.replace('\n', '')
            if clean_line:
                 # decode logic will happen in render to keep styles
                 self.log_lines.append(clean_line)

    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="upper", size=12),
            Layout(name="lower")
        )
        
        # Upper: Metrics
        iter_num = self.metrics.get('iteration', 0)
        total_iters = 350
        
        # Get latest model file
        latest_model = 0
        try:
             latest_model = max([int(p.stem.split('_')[1]) for p in self.checkpoint_dir.glob('model_*.pt') if 'model_' in p.stem], default=0)
        except:
             pass

        status_text = Text()
        status_text.append(f"AlphaZero Training Monitor\n", style="bold magenta")
        status_text.append(f"Iteration: {iter_num}/{total_iters}  ({(iter_num/total_iters)*100:.1f}%)\n\n", style="cyan")
        
        status_text.append(f"Latest Metrics:\n", style="bold underline")
        status_text.append(f"  Total Loss : {self.metrics.get('total_loss', 0):.4f}\n")
        status_text.append(f"  Policy Loss: {self.metrics.get('policy_loss', 0):.4f}\n")
        status_text.append(f"  Value Loss : {self.metrics.get('value_loss', 0):.4f}\n\n")
        
        status_text.append(f"Latest Checkpoint: model_{latest_model}.pt\n", style="green")
        
        elapsed = datetime.now() - self.start_time
        status_text.append(f"Monitor Running: {str(elapsed).split('.')[0]}", style="dim")

        layout["upper"].update(
            Panel(status_text, title="Training Status", border_style="blue")
        )
        
        # Lower: Log Stream
        log_renderables = []
        for line in self.log_lines:
            # TQDM uses \r to return to start of line. 
            # We want to display the FINAL state of that line.
            # Rich's AnsiDecoder is good at this.
            log_renderables.append(self.decoder.decode_line(line))
            
        layout["lower"].update(
            Panel(Group(*log_renderables), title="Live Log Stream", border_style="yellow")
        )
        
        return layout

    def run(self):
        print("Waiting for log file...")
        while not os.path.exists(self.log_file):
            time.sleep(1)
            
        self.open_log()
        
        with Live(self.generate_layout(), refresh_per_second=4, screen=True) as live:
            while True:
                self.update_log()
                
                # Update metrics less frequently
                if int(time.time() * 10) % 20 == 0: # every 2s
                    self.update_metrics()
                    
                live.update(self.generate_layout())
                time.sleep(0.1)

if __name__ == "__main__":
    monitor = RichMonitor(
        checkpoint_dir='checkpoints/connect4',
        log_file='training_log_v2.txt'
    )
    monitor.run()
