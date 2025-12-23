"""
Telemetry publisher for streaming training data to C++ viewer via ZeroMQ.

This module provides a lightweight, non-blocking publisher that sends
JSON-formatted telemetry messages to connected viewers without impacting
training performance.
"""

import json
import time
import numpy as np
import zmq
from typing import Dict, Any, Optional, List


class TelemetryPublisher:
    """
    ZeroMQ publisher for AlphaZero training telemetry.
    
    Sends three message types:
    - 'frame': Per-move game state, MCTS stats, chosen action
    - 'metrics': Per-iteration training metrics (loss, win rate, etc.)
    - 'net_summary': Periodic network statistics (layer norms, activations)
    
    Uses PUB socket pattern - gracefully handles no subscribers.
    """
    
    def __init__(self, endpoint: str, send_frame_frequency: int = 1,
                 send_metrics_frequency: int = 1,
                 send_net_summary_frequency: int = 5):
        """
        Initialize telemetry publisher.
        
        Args:
            endpoint: ZeroMQ endpoint (e.g., "tcp://127.0.0.1:5556")
            send_frame_frequency: Send every Nth move (1 = all moves)
            send_metrics_frequency: Send every Nth iteration
            send_net_summary_frequency: Send network summary every Nth iteration
        """
        self.endpoint = endpoint
        self.send_frame_frequency = send_frame_frequency
        self.send_metrics_frequency = send_metrics_frequency
        self.send_net_summary_frequency = send_net_summary_frequency
        
        # Message counters for frequency control
        self._frame_counter = 0
        self._metrics_counter = 0
        self._net_summary_counter = 0
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # Set high water mark to avoid blocking (drop old messages if slow consumer)
        self.socket.setsockopt(zmq.SNDHWM, 100)
        
        # Bind to endpoint
        self.socket.bind(endpoint)
        print(f"📡 Telemetry publisher bound to {endpoint}")
        
        # Give subscribers time to connect (ZMQ late-joiner issue)
        # Increased to 2 seconds to ensure viewer has time to subscribe
        time.sleep(2.0)
    
    def _send_json(self, message: Dict[str, Any]):
        """Send JSON message (non-blocking)."""
        try:
            json_str = json.dumps(message, default=self._json_default)
            self.socket.send_string(json_str, flags=zmq.NOBLOCK)
        except zmq.Again:
            # Send buffer full - drop message (non-blocking publisher)
            pass
        except Exception as e:
            # Suppress errors to avoid breaking training
            print(f"⚠️ Telemetry send error: {e}")
    
    @staticmethod
    def _json_default(obj):
        """Handle numpy types in JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def send_frame(self, iteration: int, game_idx: int, move_idx: int,
                   player: int, board: np.ndarray, valid_moves: np.ndarray,
                   policy_head: np.ndarray, value_head: float,
                   mcts_policy: np.ndarray, root_visits: List[int],
                   root_q: List[float], chosen_action: int,
                   temperature: float, is_terminal: bool,
                   terminal_value: Optional[float] = None):
        """
        Send a frame telemetry message (game state + MCTS thinking).
        
        Args:
            iteration: Current training iteration
            game_idx: Game index within iteration
            move_idx: Move index within game
            player: Current player (+1 or -1)
            board: Board state (6x7 for Connect Four)
            valid_moves: Valid action mask (length 7)
            policy_head: Raw network policy output (after softmax)
            value_head: Network value output (-1 to +1)
            mcts_policy: MCTS search result distribution
            root_visits: MCTS visit counts per action
            root_q: MCTS Q-values per action
            chosen_action: Action selected by policy
            temperature: Temperature used for action selection
            is_terminal: Whether game ended after this move
            terminal_value: Final game value if terminal
        """
        self._frame_counter += 1
        if self._frame_counter % self.send_frame_frequency != 0:
            return
        
        message = {
            'type': 'frame',
            'timestamp_ms': int(time.time() * 1000),
            'iteration': iteration,
            'game_idx': game_idx,
            'move_idx': move_idx,
            'player': player,
            'board': board,
            'valid_moves': valid_moves,
            'policy_head': policy_head,
            'value_head': value_head,
            'mcts_policy': mcts_policy,
            'root_visits': root_visits,
            'root_q': root_q,
            'chosen_action': chosen_action,
            'temperature': temperature,
            'is_terminal': is_terminal,
            'terminal_value': terminal_value
        }
        
        self._send_json(message)
    
    def send_metrics(self, iteration: int, total_loss: float,
                     policy_loss: float, value_loss: float,
                     policy_entropy: Optional[float] = None,
                     examples_seen: Optional[int] = None,
                     eval_winrate: Optional[float] = None,
                     avg_game_length: Optional[float] = None):
        """
        Send training metrics telemetry message.
        
        Args:
            iteration: Current training iteration
            total_loss: Combined loss value
            policy_loss: Policy head loss
            value_loss: Value head loss
            policy_entropy: Optional policy entropy metric
            examples_seen: Optional total training examples processed
            eval_winrate: Optional evaluation win rate
            avg_game_length: Optional average game length
        """
        self._metrics_counter += 1
        if self._metrics_counter % self.send_metrics_frequency != 0:
            return
        
        message = {
            'type': 'metrics',
            'timestamp_ms': int(time.time() * 1000),
            'iteration': iteration,
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_entropy': policy_entropy,
            'examples_seen': examples_seen,
            'eval_winrate': eval_winrate,
            'avg_game_length': avg_game_length
        }
        
        self._send_json(message)
    
    def send_net_summary(self, iteration: int, layer_norms: Dict[str, float],
                         activation_norms: Optional[Dict[str, float]] = None,
                         conv1_filters: Optional[np.ndarray] = None):
        """
        Send network summary telemetry message.
        
        Args:
            iteration: Current training iteration
            layer_norms: Dictionary of layer weight norms (e.g., {'conv1.weight': 5.2})
            activation_norms: Optional activation norms per layer
            conv1_filters: Optional subset of conv1 filters for visualization
        """
        self._net_summary_counter += 1
        if self._net_summary_counter % self.send_net_summary_frequency != 0:
            return
        
        message = {
            'type': 'net_summary',
            'timestamp_ms': int(time.time() * 1000),
            'iteration': iteration,
            'layer_norms': layer_norms,
            'activation_norms': activation_norms,
            'conv1_filters': conv1_filters
        }
        
        self._send_json(message)
    
    def close(self):
        """Close ZeroMQ socket and context."""
        self.socket.close()
        self.context.term()
        print("📡 Telemetry publisher closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
