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
                   terminal_value: Optional[float] = None,
                   policy_entropy: Optional[float] = None,
                   mcts_improvement: Optional[float] = None):
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
            policy_entropy: Shannon entropy of policy distribution
            mcts_improvement: How much MCTS improved over raw policy
        """
        self._frame_counter += 1
        if self._frame_counter % self.send_frame_frequency != 0:
            return
        
        # Calculate policy entropy if not provided
        if policy_entropy is None and policy_head is not None:
            policy_entropy = self._calculate_entropy(policy_head)
        
        # Default values for missing metrics
        if policy_entropy is None:
            policy_entropy = 0.0
        if mcts_improvement is None:
            mcts_improvement = 0.0
        
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
            'terminal_value': terminal_value if terminal_value is not None else 0.0,
            'policy_entropy': policy_entropy,
            'mcts_improvement': mcts_improvement
        }
        
        self._send_json(message)
    
    @staticmethod
    def _calculate_entropy(distribution: np.ndarray) -> float:
        """Calculate Shannon entropy of a probability distribution."""
        # Filter out zeros to avoid log(0)
        probs = distribution[distribution > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs + 1e-10)))
    
    def send_metrics(self, iteration: int, total_loss: float,
                     policy_loss: float, value_loss: float,
                     policy_entropy: Optional[float] = None,
                     examples_seen: Optional[int] = None,
                     eval_winrate: Optional[float] = None,
                     avg_game_length: Optional[float] = None,
                     gradient_norms: Optional[Dict[str, float]] = None,
                     policy_sharpness: Optional[float] = None,
                     mcts_improvement_avg: Optional[float] = None):
        """
        Send training metrics telemetry message.
        
        Args:
            iteration: Current training iteration
            total_loss: Combined loss value
            policy_loss: Policy head loss
            value_loss: Value head loss
            policy_entropy: Average policy entropy this iteration
            examples_seen: Total training examples processed
            eval_winrate: Evaluation win rate
            avg_game_length: Average game length this iteration
            gradient_norms: Dict of gradient norms by layer group
            policy_sharpness: Average policy sharpness (confidence)
            mcts_improvement_avg: Average MCTS improvement factor
        """
        self._metrics_counter += 1
        if self._metrics_counter % self.send_metrics_frequency != 0:
            return
        
        # Ensure no None values are sent
        message = {
            'type': 'metrics',
            'timestamp_ms': int(time.time() * 1000),
            'iteration': iteration,
            'total_loss': float(total_loss) if total_loss is not None else 0.0,
            'policy_loss': float(policy_loss) if policy_loss is not None else 0.0,
            'value_loss': float(value_loss) if value_loss is not None else 0.0,
            'policy_entropy': float(policy_entropy) if policy_entropy is not None else 0.0,
            'examples_seen': int(examples_seen) if examples_seen is not None else 0,
            'eval_winrate': float(eval_winrate) if eval_winrate is not None else 0.0,
            'avg_game_length': float(avg_game_length) if avg_game_length is not None else 0.0,
            'gradient_norms': gradient_norms if gradient_norms is not None else {},
            'policy_sharpness': float(policy_sharpness) if policy_sharpness is not None else 0.0,
            'mcts_improvement_avg': float(mcts_improvement_avg) if mcts_improvement_avg is not None else 0.0
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
