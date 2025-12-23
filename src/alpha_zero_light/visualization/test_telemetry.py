"""
Test script for telemetry publisher - sends mock training frames.

Usage:
    python -m alpha_zero_light.visualization.test_telemetry

This sends mock frame messages at 10 Hz to test C++ viewer connection.
Press Ctrl+C to stop.
"""

import time
import numpy as np
from alpha_zero_light.visualization.telemetry import TelemetryPublisher


def main():
    print("🔬 Starting telemetry test publisher")
    print("   Endpoint: tcp://127.0.0.1:5556")
    print("   Frequency: 10 Hz (mock Connect Four frames)")
    print("   Press Ctrl+C to stop\n")
    
    publisher = TelemetryPublisher("tcp://127.0.0.1:5556", send_frame_frequency=1)
    
    # Mock Connect Four game state
    iteration = 0
    game_idx = 0
    move_idx = 0
    
    try:
        while True:
            # Create mock board (6x7 with some pieces)
            board = np.random.choice([-1, 0, 1], size=(6, 7), p=[0.2, 0.4, 0.4])
            valid_moves = (board[0, :] == 0).astype(int)  # Top row empty = valid
            
            # Mock network outputs
            policy_head = np.random.dirichlet([1.0] * 7)
            value_head = np.random.uniform(-1, 1)
            
            # Mock MCTS results
            mcts_policy = np.random.dirichlet([2.0] * 7)
            mcts_policy *= valid_moves
            if mcts_policy.sum() > 0:
                mcts_policy /= mcts_policy.sum()
            
            root_visits = np.random.randint(0, 100, size=7).tolist()
            root_q = (np.random.randn(7) * 0.5).tolist()
            
            chosen_action = np.random.choice(np.where(valid_moves)[0])
            
            # Send frame
            publisher.send_frame(
                iteration=iteration,
                game_idx=game_idx,
                move_idx=move_idx,
                player=1 if move_idx % 2 == 0 else -1,
                board=board,
                valid_moves=valid_moves,
                policy_head=policy_head,
                value_head=value_head,
                mcts_policy=mcts_policy,
                root_visits=root_visits,
                root_q=root_q,
                chosen_action=chosen_action,
                temperature=1.0,
                is_terminal=False
            )
            
            print(f"📤 Sent frame: iter={iteration}, game={game_idx}, move={move_idx}, action={chosen_action}", end='\r')
            
            # Advance state
            move_idx += 1
            if move_idx >= 20:  # Mock game end
                move_idx = 0
                game_idx += 1
                if game_idx >= 5:
                    game_idx = 0
                    iteration += 1
                    
                    # Also send metrics on iteration boundary
                    publisher.send_metrics(
                        iteration=iteration,
                        total_loss=np.random.uniform(0.5, 2.0),
                        policy_loss=np.random.uniform(0.3, 1.0),
                        value_loss=np.random.uniform(0.2, 1.0),
                        eval_winrate=np.random.uniform(0.3, 0.7)
                    )
                    print(f"\n📊 Sent metrics for iteration {iteration}")
            
            time.sleep(0.1)  # 10 Hz
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping telemetry test")
        publisher.close()


if __name__ == "__main__":
    main()
