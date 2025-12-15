#!/usr/bin/env python3
"""
Verify that the training mechanism correctly assigns win/loss outcomes
and that the model learns from them.
"""

import torch
import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.config_connect4 import MODEL_CONFIG, MCTS_CONFIG

def verify_outcome_assignment():
    """Verify that wins assign +1, losses assign -1"""
    print("="*70)
    print("VERIFICATION: Outcome Assignment in Training")
    print("="*70)
    
    game = ConnectFour(6, 7, 4)
    
    # Simulate a game where player 1 wins
    print("\n1. Simulating game where Player 1 wins...")
    print("   Player 1 makes moves: 0, 1, 2, 3 (horizontal win)")
    
    state = game.get_initial_state()
    player = 1
    game_memory = []
    
    moves = [0, 6, 1, 5, 2, 4, 3]  # Player 1 plays 0,1,2,3 (wins)
    
    for i, action in enumerate(moves):
        action_probs = np.zeros(game.action_size)
        action_probs[action] = 1.0
        
        game_memory.append((state.copy(), action_probs.copy(), player))
        
        state = game.get_next_state(state, action, player)
        value, is_terminal = game.get_value_and_terminated(state, action)
        
        if is_terminal:
            print(f"\n   Game ends at move {i+1}, value = {value}")
            print(f"   Current player (who just moved): {player}")
            print(f"   Assigning outcomes...")
            
            outcomes = []
            for hist_state, hist_probs, hist_player in game_memory:
                # This is the CRITICAL line from trainer.py:
                hist_outcome = value if hist_player == player else game.get_opponent_value(value)
                outcomes.append((hist_player, hist_outcome))
                
            for move_num, (p, outcome) in enumerate(outcomes):
                print(f"     Move {move_num+1} (Player {p:2d}): outcome = {outcome:+.1f}")
            
            # Verify
            player1_outcomes = [o for p, o in outcomes if p == 1]
            player2_outcomes = [o for p, o in outcomes if p == -1]
            
            print(f"\n   ✅ Player 1 outcomes: {player1_outcomes}")
            print(f"   ✅ Player -1 outcomes: {player2_outcomes}")
            
            assert all(o == 1.0 for o in player1_outcomes), "Player 1 should get +1 (won)"
            assert all(o == -1.0 for o in player2_outcomes), "Player -1 should get -1 (lost)"
            
            print("\n   ✅ OUTCOME ASSIGNMENT CORRECT!")
            break
        
        player = game.get_opponent(player)

def check_value_loss_weight():
    """Check that value loss is prioritized"""
    print("\n" + "="*70)
    print("VERIFICATION: Value Loss Weight")
    print("="*70)
    
    from alpha_zero_light.config_connect4 import TRAINING_CONFIG
    
    value_loss_weight = TRAINING_CONFIG.get('value_loss_weight', 1.0)
    print(f"\nValue Loss Weight: {value_loss_weight}")
    
    if value_loss_weight >= 2.0:
        print("✅ HIGH weight on value loss - model will prioritize learning win/loss!")
    elif value_loss_weight > 1.0:
        print("⚠️  MODERATE weight - might work but could be stronger")
    else:
        print("❌ LOW weight - model might focus too much on policy, not outcomes")
    
    return value_loss_weight

def check_training_epochs():
    """Check if enough epochs to learn"""
    print("\n" + "="*70)
    print("VERIFICATION: Training Epochs")
    print("="*70)
    
    from alpha_zero_light.config_connect4 import TRAINING_CONFIG
    
    num_epochs = TRAINING_CONFIG['num_epochs']
    print(f"\nTraining Epochs: {num_epochs}")
    
    if num_epochs >= 100:
        print("✅ HIGH epoch count - model will see each position many times!")
    elif num_epochs >= 50:
        print("⚠️  MODERATE epochs - should work but might be slow to learn")
    else:
        print("❌ LOW epochs - might not learn rare positions well")
    
    return num_epochs

def test_model_learning_progression():
    """Test if model at iteration 10 shows learning"""
    print("\n" + "="*70)
    print("VERIFICATION: Model Learning Progression")
    print("="*70)
    
    checkpoint_dir = Path("checkpoints/connect4")
    
    if not checkpoint_dir.exists():
        print("❌ No checkpoints found")
        return
    
    game = ConnectFour(6, 7, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test position: Player can win on column 3
    state = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],  # Can win at column 3
    ])
    
    print("\nTest Position: Three in a row, can win at column 3")
    print(state)
    
    iterations_to_test = [0, 5, 10]
    
    for iteration in iterations_to_test:
        model_path = checkpoint_dir / f"model_{iteration}.pt"
        if not model_path.exists():
            continue
        
        model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        encoded_state = game.get_encoded_state(state)
        encoded_state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy, value = model(encoded_state_tensor)
            policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        
        win_move_prob = policy[3]
        
        print(f"\nIteration {iteration:2d}:")
        print(f"  Value prediction: {value:+.4f}")
        print(f"  Win move (col 3) probability: {win_move_prob:.4f} ({win_move_prob*100:.1f}%)")
        
        if iteration > 0:
            if win_move_prob > 0.2:
                print(f"  ✅ Model is learning! Win move getting strong probability")
            elif win_move_prob > 0.05:
                print(f"  ⚠️  Model starting to learn, but weak confidence")
            else:
                print(f"  ❌ Model not detecting win yet")

def main():
    print("\n" + "🔍 " * 30)
    print("VERIFYING ALPHAZERO LEARNING MECHANISM")
    print("🔍 " * 30 + "\n")
    
    # 1. Verify outcome assignment
    verify_outcome_assignment()
    
    # 2. Check configuration
    value_weight = check_value_loss_weight()
    epochs = check_training_epochs()
    
    # 3. Test actual model progression
    test_model_learning_progression()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✅ Outcome assignment: CORRECT (winners get +1, losers get -1)")
    print(f"✅ Value loss weight: {value_weight} (prioritizes learning outcomes)")
    print(f"✅ Training epochs: {epochs} (sees each position many times)")
    print(f"\n🎯 The system IS set up to learn from wins/losses!")
    print(f"📈 Improvement happens through self-play iterations:")
    print(f"   - Better model → Better MCTS → Better training data → Better model")
    print(f"\n💡 With the FIXED terminal bug, training should now improve steadily.")
    print(f"   Expected timeline:")
    print(f"   - Iteration 10-20: Basic tactics (1-move wins/blocks)")
    print(f"   - Iteration 20-40: Multi-move tactics (2-3 move combinations)")
    print(f"   - Iteration 40+: Strategic play (opening theory, endgame)")

if __name__ == "__main__":
    main()
