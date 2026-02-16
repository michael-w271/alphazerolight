#!/usr/bin/env python3
"""
Test the final trained model (iteration 200) for tactical abilities.
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

def test_tactical_scenarios(iteration):
    """Test model on critical tactical positions"""
    
    print("="*70)
    print(f"TESTING MODEL AT ITERATION {iteration}")
    print("="*70)
    
    game = ConnectFour(6, 7, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = Path(f"checkpoints/connect4/model_{iteration}.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = ResNet(game, MODEL_CONFIG['num_res_blocks'], MODEL_CONFIG['num_hidden']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create MCTS
    args = {**MCTS_CONFIG, 'num_searches': 80}
    mcts = MCTS(game, args, model)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Empty Board',
            'state': game.get_initial_state(),
            'expected': 'Should be balanced (~0.0)',
        },
        {
            'name': 'Immediate Win Available (Column 3)',
            'state': np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
            ]),
            'expected': 'Should strongly prefer column 3 (win)',
            'best_move': 3,
        },
        {
            'name': 'Must Block Opponent Threat (Column 3)',
            'state': np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [-1, -1, -1, 0, 0, 0, 0],
            ]),
            'expected': 'Should block at column 3',
            'best_move': 3,
        },
        {
            'name': 'Vertical Win Setup (Column 2)',
            'state': np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
            ]),
            'expected': 'Should win at column 2',
            'best_move': 2,
        },
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"TEST: {scenario['name']}")
        print(f"{'='*70}")
        print("\nBoard:")
        print(scenario['state'])
        print(f"\nExpected: {scenario['expected']}")
        
        state = scenario['state']
        
        # Get neural network raw prediction
        encoded_state = game.get_encoded_state(state)
        encoded_state_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy_logits, value = model(encoded_state_tensor)
            policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()
        
        # Get MCTS decision
        mcts_probs = mcts.search(state, 1)  # Player 1's turn
        
        # Get valid moves
        valid_moves = game.get_valid_moves(state)
        
        print(f"\n📊 Neural Network Raw Output:")
        print(f"   Value: {value:+.4f}")
        print(f"   Policy (top 3):")
        top_moves = np.argsort(policy)[::-1][:3]
        for move in top_moves:
            if valid_moves[move]:
                print(f"      Column {move}: {policy[move]:.4f} ({policy[move]*100:.1f}%)")
        
        print(f"\n🌳 MCTS After Search (80 simulations):")
        print(f"   Visit distribution (top 3):")
        top_mcts = np.argsort(mcts_probs)[::-1][:3]
        for move in top_mcts:
            if valid_moves[move]:
                print(f"      Column {move}: {mcts_probs[move]:.4f} ({mcts_probs[move]*100:.1f}%)")
        
        # Check if best move is correct
        best_move = np.argmax(mcts_probs)
        expected_move = scenario.get('best_move')
        
        if expected_move is not None:
            if best_move == expected_move:
                result = f"✅ CORRECT - Chose column {best_move}"
                success = True
            else:
                result = f"❌ WRONG - Chose {best_move}, should be {expected_move}"
                success = False
                print(f"\n{result}")
        else:
            # No specific move expected, just check value
            result = f"Value: {value:+.4f}"
            success = abs(value) < 0.2  # Balanced for empty board
            if success:
                result = f"✅ {result} (balanced)"
            else:
                result = f"⚠️ {result} (biased)"
            print(f"\n{result}")
        
        results.append({
            'scenario': scenario['name'],
            'success': success,
            'best_move': best_move,
            'value': value,
        })
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - Iteration {iteration}")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\nTests Passed: {success_count}/{total_count}")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"  {status} {r['scenario']}")
    
    if success_count == total_count:
        print(f"\n🎉 Model has learned tactical play!")
    elif success_count >= total_count / 2:
        print(f"\n⚠️ Model shows some tactical awareness, but needs improvement")
    else:
        print(f"\n❌ Model has not learned tactical play yet")
    
    return results

if __name__ == "__main__":
    print("\n🧪 TACTICAL EVALUATION OF TRAINED MODEL\n")
    
    # Test final model
    test_tactical_scenarios(200)
    
    # Also test iteration 100 for comparison
    print("\n\n" + "🔄 "*30)
    print("COMPARING WITH ITERATION 100")
    print("🔄 "*30 + "\n")
    test_tactical_scenarios(100)
