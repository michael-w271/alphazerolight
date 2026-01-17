#!/usr/bin/env python3
"""
Automated model testing that runs after each iteration.
Tests offensive and defensive capabilities and saves results to JSON.
"""
import sys
import os
import time
import json
from pathlib import Path
import numpy as np

# Get base directory and add src to path
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / 'src'))
os.chdir(base_dir)  # Change to project root

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
import torch

def clear_screen():
    """Clear terminal screen"""
    os.system('clear')

def print_header():
    """Print header"""
    clear_screen()
    print("=" * 80)
    print(" " * 20 + "🧪 ALPHAZERO MODEL TESTING - LIVE PROGRESS")
    print("=" * 80)
    print()

def test_model(model, mcts, game, iteration):
    """Run tactical tests on current model"""
    results = {
        'iteration': iteration,
        'timestamp': time.time(),
        'tests': {}
    }
    
    try:
        # Test 1: Can find immediate win
        print("  Test 1: Finding immediate win...")
        # Create state with 3 in a row vertically in column 3 (bottom rows 5,4,3)
        state = np.zeros((6, 7), dtype=np.float32)
        state[5, 3] = 1  # Bottom
        state[4, 3] = 1
        state[3, 3] = 1
        # Column 3 has 3 pieces at bottom, playing col 3 wins
        
        action_probs = mcts.search(state)
        win_prob = action_probs[3]
        results['tests']['find_win'] = {
            'passed': win_prob > 0.5,
            'probability': float(win_prob),
            'expected_column': 3
        }
        print(f"    → Win move probability: {win_prob:.3f} {'✅' if win_prob > 0.5 else '❌'}")
        
        # Test 2: Can block opponent threat
        print("  Test 2: Blocking opponent threat...")
        state = np.zeros((6, 7), dtype=np.float32)
        state[5, 5] = -1  # Bottom
        state[4, 5] = -1
        state[3, 5] = -1
        # Opponent has 3 in column 5 at bottom, MUST block
        
        action_probs = mcts.search(state)
        block_prob = action_probs[5]
        results['tests']['block_threat'] = {
            'passed': block_prob > 0.3,
            'probability': float(block_prob),
            'expected_column': 5
        }
        print(f"    → Block move probability: {block_prob:.3f} {'✅' if block_prob > 0.3 else '❌'}")
        
        # Test 3: Empty board evaluation
        print("  Test 3: Empty board value (should be ~0)...")
        state = np.zeros((6, 7), dtype=np.float32)
        with torch.no_grad():
            # Get encoded state for model
            encoded = game.get_encoded_state(state)
            policy, value = model(
                torch.tensor(encoded, dtype=torch.float32, device=model.device).unsqueeze(0)
            )
            value = value.item()
        
        results['tests']['empty_board_value'] = {
            'passed': abs(value) < 0.5,
            'value': float(value),
            'expected': 0.0
        }
        print(f"    → Empty board value: {value:.3f} {'✅' if abs(value) < 0.5 else '❌'}")
        
        # Test 4: Prefers center on empty board
        print("  Test 4: Prefers center opening...")
        action_probs = mcts.search(state)
        center_col = 3
        center_prob = action_probs[center_col]
        max_prob = np.max(action_probs)
        
        results['tests']['prefer_center'] = {
            'passed': bool(center_prob == max_prob),
            'center_probability': float(center_prob),
            'max_probability': float(max_prob)
        }
        print(f"    → Center preference: {center_prob:.3f} {'✅' if center_prob == max_prob else '❌'}")
        
        # Convert all numpy bools to Python bools for JSON serialization
        for test_key, test_data in results['tests'].items():
            if 'passed' in test_data and hasattr(test_data['passed'], 'item'):
                test_data['passed'] = bool(test_data['passed'])
        
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        results['error'] = str(e)
        # Fill in failed tests
        for test_name in ['find_win', 'block_threat', 'empty_board_value', 'prefer_center']:
            if test_name not in results['tests']:
                results['tests'][test_name] = {'passed': False, 'error': str(e)}
    
    # Overall score
    passed = sum(1 for test in results['tests'].values() if test.get('passed', False))
    total = len(results['tests'])
    results['overall_score'] = f"{passed}/{total}"
    results['pass_rate'] = float(passed) / float(total) if total > 0 else 0.0
    
    return results

def main():
    """Main testing loop"""
    base_dir = Path(__file__).parent.parent
    checkpoints_dir = base_dir / 'checkpoints' / 'connect4'  # Use correct subdirectory
    results_file = base_dir / 'model_test_results.json'
    
    # Initialize game and model
    game = ConnectFour()
    model = ResNet(game, num_res_blocks=10, num_hidden=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # MCTS config for testing
    args = {
        'C': 2,
        'num_searches': 100,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3
    }
    mcts = MCTS(game, args, model)
    
    # Load or create results history
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        # Get last tested iteration from history
        if all_results.get('test_history'):
            last_tested_iteration = all_results['test_history'][-1]['iteration']
        else:
            last_tested_iteration = -1
    else:
        all_results = {'test_history': []}
        last_tested_iteration = -1
    
    print_header()
    print(f"Last tested: iteration {last_tested_iteration}")
    print("Testing schedule: Every 10 iterations (10, 20, 30, 40, 50...)")
    print()
    print("Scanning for untested checkpoints...")
    sys.stdout.flush()
    
    while True:
        try:
            # Find all checkpoint files
            model_files = sorted(checkpoints_dir.glob('model_*.pt'))
            
            if not model_files:
                print("No checkpoints found yet, waiting...")
                sys.stdout.flush()
                time.sleep(5)
                continue
            
            # Find all iterations that should be tested
            all_iterations = [int(f.stem.split('_')[1]) for f in model_files]
            test_iterations = [i for i in all_iterations if i % 10 == 0 and i > last_tested_iteration]
            
            if not test_iterations:
                # No new tests needed, wait for next checkpoint
                latest_iteration = all_iterations[-1]
                next_test = ((latest_iteration // 10) + 1) * 10
                print_header()
                print(f"✅ All tests up to date (last tested: {last_tested_iteration})")
                print(f"Current iteration: {latest_iteration}")
                print(f"Waiting for iteration {next_test}...")
                print()
                
                # Show recent test results
                if all_results.get('test_history'):
                    print("📊 RECENT TEST RESULTS:")
                    print("=" * 80)
                    recent = all_results['test_history'][-5:]  # Last 5 tests
                    for r in recent:
                        it = r['iteration']
                        score = r['overall_score']
                        rate = r['pass_rate'] * 100
                        fw = '✅' if r['tests']['find_win']['passed'] else '❌'
                        bt = '✅' if r['tests']['block_threat']['passed'] else '❌'
                        eb = '✅' if r['tests']['empty_board_value']['passed'] else '❌'
                        pc = '✅' if r['tests']['prefer_center']['passed'] else '❌'
                        
                        print(f"  Iteration {it:3d}: {score} ({rate:.0f}%)")
                        print(f"    Win Detection: {fw}  |  Block Threat: {bt}  |  Board Eval: {eb}  |  Center Pref: {pc}")
                        
                        # Show probabilities for failed tests
                        if not r['tests']['find_win']['passed']:
                            prob = r['tests']['find_win'].get('probability', 0)
                            print(f"      → Win move prob: {prob:.3f} (need >0.5)")
                        if not r['tests']['block_threat']['passed']:
                            prob = r['tests']['block_threat'].get('probability', 0)
                            print(f"      → Block move prob: {prob:.3f} (need >0.3)")
                        print()
                    print("=" * 80)
                print()
                sys.stdout.flush()
                time.sleep(10)
                continue
            
            # Test the next untested iteration
            iteration = test_iterations[0]
            checkpoint_file = checkpoints_dir / f'model_{iteration}.pt'
            
            # Test this iteration
            print_header()
            print(f"📍 Testing Model at Iteration {iteration}")
            print(f"   ({test_iterations.index(iteration) + 1}/{len(test_iterations)} pending tests)")
            print("=" * 80)
            print()
            sys.stdout.flush()
            
            # Load model
            model.load_state_dict(torch.load(checkpoint_file, map_location=model.device))
            model.eval()
            
            # Run tests
            results = test_model(model, mcts, game, iteration)
            
            # Save results
            all_results['test_history'].append(results)
            all_results['latest_iteration'] = iteration
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Display summary
            print()
            print("=" * 80)
            print(f"📊 ITERATION {iteration} SUMMARY:")
            print(f"   Overall Score: {results['overall_score']} ({results['pass_rate']*100:.0f}%)")
            print()
            
            # Show recent progress
            if len(all_results['test_history']) > 1:
                print("📈 Recent Progress (last 5 iterations):")
                recent = all_results['test_history'][-5:]
                for r in recent:
                    it = r['iteration']
                    score = r['overall_score']
                    fw = '✅' if r['tests']['find_win']['passed'] else '❌'
                    bt = '✅' if r['tests']['block_threat']['passed'] else '❌'
                    print(f"   Iter {it:3d}: {score} | Win:{fw} Block:{bt}")
            
            print()
            print("=" * 80)
            print(f"💾 Results saved to: {results_file}")
            print()
            sys.stdout.flush()  # Force immediate output
            
            last_tested_iteration = iteration
            time.sleep(2)  # Brief pause before next test
            
        except KeyboardInterrupt:
            print("\n\n🛑 Testing stopped by user")
            sys.stdout.flush()
            break
        except Exception as e:
            import traceback
            print(f"\n⚠️  Error occurred: {e}")
            print("Stack trace:")
            traceback.print_exc()
            print("\n🔄 Continuing to next iteration...")
            sys.stdout.flush()
            time.sleep(5)

if __name__ == '__main__':
    main()
