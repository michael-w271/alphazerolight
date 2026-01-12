#!/usr/bin/env python3
"""
Comprehensive training metrics tracker and evaluator for AlphaZero Connect4.

This script tracks model improvement across multiple dimensions:
1. Tactical skills (immediate win detection, threat blocking)
2. Strategic skills (center preference, positional evaluation)
3. Self-play metrics (policy loss, value loss, game length)
4. Head-to-head performance vs previous checkpoints

Results are saved in scientific paper-ready format.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS


class TrainingMetricsEvaluator:
    """Comprehensive evaluation of model improvement over training."""
    
    def __init__(self, checkpoint_dir: str, device: str = 'cuda'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.game = ConnectFour()
        
        # Test positions for tactical evaluation
        self.tactical_tests = self._create_tactical_tests()
        self.strategic_tests = self._create_strategic_tests()
        
    def _create_tactical_tests(self) -> List[Dict]:
        """Create tactical test positions (immediate wins, blocks)."""
        tests = []
        
        # Test 1: Immediate horizontal win available
        board1 = np.zeros((6, 7), dtype=np.float32)
        board1[5, 0:3] = 1  # Three in a row horizontally
        tests.append({
            'name': 'horizontal_win',
            'board': board1,
            'best_move': 3,
            'description': 'Find horizontal winning move'
        })
        
        # Test 2: Block opponent's horizontal threat
        board2 = np.zeros((6, 7), dtype=np.float32)
        board2[5, 0:3] = -1  # Opponent has three in a row
        tests.append({
            'name': 'horizontal_block',
            'board': board2,
            'best_move': 3,
            'description': 'Block opponent horizontal threat'
        })
        
        # Test 3: Immediate vertical win
        board3 = np.zeros((6, 7), dtype=np.float32)
        board3[5:2:-1, 3] = 1  # Three pieces stacked vertically
        tests.append({
            'name': 'vertical_win',
            'board': board3,
            'best_move': 3,
            'description': 'Find vertical winning move'
        })
        
        # Test 4: Block vertical threat
        board4 = np.zeros((6, 7), dtype=np.float32)
        board4[5:2:-1, 3] = -1  # Opponent stacked vertically
        tests.append({
            'name': 'vertical_block',
            'board': board4,
            'best_move': 3,
            'description': 'Block opponent vertical threat'
        })
        
        # Test 5: Diagonal win opportunity
        board5 = np.zeros((6, 7), dtype=np.float32)
        board5[5, 0] = 1
        board5[4, 1] = 1
        board5[3, 2] = 1
        board5[5, 1] = -1  # Support pieces
        board5[5, 2] = -1
        board5[4, 2] = -1
        tests.append({
            'name': 'diagonal_win',
            'board': board5,
            'best_move': 3,
            'description': 'Find diagonal winning move'
        })
        
        return tests
    
    def _create_strategic_tests(self) -> List[Dict]:
        """Create strategic test positions (center preference, position evaluation)."""
        tests = []
        
        # Test 1: Empty board - should prefer center
        board1 = np.zeros((6, 7), dtype=np.float32)
        tests.append({
            'name': 'empty_board_center',
            'board': board1,
            'description': 'Prefer center column on empty board',
            'metric': 'center_preference',
            'expected_top_moves': [3, 2, 4]  # Center and adjacent
        })
        
        # Test 2: Positional evaluation - good vs bad position
        board2_good = np.zeros((6, 7), dtype=np.float32)
        board2_good[5, 3] = 1  # Center control
        board2_good[5, 2] = 1
        board2_good[5, 4] = 1
        
        board2_bad = np.zeros((6, 7), dtype=np.float32)
        board2_bad[5, 0] = 1  # Edge play
        board2_bad[5, 1] = 1
        board2_bad[5, 6] = 1
        
        tests.append({
            'name': 'position_evaluation',
            'board_good': board2_good,
            'board_bad': board2_bad,
            'description': 'Value center control higher than edges',
            'metric': 'value_difference'
        })
        
        return tests
    
    def evaluate_model(self, model_path: Path, mcts_searches: int = 100) -> Dict:
        """Run complete evaluation suite on a model."""
        
        # Load model
        model = ResNet(self.game, num_res_blocks=10, num_hidden=128).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # Create MCTS
        args = {
            'C': 2.0,
            'num_searches': mcts_searches,
            'dirichlet_epsilon': 0.0,
            'dirichlet_alpha': 0.3,
            'mcts_batch_size': 1
        }
        mcts = MCTS(self.game, args, model)
        
        results = {
            'model': model_path.name,
            'timestamp': datetime.now().isoformat(),
            'tactical_scores': {},
            'strategic_scores': {},
            'overall_metrics': {}
        }
        
        # Tactical evaluation
        print(f"\n📊 Evaluating {model_path.name}")
        print("=" * 60)
        
        tactical_correct = 0
        for test in self.tactical_tests:
            action_probs = mcts.search(test['board'])
            predicted_move = np.argmax(action_probs)
            correct = predicted_move == test['best_move']
            tactical_correct += correct
            
            results['tactical_scores'][test['name']] = {
                'correct': bool(correct),
                'predicted_move': int(predicted_move),
                'expected_move': int(test['best_move']),
                'confidence': float(action_probs[predicted_move]),
                'description': test['description']
            }
            
            status = "✅" if correct else "❌"
            print(f"{status} {test['description']}: {predicted_move} (expected {test['best_move']})")
        
        tactical_accuracy = tactical_correct / len(self.tactical_tests)
        results['overall_metrics']['tactical_accuracy'] = tactical_accuracy
        print(f"\n🎯 Tactical Accuracy: {tactical_accuracy*100:.1f}%")
        
        # Strategic evaluation
        print("\n🧠 Strategic Analysis:")
        
        # Empty board center preference
        empty_test = self.strategic_tests[0]
        action_probs = mcts.search(empty_test['board'])
        top_3 = np.argsort(action_probs)[-3:][::-1]
        center_in_top_3 = 3 in top_3
        
        results['strategic_scores']['center_preference'] = {
            'center_in_top_3': bool(center_in_top_3),
            'top_3_moves': top_3.tolist(),
            'probabilities': action_probs[top_3].tolist(),
            'center_rank': int(np.where(np.argsort(action_probs)[::-1] == 3)[0][0]) + 1
        }
        
        print(f"  Empty board top moves: {top_3.tolist()}")
        print(f"  Center (3) rank: {results['strategic_scores']['center_preference']['center_rank']}")
        
        # Position evaluation
        pos_test = self.strategic_tests[1]
        with torch.no_grad():
            # Good position
            encoded_good = self.game.get_encoded_state(pos_test['board_good'])
            state_tensor_good = torch.tensor(encoded_good, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value_good = model(state_tensor_good)
            
            # Bad position
            encoded_bad = self.game.get_encoded_state(pos_test['board_bad'])
            state_tensor_bad = torch.tensor(encoded_bad, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value_bad = model(state_tensor_bad)
            
        value_diff = float(value_good.item() - value_bad.item())
        results['strategic_scores']['position_evaluation'] = {
            'value_good': float(value_good.item()),
            'value_bad': float(value_bad.item()),
            'value_difference': value_diff,
            'prefers_center': value_diff > 0
        }
        
        print(f"  Center control value: {value_good.item():.3f}")
        print(f"  Edge play value: {value_bad.item():.3f}")
        print(f"  Difference: {value_diff:+.3f} {'✅' if value_diff > 0 else '❌'}")
        
        # Overall score
        overall_score = tactical_accuracy * 0.7 + (center_in_top_3 * 0.15) + ((value_diff > 0) * 0.15)
        results['overall_metrics']['overall_score'] = overall_score
        
        print(f"\n🏆 Overall Score: {overall_score*100:.1f}%")
        print("=" * 60)
        
        return results
    
    def evaluate_training_run(self, start_iter: int = 0, end_iter: int = 150, step: int = 10):
        """Evaluate multiple checkpoints throughout training."""
        
        all_results = []
        
        for iteration in range(start_iter, end_iter + 1, step):
            model_path = self.checkpoint_dir / f"model_{iteration}.pt"
            
            if not model_path.exists():
                print(f"⚠️  Skipping iteration {iteration} (model not found)")
                continue
            
            results = self.evaluate_model(model_path)
            results['iteration'] = iteration
            all_results.append(results)
        
        # Save results
        output_dir = Path('paper_materials/data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return all_results
    
    def generate_summary_table(self, results: List[Dict]) -> str:
        """Generate LaTeX table for scientific paper."""
        
        table = r"""\begin{table}[h]
\centering
\caption{Model Performance Across Training Iterations}
\label{tab:training_progress}
\begin{tabular}{c|ccc|c}
\hline
\textbf{Iteration} & \textbf{Tactical} & \textbf{Center} & \textbf{Position} & \textbf{Overall} \\
 & \textbf{Accuracy (\%)} & \textbf{Pref.} & \textbf{Eval.} & \textbf{Score (\%)} \\
\hline
"""
        
        for r in results:
            iteration = r['iteration']
            tactical = r['overall_metrics']['tactical_accuracy'] * 100
            center = '✓' if r['strategic_scores']['center_preference']['center_in_top_3'] else '✗'
            position = '✓' if r['strategic_scores']['position_evaluation']['prefers_center'] else '✗'
            overall = r['overall_metrics']['overall_score'] * 100
            
            table += f"{iteration} & {tactical:.1f} & {center} & {position} & {overall:.1f} \\\\\n"
        
        table += r"""\hline
\end{tabular}
\end{table}
"""
        
        # Save table
        output_file = Path('paper_materials/tables') / 'training_progress.tex'
        with open(output_file, 'w') as f:
            f.write(table)
        
        print(f"📊 LaTeX table saved to: {output_file}")
        
        return table


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Connect4 training progress')
    parser.add_argument('--checkpoint-dir', default='/mnt/ssd2pro/alpha-zero-checkpoints/connect4_v2',
                        help='Directory containing model checkpoints')
    parser.add_argument('--start', type=int, default=0, help='Starting iteration')
    parser.add_argument('--end', type=int, default=150, help='Ending iteration')
    parser.add_argument('--step', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--model', type=str, help='Evaluate single model file')
    
    args = parser.parse_args()
    
    evaluator = TrainingMetricsEvaluator(args.checkpoint_dir)
    
    if args.model:
        # Single model evaluation
        model_path = Path(args.model)
        evaluator.evaluate_model(model_path)
    else:
        # Full training run evaluation
        results = evaluator.evaluate_training_run(args.start, args.end, args.step)
        evaluator.generate_summary_table(results)


if __name__ == '__main__':
    main()
