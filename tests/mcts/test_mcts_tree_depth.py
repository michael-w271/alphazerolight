#!/usr/bin/env python3
"""
Deep dive into MCTS search tree to see if it's exploring opponent responses.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS, Node


def load_latest_model(game):
    """Load the most recent model checkpoint"""
    checkpoint_dir = Path("checkpoints/connect4")
    model_files = list(checkpoint_dir.glob("model_*.pt"))
    if not model_files:
        return None
    
    latest_model = max(model_files, key=lambda p: int(p.stem.split('_')[1]))
    iteration = int(latest_model.stem.split('_')[1])
    
    model = ResNet(game, num_res_blocks=10, num_hidden=128)
    model.load_state_dict(torch.load(latest_model, map_location='cpu'))
    model.eval()
    
    return model, iteration


def print_board(state):
    """Print the board"""
    print("  0 1 2 3 4 5 6")
    for row in range(6):
        print(f"{row} ", end="")
        for col in range(7):
            val = state[row, col]
            if val == 1:
                print("X ", end="")
            elif val == -1:
                print("O ", end="")
            else:
                print(". ", end="")
        print()


def analyze_tree_depth(root_node, max_depth=3):
    """Recursively analyze the MCTS tree"""
    
    def explore_node(node, depth=0, action_path=[]):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        if depth == 0:
            print(f"{indent}ROOT (visits={node.visit_count})")
        
        for child in node.children:
            q_val = child.value_sum / child.visit_count if child.visit_count > 0 else 0
            path_str = " -> ".join(map(str, action_path + [child.action_taken]))
            
            print(f"{indent}  Action {child.action_taken} [path: {path_str}]")
            print(f"{indent}    visits={child.visit_count}, Q={q_val:+.3f}, "
                  f"terminal={child.is_terminal}, term_val={child.terminal_value}")
            
            if child.visit_count > 0 and depth < max_depth:
                explore_node(child, depth + 1, action_path + [child.action_taken])
    
    explore_node(root_node)


def main():
    game = ConnectFour(row_count=6, column_count=7, win_length=4)
    
    model, iteration = load_latest_model(game)
    if model is None:
        print("No model found")
        return
    
    print(f"="*70)
    print(f"MCTS TREE DEPTH ANALYSIS - Iteration {iteration}")
    print(f"="*70)
    
    # Create blocking scenario
    state = game.get_initial_state()
    state = game.get_next_state(state, 0, 1)
    state = game.get_next_state(state, 5, -1)
    state = game.get_next_state(state, 1, 1)
    state = game.get_next_state(state, 5, -1)
    state = game.get_next_state(state, 2, 1)
    state = game.get_next_state(state, 5, -1)
    
    print("\nBoard position:")
    print_board(state)
    print("\nO (opponent) has 3 in column 5")
    print("X MUST block at column 5 or O wins next turn")
    
    # Run MCTS and capture the root
    args = {
        'num_searches': 100,
        'C': 2.0,
        'dirichlet_epsilon': 0.0,
        'dirichlet_alpha': 0.3
    }
    
    mcts = MCTS(game, args, model)
    neutral_state = game.change_perspective(state, 1)
    
    # Manually build root to analyze
    root = Node(game, args, neutral_state, visit_count=0)
    
    # Do one batch of searches
    batch_size = 8
    for batch_start in range(0, 100, batch_size):
        batch_end = min(batch_start + batch_size, 100)
        current_batch_size = batch_end - batch_start
        
        leaf_nodes = []
        for _ in range(current_batch_size):
            node = root
            while node.is_fully_expanded():
                node = node.select()
            leaf_nodes.append(node)
        
        non_terminal_nodes = []
        non_terminal_states = []
        
        for node in leaf_nodes:
            if node.is_terminal:
                node.backpropagate(node.terminal_value)
            else:
                non_terminal_nodes.append(node)
                non_terminal_states.append(game.get_encoded_state(node.state))
        
        if non_terminal_nodes:
            states_tensor = torch.tensor(np.array(non_terminal_states), 
                                        dtype=torch.float32, 
                                        device=model.device)
            
            with torch.no_grad():
                policies, values = model(states_tensor)
                policies = torch.softmax(policies, dim=1).cpu().numpy()
                values = values.cpu().numpy().flatten()
            
            for idx, node in enumerate(non_terminal_nodes):
                policy = policies[idx]
                value = values[idx]
                
                valid_moves = game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                node.expand(policy)
                node.backpropagate(value)
    
    print(f"\n{'='*70}")
    print("MCTS TREE STRUCTURE (first 2 levels):")
    print(f"{'='*70}\n")
    
    analyze_tree_depth(root, max_depth=2)
    
    # Specifically check what happens after bad moves
    print(f"\n{'='*70}")
    print("CHECKING: What happens if X plays column 0 (doesn't block)?")
    print(f"{'='*70}")
    
    col0_child = next((c for c in root.children if c.action_taken == 0), None)
    if col0_child and col0_child.children:
        print(f"\nAfter X plays column 0, opponent can respond:")
        for grandchild in col0_child.children:
            q = grandchild.value_sum / grandchild.visit_count if grandchild.visit_count > 0 else 0
            print(f"  Opponent column {grandchild.action_taken}: visits={grandchild.visit_count}, "
                  f"Q={q:+.3f}, terminal={grandchild.is_terminal}, term_val={grandchild.terminal_value}")
            
            if grandchild.action_taken == 5:
                print(f"    ^^^ This should be terminal with term_val=+1 (opponent wins!)")
    else:
        print("Column 0 was not explored deeply enough!")


if __name__ == "__main__":
    main()
