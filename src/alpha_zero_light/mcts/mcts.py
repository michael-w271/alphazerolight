import torch
import math
import numpy as np

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                if isinstance(self.state, torch.Tensor):
                    child_state = self.state.clone()
                else:
                    child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=0)
        
        for i in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                # Get encoded state (already a tensor)
                encoded = self.game.get_encoded_state(node.state)
                # Ensure it's (3, H, W), add batch dim to get (1, 3, H, W)
                if encoded.ndim == 4:
                    encoded = encoded.squeeze(0)
                policy, value = self.model(encoded.unsqueeze(0).to(self.model.device))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                # Ensure valid_moves is numpy 1D array for compatibility
                if isinstance(valid_moves, torch.Tensor):
                    valid_moves = valid_moves.cpu().numpy().flatten()
                elif hasattr(valid_moves, 'flatten'):
                    valid_moves = valid_moves.flatten()
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

    @torch.no_grad()
    def search_batch(self, states, num_searches=None):
        """
        Batched MCTS search for multiple states simultaneously.
        This maximizes GPU utilization by batching neural network calls.
        """
        if num_searches is None:
            num_searches = self.args['num_searches']
            
        # Try to use C++ implementation
        try:
            import alpha_zero_light.mcts.mcts_cpp as mcts_cpp
            
            # Initialize C++ MCTS if not already done or if args changed
            if not hasattr(self, 'cpp_mcts'):
                self.cpp_mcts = mcts_cpp.MCTS_CPP(
                    self.game.board_size, 
                    num_searches, 
                    self.args['C']
                )
            
            # Wrapper for model to handle numpy <-> tensor conversion
            def model_wrapper(input_np):
                # input_np is (B, 3, H, W) numpy array from C++ MCTS
                # It reshapes the flat board states into encoded states
                input_tensor = torch.tensor(input_np, device=self.model.device, dtype=torch.float32)
                
                # Handle various input shapes
                if input_tensor.ndim == 5 and input_tensor.shape[1] == 1:
                    # Fix 5D input (B, 1, 3, H, W) -> (B, 3, H, W)
                    input_tensor = input_tensor.squeeze(1)
                elif input_tensor.ndim == 2:
                    # If input is (1, B*3*H*W), reshape to (B, 3, H, W)
                    batch_size = input_tensor.shape[0] if input_tensor.shape[0] > 1 else input_tensor.shape[1] // (3 * self.game.board_size * self.game.board_size)
                    input_tensor = input_tensor.reshape(batch_size, 3, self.game.board_size, self.game.board_size)
               
                policies, values = self.model(input_tensor)
                
                policies_np = torch.softmax(policies, axis=1).cpu().numpy()
                values_np = values.cpu().numpy()
                
                return policies_np, values_np
            
            # Run C++ search
            # states is a list of (H, W) or (1, H, W) arrays. 
            # C++ expects list of flat numpy arrays or similar.
            # Our states from GomokuGPU are tensors (1, H, W).
            # We need to convert them to numpy for C++.
            
            states_np = []
            for s in states:
                if isinstance(s, torch.Tensor):
                    s = s.cpu().numpy()
                states_np.append(s.flatten())
                
            # print("DEBUG: Calling C++ search_batch")
            cpp_results = self.cpp_mcts.search_batch(states_np, model_wrapper)
            
            # Convert list of lists to list of numpy arrays
            action_probs = [np.array(probs) for probs in cpp_results]
            return action_probs
            
        except ImportError:
            # Fallback to Python implementation
            pass
        except Exception as e:
            print(f"C++ MCTS failed, falling back to Python: {e}")
            import traceback
            traceback.print_exc()
            # Fallback
            
        batch_size = len(states)
        roots = [Node(self.game, self.args, state, visit_count=0) for state in states]
        
        for _ in range(num_searches):
            # 1. Selection
            nodes = []
            for root in roots:
                node = root
                while node.is_fully_expanded():
                    node = node.select()
                nodes.append(node)
            
            # 2. Evaluation
            encoded_states = []
            valid_indices = []
            values = np.zeros(batch_size)
            
            for i, node in enumerate(nodes):
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    values[i] = value
                else:
                    encoded_states.append(self.game.get_encoded_state(node.state))
                    valid_indices.append(i)
            
            if encoded_states:
                # Batch inference
                # encoded_states are already tensors on device (from GomokuGPU)
                if isinstance(encoded_states[0], torch.Tensor):
                    encoded_states_tensor = torch.cat(encoded_states, dim=0)
                else:
                    encoded_states_tensor = torch.tensor(np.array(encoded_states), device=self.model.device)
                
                # print(f"DEBUG: Python MCTS input shape: {encoded_states_tensor.shape}")
                if encoded_states_tensor.ndim == 5:
                     encoded_states_tensor = encoded_states_tensor.squeeze(1)
                
                policies, nn_values = self.model(encoded_states_tensor)
                
                policies = torch.softmax(policies, axis=1).cpu().numpy()
                nn_values = nn_values.cpu().numpy().flatten()
                
                # 3. Expansion
                for idx, policy, nn_value in zip(valid_indices, policies, nn_values):
                    node = nodes[idx]
                    valid_moves = self.game.get_valid_moves(node.state)
                    if isinstance(valid_moves, torch.Tensor):
                        valid_moves = valid_moves.cpu().numpy().flatten()
                    
                    policy *= valid_moves
                    policy /= np.sum(policy)
                    
                    node.expand(policy)
                    values[idx] = nn_value
            
            # 4. Backpropagation
            for node, value in zip(nodes, values):
                node.backpropagate(value)
        
        # Calculate action probabilities
        all_action_probs = []
        for root in roots:
            action_probs = np.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            all_action_probs.append(action_probs)
            
        return all_action_probs
