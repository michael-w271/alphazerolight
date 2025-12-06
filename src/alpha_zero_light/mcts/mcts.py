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
        # Use actual Q-value (average reward), not inverted
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count
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
    def search(self, state, add_noise=True, temperature=None):
        """
        MCTS search from given state.
        add_noise: Add Dirichlet noise at root for exploration
        temperature: Temperature for action selection (if None, uses config)
        """
        root = Node(self.game, self.args, state, visit_count=0)
        
        for i in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                # Add Dirichlet noise at root node for exploration
                if node == root and add_noise:
                    noise = np.random.dirichlet([self.args.get('dirichlet_alpha', 0.3)] * len(policy))
                    epsilon = self.args.get('dirichlet_epsilon', 0.25)
                    policy = (1 - epsilon) * policy + epsilon * noise
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
        # Return action probabilities based on visit counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
            
        # Apply temperature for action selection
        if temperature is None:
            temperature = self.args.get('temperature', 1.0)
            
        if temperature == 0:
            # Argmax (deterministic)
            action_probs = np.zeros_like(action_probs)
            action_probs[np.argmax(action_probs)] = 1
        else:
            # Temperature-based sampling
            action_probs = action_probs ** (1.0 / temperature)
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
                # input_np is (B, 3, H, W) numpy array
                input_tensor = torch.tensor(input_np, device=self.model.device, dtype=torch.float32)
                
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
                
            cpp_results = self.cpp_mcts.search_batch(states_np, model_wrapper)
            
            # Convert list of lists to list of numpy arrays
            action_probs = [np.array(probs) for probs in cpp_results]
            return action_probs
            
        except ImportError:
            # Fallback to Python implementation
            pass
        except Exception as e:
            print(f"C++ MCTS failed, falling back to Python: {e}")
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
