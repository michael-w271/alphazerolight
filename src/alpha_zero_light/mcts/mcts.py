import torch
import math
import numpy as np

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0, 
                 is_terminal=False, terminal_value=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
        # Store terminal state info (checked before perspective flip)
        self.is_terminal = is_terminal
        self.terminal_value = terminal_value
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_ucb = -np.inf
        best_children = []
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb + 1e-12:
                best_ucb = ucb
                best_children = [child]
            elif abs(ucb - best_ucb) <= 1e-12:
                best_children.append(child)
        
        # Break ties randomly to avoid deterministic artifacts
        return best_children[np.random.randint(len(best_children))]
    
    def get_ucb(self, child):
        """
        PUCT score from *this node's perspective*.
        NOTE: child.value_sum/visit_count is from the CHILD's perspective (child is opponent-to-move),
        so we negate it to obtain Q from the parent's perspective.
        """
        if child.visit_count == 0:
            q_parent = 0.0
        else:
            q_child = child.value_sum / child.visit_count
            q_parent = -q_child  # CRITICAL: convert to parent perspective
        
        # Standard PUCT exploration term with proper visit count handling
        parent_visits = max(1, self.visit_count)
        u = (
            self.args['C']
            * child.prior
            * (math.sqrt(parent_visits) / (1 + child.visit_count))
        )
        return q_parent + u
    
    def expand(self, policy):
        # Only expand once - critical bug fix!
        if len(self.children) > 0:
            return  # Already expanded
        
        # CRITICAL: Get valid moves and expand ALL legal actions
        # Even if network assigns zero probability, we must add the child
        # Otherwise tactical moves (like blocking) can be permanently excluded
        valid_moves = self.game.get_valid_moves(self.state)
            
        for action in range(self.game.action_size):
            # Skip invalid moves
            if valid_moves[action] == 0:
                continue
            
            # Get probability and clamp to minimum prior
            # This ensures MCTS can explore all legal moves via UCB exploration term
            prob = float(policy[action])
            prob = max(prob, 1e-8)  # Force minimum prior for UCB exploration
            
            if isinstance(self.state, torch.Tensor):
                child_state = self.state.clone()
            else:
                child_state = self.state.copy()
                
            # Apply the move from current player's perspective
            child_state = self.game.get_next_state(child_state, action, 1)
            
            # CRITICAL FIX: Check terminal state BEFORE flipping perspective
            # This ensures we check if the action wins for the current player
            terminal_value, is_terminal = self.game.get_value_and_terminated(child_state, action)
            
            # Now flip to opponent's perspective for the child node
            child_state = self.game.change_perspective(child_state, player=-1)
            
            # If terminal from parent's view (value=1 means parent won),
            # from child's view this is a loss (value=-1)
            if is_terminal and terminal_value != 0:
                terminal_value = -terminal_value
            
            child = Node(self.game, self.args, child_state, self, action, prob,
                       is_terminal=is_terminal, terminal_value=terminal_value)
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
        MCTS search with GPU-optimized batched inference.
        Collects multiple leaf nodes and evaluates them together for better GPU utilization.
        """
        import os
        debug = os.environ.get('DEBUG_MCTS') == '1'
        
        root = Node(self.game, self.args, state, visit_count=0)
        
        # Batch size for leaf evaluation
        # Use configured batch size (default 1 to avoid duplicate-leaf artifacts)
        # For Connect4 tactics, batch_size=1 is critical to avoid shallow-tree distortion
        batch_size = int(self.args.get('mcts_batch_size', 1))
        batch_size = max(1, min(batch_size, self.args['num_searches']))
        
        if debug:
            print(f"Starting search: {self.args['num_searches']} searches, batch_size={batch_size}")
        
        for batch_start in range(0, self.args['num_searches'], batch_size):
            batch_end = min(batch_start + batch_size, self.args['num_searches'])
            current_batch_size = batch_end - batch_start
            
            if debug and batch_start % 32 == 0:
                print(f"  Batch {batch_start}-{batch_end}, root expanded: {len(root.children) > 0}, root visits: {root.visit_count}")
            
            # Collect leaf nodes for this batch
            leaf_nodes = []
            for _ in range(current_batch_size):
                node = root
                while node.is_fully_expanded():
                    node = node.select()
                leaf_nodes.append(node)
            
            # Separate terminal and non-terminal nodes
            non_terminal_nodes = []
            non_terminal_states = []
            
            for node in leaf_nodes:
                # Use the pre-computed terminal info stored in the node
                # This was checked BEFORE perspective flip, so it's correct
                if node.is_terminal:
                    # Terminal node - backpropagate immediately
                    # terminal_value is already from this node's perspective (flipped in expand())
                    node.backpropagate(node.terminal_value)
                else:
                    non_terminal_nodes.append(node)
                    non_terminal_states.append(self.game.get_encoded_state(node.state))
            
            # Batch evaluate non-terminal nodes on GPU
            if non_terminal_nodes:
                states_tensor = torch.tensor(np.array(non_terminal_states), 
                                            dtype=torch.float32, 
                                            device=self.model.device)
                
                with torch.no_grad():
                    policies, values = self.model(states_tensor)
                    policies = torch.softmax(policies, dim=1).cpu().numpy()
                    values = values.cpu().numpy().flatten()
                
                # Expand and backpropagate for each non-terminal node
                for idx, node in enumerate(non_terminal_nodes):
                    policy = policies[idx]
                    value = values[idx]
                    
                    # Mask invalid moves
                    valid_moves = self.game.get_valid_moves(node.state)
                    policy *= valid_moves
                    policy_sum = np.sum(policy)
                    if policy_sum > 0:
                        policy /= policy_sum
                    else:
                        policy = valid_moves / np.sum(valid_moves)
                    
                    # Add Dirichlet noise only at root
                    if node == root and add_noise:
                        noise = np.random.dirichlet([self.args.get('dirichlet_alpha', 0.3)] * len(policy))
                        epsilon = self.args.get('dirichlet_epsilon', 0.25)
                        policy = (1 - epsilon) * policy + epsilon * noise
                        policy *= valid_moves
                        policy /= np.sum(policy)
                    
                    node.expand(policy)
                    node.backpropagate(value)    
            
        # Return action probabilities based on visit counts
        action_probs = np.zeros(self.game.action_size)
        
        # DEBUG: Print visit counts
        import os
        if os.environ.get('DEBUG_MCTS') == '1':
            print(f"\n=== MCTS Debug: Visit Counts After {self.args.get('num_searches', 50)} Searches ===")
            for child in root.children:
                q_val = child.value_sum / child.visit_count if child.visit_count > 0 else 0
                print(f"  Action {child.action_taken}: visits={child.visit_count:4d}, "
                      f"value_sum={child.value_sum:7.2f}, Q={q_val:6.3f}, "
                      f"prior={child.prior:.4f}, terminal={child.is_terminal}, "
                      f"term_val={child.terminal_value}")
        
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
            
        # Ensure we have valid probabilities
        if np.sum(action_probs) == 0:
            # No visits - return uniform over valid moves
            valid_moves = self.game.get_valid_moves(root.state)
            action_probs = valid_moves / np.sum(valid_moves)
            return action_probs
            
        # Apply temperature for action selection
        if temperature is None:
            temperature = self.args.get('temperature', 1.0)
            
        if temperature == 0:
            # Argmax (deterministic)
            best_action = np.argmax(action_probs)
            action_probs = np.zeros_like(action_probs)
            action_probs[best_action] = 1
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
                # Use pre-computed terminal info from the node
                if node.is_terminal:
                    values[i] = node.terminal_value
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
                    
                    # Add Dirichlet noise at root for exploration (CRITICAL for learning!)
                    if node.parent is None:  # Root node
                        noise = np.random.dirichlet([self.args.get('dirichlet_alpha', 0.3)] * len(policy))
                        epsilon = self.args.get('dirichlet_epsilon', 0.25)
                        policy = (1 - epsilon) * policy + epsilon * noise
                    
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
