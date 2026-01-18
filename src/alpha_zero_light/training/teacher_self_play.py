"""
Teacher-guided self-play method for AlphaZeroTrainer.

This replaces MCTS-heavy self-play with solver/NN-based move selection.
"""


def self_play_teacher_guided(self, temperature=None):
    """
    Self-play using teacher-solver guidance instead of heavy MCTS.
    
    Move selection priority:
    1. Forced-win override (solver detects win ≤8 moves)
    2. Scheduled solver probability by ply
    3. Safe opening random (plies 0-4, high-temp NN)
    4. NN-only moves
    5. Minimal MCTS fallback (rare)
    
    Args:
        temperature: Temperature for NN/opening random sampling
    
    Returns:
        (memory, outcome): Training data and game outcome
    """
    memory = []
    player = 1
    state = self.game.get_initial_state()
    
    if temperature is None:
        temperature = self.args.get('temperature', 1.0)
    
    # Track move sources for logging
    move_sources = []
    
    while True:
        # Count ply (number of pieces on board)
        ply = int(np.sum(np.abs(state)))
        
       # Change perspective to current player
        neutral_state = self.game.change_perspective(state, player)
       
        # Query teacher-solver for policy and value
        pi, v_target, meta = self.teacher_solver.query_teacher(
            state=state,
            player=player,
            ply=ply,
            model=self.model,
            mcts=self.mcts
        )
        
        # Store move source for stats
        move_sources.append(meta.get('source', 'unknown'))
        
        # Store in memory (will use game outcome or solver value)
        memory.append((neutral_state, pi, player, v_target, meta))
        
        # Sample action from policy with temperature
        temperature_pi = pi ** (1 / temperature)
        temperature_pi /= np.sum(temperature_pi)
        action = np.random.choice(self.game.action_size, p=temperature_pi)
        
        # Apply move
        state = self.game.get_next_state(state, action, player)
        
        # Check termination
        value, is_terminal = self.game.get_value_and_terminated(state, action)
        
        if is_terminal:
            # Game finished - process memory
            return_memory = []
            for hist_neutral_state, hist_pi, hist_player, hist_v_solver, hist_meta in memory:
                # Use game outcome as value target
                hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                
                # Store: (encoded_state, policy_target, value_target)
                return_memory.append((
                    self.game.get_encoded_state(hist_neutral_state),
                    hist_pi,
                    hist_outcome  # Use game outcome, not solver value
                ))
            
            # Game outcome from player 1's perspective
            model_outcome = value if player == 1 else self.game.get_opponent_value(value)
            
            return return_memory, model_outcome, move_sources
        
        player = self.game.get_opponent(player)
