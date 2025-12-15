# CRITICAL ENCODING BUG FIXED - Dec 15, 2025

## The Bug
During training, states were stored in "neutral" perspective (Player 1's view),  
but when encoding for the neural network, they were NOT converted to the  
acting player's perspective.

This meant:
- Player -1's moves were encoded with Player -1's pieces marked as "opponent"
- Player -1 learned: "When I see my own pieces encoded as opponent, I win"
- This inverted the learning completely!

## The Fix
Added perspective conversion before encoding:

```python
state_from_player_perspective = game.change_perspective(hist_state, hist_player)
worker_memory.append((
    game.get_encoded_state(state_from_player_perspective),
    hist_probs,
    hist_outcome
))
```

## Impact
ALL previous training (iterations 0-99) used the WRONG encoding.  
The model learned inverted patterns and collapsed to 1-3% win rate.

## Solution
Start fresh training with the fixed code.
