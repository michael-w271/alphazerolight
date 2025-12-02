import torch
import torch.optim as optim
import sys
import os
import time
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.gomoku_gpu import GomokuGPU
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.config_gomoku_gpu_test import TRAINING_CONFIG, MCTS_CONFIG, MODEL_CONFIG, PATHS

def simple_self_play_gpu(game, model, num_games, num_searches):
    """
    Simplified self-play on GPU without full MCTS.
    Uses policy network directly for faster training test.
    """
    device = model.device
    states = game.get_initial_state(num_games)
    players = torch.ones(num_games, device=device)
    
    memory = []
    active_games = torch.ones(num_games, dtype=torch.bool, device=device)
    
    max_moves = 81  # 9x9 board
    
    for move_num in range(max_moves):
        if not active_games.any():
            break
            
        # Get policy from model
        encoded = game.get_encoded_state(states)
        with torch.no_grad():
            policy_logits, values = model(encoded)
        
        # Mask invalid moves
        valid_moves = game.get_valid_moves(states)
        policy_logits[valid_moves == 0] = -float('inf')
        
        # Sample actions (with temperature for exploration)
        probs = torch.softmax(policy_logits, dim=1)
        actions = torch.multinomial(probs, 1).squeeze(1)
        
        # Store experience
        for i in range(num_games):
            if active_games[i]:
                memory.append({
                    'state': encoded[i].cpu(),
                    'policy': probs[i].cpu(),
                    'player': players[i].item()
                })
        
        # Execute moves
        states = game.get_next_state(states, actions, players)
        
        # Check for wins
        wins = game.check_win(states, players)
        values_term, terminated = game.get_value_and_terminated(states, actions, players)
        
        # Update active games
        newly_terminated = terminated & active_games
        active_games = active_games & ~terminated
        
        # Assign outcomes to memory
        # (This is simplified - proper implementation would backfill rewards)
        
        # Switch players
        players = -players
    
    return memory

def train_epoch(model, optimizer, memory, batch_size):
    """Train model on collected experience"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Convert memory to tensors
    states = torch.stack([m['state'] for m in memory])
    policies = torch.stack([m['policy'] for m in memory])
    # Simplified: use random values for now
    values = torch.randn(len(memory), 1)
    
    dataset_size = len(states)
    indices = torch.randperm(dataset_size)
    
    for i in range(0, dataset_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        
        batch_states = states[batch_indices].to(model.device)
        batch_policies = policies[batch_indices].to(model.device)
        batch_values = values[batch_indices].to(model.device)
        
        # Forward pass
        pred_policy, pred_value = model(batch_states)
        
        # Loss
        policy_loss = -(batch_policies * torch.log_softmax(pred_policy, dim=1)).sum(dim=1).mean()
        value_loss = ((pred_value - batch_values) ** 2).mean()
        loss = policy_loss + value_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def main():
    print("="*70)
    print("GPU Training - 1 Hour Test")
    print("="*70)
    print(f"Target: ~200W GPU usage (~80%)")
    print(f"Duration: ~1 hour (6 iterations)")
    print()
    
    device = torch.device('cuda')
    
    # Create game
    game = GomokuGPU(board_size=9, device=device)
    
    # Create model
    class DummyGame:
        row_count = 9
        column_count = 9
        action_size = 81
    
    print("Creating model...")
    print(f"  - {MODEL_CONFIG['num_res_blocks']} residual blocks")
    print(f"  - {MODEL_CONFIG['num_hidden']} hidden channels")
    
    model = ResNet(
        DummyGame(),
        num_res_blocks=MODEL_CONFIG['num_res_blocks'],
        num_hidden=MODEL_CONFIG['num_hidden']
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  - {params:,} parameters")
    print()
    
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'],
                          weight_decay=MODEL_CONFIG['weight_decay'])
    
    # Create checkpoint dir
    checkpoint_path = Path(PATHS['checkpoints'])
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training...")
    print(f"Monitor GPU: watch -n 1 nvidia-smi")
    print()
    
    history = {'iterations': [], 'loss': []}
    
    start_time = time.time()
    
    for iteration in range(TRAINING_CONFIG['num_iterations']):
        iter_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{TRAINING_CONFIG['num_iterations']}")
        print(f"{'='*60}")
        
        # Self-play
        print(f"Self-play ({TRAINING_CONFIG['num_self_play_iterations']} games)...")
        memory = simple_self_play_gpu(
            game, model,
            TRAINING_CONFIG['num_self_play_iterations'],
            MCTS_CONFIG['num_searches']
        )
        print(f"✅ Collected {len(memory)} samples")
        
        # Train
        print(f"Training ({TRAINING_CONFIG['num_epochs']} epochs)...")
        for epoch in range(TRAINING_CONFIG['num_epochs']):
            loss = train_epoch(model, optimizer, memory, TRAINING_CONFIG['batch_size'])
            if epoch == TRAINING_CONFIG['num_epochs'] - 1:
                print(f"📊 Final loss: {loss:.4f}")
        
        # Save
        torch.save(model.state_dict(), checkpoint_path / f"model_{iteration}.pt")
        
        # Timing
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        eta = (iter_time * (TRAINING_CONFIG['num_iterations'] - iteration - 1)) / 60
        
        print(f"⏱️  Iteration time: {iter_time/60:.1f}min | ETA: {eta:.0f}min")
        
        history['iterations'].append(iteration)
        history['loss'].append(loss)
        
        with open(checkpoint_path / 'history.json', 'w') as f:
            json.dump(history, f)
    
    total_time = time.time() - start_time
    print()
    print("="*70)
    print(f"✅ Training complete in {total_time/60:.1f} minutes!")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print("="*70)

if __name__ == "__main__":
    main()
