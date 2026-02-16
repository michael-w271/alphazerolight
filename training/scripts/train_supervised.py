#!/usr/bin/env python3
"""
Train WDL ResNet on solver-labeled positions (supervised learning).

Fine-tunes AlphaZero model_195 on perfect solver labels for Stockfish-like evaluation.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from alpha_zero_light.game.connect_four import ConnectFour
from alpha_zero_light.model.wdl_network import WDLResNet
from alpha_zero_light.data.supervised_dataset import create_dataloaders


def train_wdl_model(
    dataset_path='data/selfplay_42k_labeled.npz',
    checkpoint_path='checkpoints/connect4/model_195.pt',
    output_dir='checkpoints/connect4',
    num_epochs=100,
    batch_size=256,
    learning_rate=5e-5,
    val_split=0.2,
    device='cuda',
    early_stopping_patience=10,
    weight_decay=5e-4,
    gradient_clip=1.0
):
    """Train WDL model on solver labels with early stopping and regularization."""
    
    print("="*70)
    print(" "*20 + "WDL SUPERVISED TRAINING")
    print("="*70)
    
    # Setup
    game = ConnectFour()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model from AlphaZero checkpoint
    print(f"\nLoading model from {checkpoint_path}...")
    model = WDLResNet.from_alphazero_checkpoint(checkpoint_path, game, device)
    
    # Create dataloaders
    print(f"\nLoading dataset from {dataset_path}...")
    train_loader, val_loader = create_dataloaders(
        dataset_path,
        batch_size=batch_size,
        val_split=val_split
    )
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Training loop
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}\n")
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            states = batch['state'].to(device)
            wdl_labels = batch['wdl_label'].to(device)
            
            optimizer.zero_grad()
            
            policy, wdl_logits = model(states)
            loss = criterion(wdl_logits, wdl_labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * states.size(0)
            _, predicted = torch.max(wdl_logits, 1)
            train_correct += (predicted == wdl_labels).sum().item()
            train_total += states.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.3f}'
            })
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                wdl_labels = batch['wdl_label'].to(device)
                
                policy, wdl_logits = model(states)
                loss = criterion(wdl_logits, wdl_labels)
                
                val_loss += loss.item() * states.size(0)
                _, predicted = torch.max(wdl_logits, 1)
                val_correct += (predicted == wdl_labels).sum().item()
                val_total += states.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_path = Path(output_dir) / 'model_wdl_best.pt'
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            print(f"  Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_file = Path(output_dir) / f'model_wdl_epoch_{epoch+1}.pt'
            torch.save(model.state_dict(), checkpoint_file)
            print(f"  ✓ Saved checkpoint: {checkpoint_file.name}")
    
    # Save final model
    final_path = Path(output_dir) / 'model_wdl_final.pt'
    torch.save(model.state_dict(), final_path)
    
    # Save training history
    history_path = Path(output_dir) / 'wdl_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Best validation accuracy: {best_val_acc:.4f}")
    print(f"✓ Saved final model: {final_path}")
    print(f"✓ Saved training history: {history_path}")
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train WDL model on solver labels')
    parser.add_argument('--dataset', default='data/selfplay_42k_labeled.npz',
                       help='Path to labeled dataset')
    parser.add_argument('--checkpoint', default='checkpoints/connect4/model_195.pt',
                       help='AlphaZero checkpoint to fine-tune')
    parser.add_argument('--output-dir', default='checkpoints/connect4',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split fraction')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_wdl_model(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        device=args.device,
        early_stopping_patience=args.early_stopping,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip
    )
