import torch
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from alpha_zero_light.game.tictactoe import TicTacToe
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS
from alpha_zero_light.training.trainer import AlphaZeroTrainer

def main():
    game = TicTacToe()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNet(game, num_res_blocks=4, num_hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_self_play_iterations': 10,
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25
    }
    
    mcts = MCTS(game, args, model)
    
    trainer = AlphaZeroTrainer(model, optimizer, game, args, mcts)
    trainer.learn()

if __name__ == "__main__":
    main()
