#!/usr/bin/env python3
"""
Train neural network to imitate MPC controller
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.learning.mpc_imitator import MPCImitator


def load_data(data_path):
    """Load training data."""
    data = np.load(data_path)
    observations = data['observations']
    actions = data['actions']
    print(f"Loaded data from {data_path}")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    return observations, actions


def plot_training_history(history, save_path=None):
    """Plot training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training History')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MPC imitator')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (.npz)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 128, 64], 
                       help='Hidden layer sizes')
    parser.add_argument('--save-dir', type=str, default='data/models', help='Model save directory')
    
    args = parser.parse_args()
    
    # Load data
    observations, actions = load_data(args.data)
    
    # Initialize model
    print("\nInitializing model...")
    print(f"  Hidden layers: {args.hidden}")
    imitator = MPCImitator(
        obs_dim=observations.shape[1],
        action_dim=actions.shape[1],
        hidden_dims=args.hidden
    )
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = imitator.train(
        observations,
        actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Save model
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = save_path / f'mpc_imitator_{timestamp}.pth'
    imitator.save(model_path)
    
    # Plot training history
    plot_path = save_path / f'training_history_{timestamp}.png'
    plot_training_history(history, save_path=plot_path)
    
    # Print final stats
    print(f"\n✅ Training complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

