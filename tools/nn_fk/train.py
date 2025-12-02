#!/usr/bin/env python3
"""Train neural network to approximate forward kinematics."""

import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def make_features(q: np.ndarray) -> np.ndarray:
    """
    Convert joint angles to sin/cos features for better periodicity.
    
    Args:
        q: Joint angles (N, 6)
    
    Returns:
        features: [sin(q), cos(q)] (N, 12)
    """
    return np.concatenate([np.sin(q), np.cos(q)], axis=1)


class FKNet(nn.Module):
    """MLP for approximating forward kinematics: phi(q) -> p."""
    
    def __init__(self, in_dim=12, hidden=128, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def main(
    dataset_path="data/models/ur5e_fk_dataset.npz",
    out_path="data/models/ur5e_fk_nn.npz",
    batch_size=1024,
    lr=1e-3,
    n_epochs=50,
    hidden_dim=64,
    device=None,
):
    """
    Train neural network FK approximator.
    
    Args:
        dataset_path: Path to generated FK dataset
        out_path: Output path for trained weights
        batch_size: Training batch size
        lr: Learning rate
        n_epochs: Number of training epochs
        hidden_dim: Hidden layer dimension
        device: Device to train on (auto-detect if None)
    """
    print("=" * 70)
    print("NEURAL NETWORK FK TRAINING")
    print("=" * 70)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    q = data["q"].astype(np.float32)
    p = data["p"].astype(np.float32)
    print(f"  Loaded {q.shape[0]:,} samples")
    print(f"  Input dim: {q.shape[1]} (joint angles)")
    print(f"  Output dim: {p.shape[1]} (EE position)")
    
    # Train/val split
    n = q.shape[0]
    n_train = int(0.9 * n)
    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    
    q_train, p_train = q[train_idx], p[train_idx]
    q_val, p_val = q[val_idx], p[val_idx]
    
    print(f"\nTrain/val split:")
    print(f"  Train: {len(train_idx):,} samples")
    print(f"  Val:   {len(val_idx):,} samples")
    
    # Convert to sin/cos features for better periodicity
    print(f"\nConverting to sin/cos features (6 → 12)...")
    phi_train = make_features(q_train)  # (N, 12)
    phi_val = make_features(q_val)      # (N, 12)
    print(f"  Feature dim: {phi_train.shape[1]}")
    
    # Compute normalization statistics (on training features only)
    mu_phi = phi_train.mean(axis=0)
    sigma_phi = phi_train.std(axis=0) + 1e-8
    mu_p = p_train.mean(axis=0)
    sigma_p = p_train.std(axis=0) + 1e-8
    
    print(f"\nNormalization statistics:")
    print(f"  Features phi(q) = [sin(q), cos(q)]:")
    print(f"    Mean (μ_φ): {mu_phi}")
    print(f"    Std  (σ_φ): {sigma_phi}")
    print(f"  Positions (p):")
    print(f"    Mean (μ_p): {mu_p}")
    print(f"    Std  (σ_p): {sigma_p}")
    print(f"    Range: [{p_train.min(axis=0)} to {p_train.max(axis=0)}]")
    
    # Apply normalization
    phi_train_n = (phi_train - mu_phi) / sigma_phi
    phi_val_n = (phi_val - mu_phi) / sigma_phi
    p_train_n = (p_train - mu_p) / sigma_p
    p_val_n = (p_val - mu_p) / sigma_p
    
    # Create datasets and loaders
    train_ds = TensorDataset(
        torch.from_numpy(phi_train_n),
        torch.from_numpy(p_train_n),
    )
    val_ds = TensorDataset(
        torch.from_numpy(phi_val_n),
        torch.from_numpy(p_val_n),
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Create model (12-D input for sin/cos features)
    model = FKNet(in_dim=12, hidden=hidden_dim, out_dim=3).to(device)
    print(f"\nModel architecture:")
    print(f"  {model}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 70)
    
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        running = 0.0
        for q_b, p_b in train_loader:
            q_b = q_b.to(device)
            p_b = p_b.to(device)
            
            pred = model(q_b)
            loss = loss_fn(pred, p_b)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            running += loss.item() * q_b.size(0)
        
        train_loss = running / len(train_ds)
        
        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for q_b, p_b in val_loader:
                q_b = q_b.to(device)
                p_b = p_b.to(device)
                pred = model(q_b)
                loss = loss_fn(pred, p_b)
                running_val += loss.item() * q_b.size(0)
        
        val_loss = running_val / len(val_ds)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch:03d}/{n_epochs}: train_loss={train_loss:.6e}, val_loss={val_loss:.6e}", end="")
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            print(" ← best")
        else:
            print()
    
    print("-" * 70)
    print(f"✓ Training complete. Best val loss: {best_val:.6e} (epoch {best_epoch})")
    
    # Load best state and evaluate
    model.load_state_dict(best_state)
    model.eval()
    
    # Compute actual position error (denormalized)
    with torch.no_grad():
        phi_val_t = torch.from_numpy(phi_val_n).to(device)
        p_val_pred_n = model(phi_val_t).cpu().numpy()
    
    # Denormalize predictions
    p_val_pred = p_val_pred_n * sigma_p + mu_p
    
    # Debug: Check if denormalization is working
    print(f"\nDebug - Validation predictions:")
    print(f"  True position range: [{p_val.min(axis=0)} to {p_val.max(axis=0)}]")
    print(f"  Pred position range: [{p_val_pred.min(axis=0)} to {p_val_pred.max(axis=0)}]")
    print(f"  Normalized pred range: [{p_val_pred_n.min(axis=0)} to {p_val_pred_n.max(axis=0)}]")
    
    # Compute errors
    errors = np.linalg.norm(p_val_pred - p_val, axis=1)
    errors_mm = errors * 1000  # Convert to mm
    print(f"\nPosition error statistics (meters):")
    print(f"  Mean:   {errors.mean():.6f} m ({errors_mm.mean():.3f} mm)")
    print(f"  Median: {np.median(errors):.6f} m ({np.median(errors_mm):.3f} mm)")
    print(f"  Max:    {errors.max():.6f} m ({errors_mm.max():.3f} mm)")
    print(f"  95th percentile: {np.percentile(errors, 95):.6f} m ({np.percentile(errors_mm, 95):.3f} mm)")
    
    # Check if error is acceptable
    if errors_mm.mean() > 10.0:
        print("\n⚠️  WARNING: Error is very high (>10mm)!")
        print("   This suggests a training problem. Try:")
        print("   1. Lower learning rate: --lr 1e-4")
        print("   2. More epochs: --epochs 100")
        print("   3. Check dataset quality")
        print("   4. Try larger network: --hidden 128")
    
    # Extract weights and biases (3 hidden layers now)
    W1 = model.net[0].weight.detach().cpu().numpy()
    b1 = model.net[0].bias.detach().cpu().numpy()
    W2 = model.net[2].weight.detach().cpu().numpy()
    b2 = model.net[2].bias.detach().cpu().numpy()
    W3 = model.net[4].weight.detach().cpu().numpy()
    b3 = model.net[4].bias.detach().cpu().numpy()
    W4 = model.net[6].weight.detach().cpu().numpy()
    b4 = model.net[6].bias.detach().cpu().numpy()
    
    # Save everything (use mu_q/sigma_q for feature stats, not raw angles)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        out_path,
        W1=W1, b1=b1,
        W2=W2, b2=b2,
        W3=W3, b3=b3,
        W4=W4, b4=b4,
        mu_q=mu_phi, sigma_q=sigma_phi,  # These are feature stats now!
        mu_p=mu_p, sigma_p=sigma_p,
    )
    
    print(f"\n✓ Saved trained FK NN to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.2f} KB")
    
    # Plot training curves
    print("\nGenerating training curve plot...")
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves (log scale)
        ax1.semilogy(range(1, n_epochs + 1), train_losses, label='Train Loss', linewidth=2)
        ax1.semilogy(range(1, n_epochs + 1), val_losses, label='Val Loss', linewidth=2)
        ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss (log scale)')
        ax1.set_title('Training Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution (mm)
        ax2.hist(errors_mm, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(errors_mm.mean(), color='red', linestyle='--', label=f'Mean: {errors_mm.mean():.3f}mm')
        ax2.set_xlabel('Position Error (mm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Error Distribution ({len(errors)} validation samples)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = out_path.parent / "nn_fk_training.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved training plot to {plot_path}")
        
    except ImportError:
        print("  (matplotlib not available in headless mode, skipping plot)")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train NN FK approximator")
    parser.add_argument("--dataset", default="data/models/ur5e_fk_dataset.npz", help="Input dataset path")
    parser.add_argument("--output", default="data/models/ur5e_fk_nn.npz", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer dimension")
    args = parser.parse_args()
    
    main(
        dataset_path=args.dataset,
        out_path=args.output,
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        hidden_dim=args.hidden,
    )

