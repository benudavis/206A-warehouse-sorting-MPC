#!/usr/bin/env python3
"""One-command setup for Neural Network Forward Kinematics."""

import sys
import subprocess
from pathlib import Path
import argparse


def run_cmd(cmd, desc):
    """Run command and handle errors."""
    print(f"\n{'='*70}\n{desc}\n{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {desc}")
        return False
    
    print(f"\n✓ Completed: {desc}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup NN FK for obstacle avoidance")
    parser.add_argument("--samples", type=int, default=200_000, help="Number of samples")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick mode (50k/20 epochs)")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--skip-testing", action="store_true", help="Skip testing")
    args = parser.parse_args()
    
    if args.quick:
        args.samples, args.epochs, args.lr = 50_000, 50, 5e-4
        print("🚀 Quick mode: 50k samples, 50 epochs")
    
    root = Path(__file__).parent.parent.parent
    dataset = root / "data/models/ur5e_fk_dataset.npz"
    weights = root / "data/models/ur5e_fk_nn.npz"
    
    print(f"{'='*70}\nNN FK SETUP\n{'='*70}")
    print(f"Dataset: {dataset}\nWeights: {weights}")
    print(f"Samples: {args.samples:,}, Epochs: {args.epochs}\n{'='*70}")
    
    # Step 1: Generate dataset
    if not args.skip_dataset:
        if not run_cmd([
            "uv", "run", "python", "tools/nn_fk/generate_dataset.py",
            "--samples", str(args.samples), "--output", str(dataset)
        ], "STEP 1: Generate Dataset"):
            return 1
    else:
        print("\n⏭️  Skipping dataset generation")
        if not dataset.exists():
            print(f"❌ Dataset not found at {dataset}")
            return 1
    
    # Step 2: Train
    if not args.skip_training:
        if not run_cmd([
            "uv", "run", "python", "tools/nn_fk/train.py",
            "--dataset", str(dataset), "--output", str(weights), 
            "--epochs", str(args.epochs), "--lr", str(args.lr)
        ], "STEP 2: Train Network"):
            return 1
    else:
        print("\n⏭️  Skipping training")
        if not weights.exists():
            print(f"❌ Weights not found at {weights}")
            return 1
    
    # Step 3: Test
    if not args.skip_testing:
        run_cmd([
            "uv", "run", "python", "src/control/nn_fk_casadi.py",
            "--weights", str(weights), "--samples", "1000"
        ], "STEP 3: Test Accuracy")
    else:
        print("\n⏭️  Skipping testing")
    
    print(f"\n{'='*70}\n✓ SETUP COMPLETE!\n{'='*70}")
    print(f"Dataset: {dataset}\nWeights: {weights}")
    print("\nRun demo: uv run mjpython scripts/demo_sorting.py\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

