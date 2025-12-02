#!/usr/bin/env python3
"""Generate forward kinematics dataset from MuJoCo for neural network training."""

import sys
from pathlib import Path
import numpy as np
import mujoco

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODELS_DIR = Path(__file__).parent.parent.parent / "sim" / "models"
EE_SITE = "arm_hand_pinch"


def build_world():
    """Build minimal world with just the robot arm."""
    scene = mujoco.MjSpec.from_file(str(MODELS_DIR / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robotiq_2f85" / "2f85.xml"))

    arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
    scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")

    model = scene.compile()
    data = mujoco.MjData(model)
    return model, data


def compute_joint_limits(model, n_joints=6):
    """Extract realistic joint limits from the model."""
    q_lower = np.empty(n_joints, dtype=np.float32)
    q_upper = np.empty(n_joints, dtype=np.float32)
    
    for j in range(n_joints):
        jnt_range = model.jnt_range[j]
        lo, hi = jnt_range[0], jnt_range[1]
        
        # If no range specified, use reasonable defaults
        if lo == 0.0 and hi == 0.0:
            lo, hi = -np.pi, np.pi  # ±180° instead of ±360°
        
        q_lower[j] = lo
        q_upper[j] = hi
    
    return q_lower, q_upper


def main(
    n_samples=200_000,
    out_path="data/models/ur5e_fk_dataset.npz",
    seed=0,
):
    """
    Generate FK dataset by sampling random joint configurations.
    
    Args:
        n_samples: Number of samples to generate
        out_path: Output path for the dataset
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("FK DATASET GENERATION")
    print("=" * 70)
    print(f"Generating {n_samples:,} samples...")
    print(f"Output: {out_path}")
    print("=" * 70)
    
    np.random.seed(seed)
    model, data = build_world()
    
    n_joints = 6
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    
    # Get realistic joint limits from model
    q_lower, q_upper = compute_joint_limits(model, n_joints)
    
    print("\nJoint limits from model:")
    for j in range(n_joints):
        print(f"  Joint {j}: [{q_lower[j]:+.3f}, {q_upper[j]:+.3f}] rad ({np.degrees(q_lower[j]):+.1f}°, {np.degrees(q_upper[j]):+.1f}°)")
    
    qs = np.zeros((n_samples, n_joints), dtype=np.float32)
    ps = np.zeros((n_samples, 3), dtype=np.float32)
    
    print("\nSampling joint configurations and computing FK...")
    for i in range(n_samples):
        # Uniform random sampling in joint space
        q = q_lower + (q_upper - q_lower) * np.random.rand(n_joints)
        data.qpos[:n_joints] = q
        
        # Compute forward kinematics
        mujoco.mj_forward(model, data)
        p = data.site_xpos[ee_site_id].copy()
        
        qs[i] = q
        ps[i] = p
        
        if (i + 1) % 10_000 == 0:
            print(f"  Progress: {i+1:,}/{n_samples:,} samples ({100*(i+1)/n_samples:.1f}%)")
    
    # Compute and print statistics
    print("\nDataset statistics:")
    print(f"  Joint angles (q):")
    print(f"    Mean: {qs.mean(axis=0)}")
    print(f"    Std:  {qs.std(axis=0)}")
    print(f"  EE positions (p):")
    print(f"    Mean: {ps.mean(axis=0)}")
    print(f"    Std:  {ps.std(axis=0)}")
    print(f"    Min:  {ps.min(axis=0)}")
    print(f"    Max:  {ps.max(axis=0)}")
    
    # Save dataset
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, q=qs, p=ps)
    
    print(f"\n✓ Saved dataset to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate FK dataset from MuJoCo")
    parser.add_argument("--samples", type=int, default=200_000, help="Number of samples")
    parser.add_argument("--output", default="data/models/ur5e_fk_dataset.npz", help="Output path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    
    main(n_samples=args.samples, out_path=args.output, seed=args.seed)

