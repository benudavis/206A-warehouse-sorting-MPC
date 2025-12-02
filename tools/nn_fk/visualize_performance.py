#!/usr/bin/env python3
"""Visualize NN FK performance with detailed plots."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.control.nn_fk_casadi import build_nn_fk_function

MODELS_DIR = Path(__file__).parent.parent.parent / "sim" / "models"
EE_SITE = "arm_hand_pinch"


def compute_joint_limits(model, n_joints=6):
    """Extract realistic joint limits from the model."""
    q_lower = np.empty(n_joints, dtype=np.float32)
    q_upper = np.empty(n_joints, dtype=np.float32)
    
    for j in range(n_joints):
        jnt_range = model.jnt_range[j]
        lo, hi = jnt_range[0], jnt_range[1]
        
        if lo == 0.0 and hi == 0.0:
            lo, hi = -np.pi, np.pi
        
        q_lower[j] = lo
        q_upper[j] = hi
    
    return q_lower, q_upper


def build_robot():
    """Build robot model."""
    scene = mujoco.MjSpec.from_file(str(MODELS_DIR / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robotiq_2f85" / "2f85.xml"))
    arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
    scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")
    model = scene.compile()
    data = mujoco.MjData(model)
    return model, data


def main(weights_path="data/models/ur5e_fk_nn.npz", n_test=1000, output_dir="data/diagnostics"):
    """
    Visualize NN FK performance with detailed analysis plots.
    
    Args:
        weights_path: Path to trained NN weights
        n_test: Number of test samples
        output_dir: Where to save plots
    """
    print("=" * 70)
    print("NN FK PERFORMANCE VISUALIZATION")
    print("=" * 70)
    
    # Check if weights exist
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"❌ Weights not found at {weights_path}")
        print("\nTrain first:")
        print("  uv run python tools/nn_fk/setup.py --quick")
        return 1
    
    # Load NN FK
    print(f"\nLoading NN FK from {weights_path}...")
    nn_fk = build_nn_fk_function(str(weights_path))
    print("✓ NN FK loaded")
    
    # Build robot
    print("Building robot model...")
    model, data = build_robot()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    print("✓ Robot model ready")
    
    # Generate test samples
    print(f"\nGenerating {n_test} test samples...")
    q_lower, q_upper = compute_joint_limits(model, n_joints=6)
    print("Using realistic joint limits from model")
    
    qs = []
    ps_true = []
    ps_pred = []
    
    for i in range(n_test):
        # Random joint configuration
        q = q_lower + (q_upper - q_lower) * np.random.rand(6)
        
        # Ground truth (MuJoCo FK)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        p_true = data.site_xpos[site_id].copy()
        
        # NN prediction
        p_pred = np.array(nn_fk(q)).flatten()
        
        qs.append(q)
        ps_true.append(p_true)
        ps_pred.append(p_pred)
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_test} samples...")
    
    qs = np.array(qs)
    ps_true = np.array(ps_true)
    ps_pred = np.array(ps_pred)
    
    # Compute errors
    errors = np.linalg.norm(ps_pred - ps_true, axis=1)
    errors_mm = errors * 1000  # Convert to mm
    
    print("\n" + "=" * 70)
    print("ACCURACY STATISTICS")
    print("=" * 70)
    print(f"Mean error:   {errors.mean():.6f} m ({errors_mm.mean():.3f} mm)")
    print(f"Median error: {np.median(errors):.6f} m ({np.median(errors_mm):.3f} mm)")
    print(f"Std error:    {errors.std():.6f} m ({errors_mm.std():.3f} mm)")
    print(f"Max error:    {errors.max():.6f} m ({errors_mm.max():.3f} mm)")
    print(f"95th %ile:    {np.percentile(errors, 95):.6f} m ({np.percentile(errors_mm, 95):.3f} mm)")
    print(f"99th %ile:    {np.percentile(errors, 99):.6f} m ({np.percentile(errors_mm, 99):.3f} mm)")
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Error histogram
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(errors_mm, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(errors_mm.mean(), color='red', linestyle='--', label=f'Mean: {errors_mm.mean():.3f}mm')
    ax1.axvline(np.median(errors_mm), color='green', linestyle='--', label=f'Median: {np.median(errors_mm):.3f}mm')
    ax1.set_xlabel('Position Error (mm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error CDF
    ax2 = plt.subplot(3, 3, 2)
    sorted_errors = np.sort(errors_mm)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cdf, linewidth=2)
    ax2.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95th percentile')
    ax2.axvline(np.percentile(errors_mm, 95), color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Position Error (mm)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error vs joint configuration (first joint)
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(qs[:, 0], errors_mm, alpha=0.3, s=10)
    ax3.set_xlabel('Joint 1 Angle (rad)')
    ax3.set_ylabel('Position Error (mm)')
    ax3.set_title('Error vs Joint 1')
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-axis errors (X)
    ax4 = plt.subplot(3, 3, 4)
    errors_x = np.abs(ps_pred[:, 0] - ps_true[:, 0]) * 1000
    ax4.hist(errors_x, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax4.set_xlabel('X Error (mm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'X-Axis Error (mean: {errors_x.mean():.3f}mm)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Per-axis errors (Y)
    ax5 = plt.subplot(3, 3, 5)
    errors_y = np.abs(ps_pred[:, 1] - ps_true[:, 1]) * 1000
    ax5.hist(errors_y, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.set_xlabel('Y Error (mm)')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Y-Axis Error (mean: {errors_y.mean():.3f}mm)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Per-axis errors (Z)
    ax6 = plt.subplot(3, 3, 6)
    errors_z = np.abs(ps_pred[:, 2] - ps_true[:, 2]) * 1000
    ax6.hist(errors_z, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax6.set_xlabel('Z Error (mm)')
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'Z-Axis Error (mean: {errors_z.mean():.3f}mm)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Prediction vs Ground Truth (X)
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(ps_true[:, 0], ps_pred[:, 0], alpha=0.3, s=10)
    ax7.plot([ps_true[:, 0].min(), ps_true[:, 0].max()],
             [ps_true[:, 0].min(), ps_true[:, 0].max()],
             'r--', label='Perfect prediction')
    ax7.set_xlabel('True X (m)')
    ax7.set_ylabel('Predicted X (m)')
    ax7.set_title('X Prediction Accuracy')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Prediction vs Ground Truth (Y)
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(ps_true[:, 1], ps_pred[:, 1], alpha=0.3, s=10)
    ax8.plot([ps_true[:, 1].min(), ps_true[:, 1].max()],
             [ps_true[:, 1].min(), ps_true[:, 1].max()],
             'r--', label='Perfect prediction')
    ax8.set_xlabel('True Y (m)')
    ax8.set_ylabel('Predicted Y (m)')
    ax8.set_title('Y Prediction Accuracy')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Prediction vs Ground Truth (Z)
    ax9 = plt.subplot(3, 3, 9)
    ax9.scatter(ps_true[:, 2], ps_pred[:, 2], alpha=0.3, s=10)
    ax9.plot([ps_true[:, 2].min(), ps_true[:, 2].max()],
             [ps_true[:, 2].min(), ps_true[:, 2].max()],
             'r--', label='Perfect prediction')
    ax9.set_xlabel('True Z (m)')
    ax9.set_ylabel('Predicted Z (m)')
    ax9.set_title('Z Prediction Accuracy')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'NN FK Performance Analysis ({n_test} test samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nn_fk_performance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    
    # Show plot
    plt.show()
    
    print("=" * 70)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize NN FK performance")
    parser.add_argument("--weights", default="data/models/ur5e_fk_nn.npz", help="NN weights path")
    parser.add_argument("--samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--output", default="data/diagnostics", help="Output directory for plots")
    args = parser.parse_args()
    
    sys.exit(main(weights_path=args.weights, n_test=args.samples, output_dir=args.output))

