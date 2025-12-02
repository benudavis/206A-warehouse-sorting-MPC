"""
CasADi wrapper for neural network forward kinematics.
Embeds a trained PyTorch MLP into CasADi for use in MPC optimization.
"""

import numpy as np
import casadi as ca
from pathlib import Path


def build_nn_fk_function(
    weights_path: str,
    n_joints: int = 6,
) -> ca.Function:
    """
    Build a CasADi symbolic function nn_fk(q) -> p approximating FK.
    
    This function loads a trained neural network and creates a CasADi
    symbolic expression that can be used within optimization problems.
    
    Args:
        weights_path: Path to .npz file with network weights and normalization.
        n_joints: Number of joints (6 for UR5e).
    
    Returns:
        CasADi Function f(q) = p, with q in R^6, p in R^3
        
    Raises:
        FileNotFoundError: If weights file doesn't exist
        KeyError: If required keys are missing from weights file
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"NN FK weights not found at {weights_path}. "
            "Run 'python scripts/generate_fk_dataset.py' and "
            "'python scripts/train_nn_fk.py' first."
        )
    
    # Load parameters
    params = np.load(str(weights_path))
    
    # Extract weights and biases (3 hidden layers)
    W1 = ca.DM(params["W1"])
    b1 = ca.DM(params["b1"])
    W2 = ca.DM(params["W2"])
    b2 = ca.DM(params["b2"])
    W3 = ca.DM(params["W3"])
    b3 = ca.DM(params["b3"])
    W4 = ca.DM(params["W4"])
    b4 = ca.DM(params["b4"])
    
    # Extract normalization statistics (for features, not raw angles)
    mu_phi = ca.DM(params["mu_q"])     # Actually feature mean
    sigma_phi = ca.DM(params["sigma_q"])  # Actually feature std
    mu_p = ca.DM(params["mu_p"])
    sigma_p = ca.DM(params["sigma_p"])
    
    # Create symbolic input
    q = ca.SX.sym("q", n_joints)
    
    # Build sin/cos features (same as training)
    sin_q = ca.sin(q)
    cos_q = ca.cos(q)
    phi = ca.vertcat(sin_q, cos_q)  # 12x1 vector
    
    # Normalize features
    phi_norm = (phi - mu_phi) / sigma_phi
    
    # Forward pass through network (3 hidden layers)
    # Layer 1: h1 = tanh(W1 @ phi_norm + b1)
    h1 = ca.tanh(W1 @ phi_norm + b1)
    
    # Layer 2: h2 = tanh(W2 @ h1 + b2)
    h2 = ca.tanh(W2 @ h1 + b2)
    
    # Layer 3: h3 = tanh(W3 @ h2 + b3)
    h3 = ca.tanh(W3 @ h2 + b3)
    
    # Output layer: p_norm = W4 @ h3 + b4
    p_norm = W4 @ h3 + b4
    
    # Denormalize output
    p = sigma_p * p_norm + mu_p
    
    # Create CasADi function
    nn_fk = ca.Function("nn_fk", [q], [p], ["q"], ["p"])
    
    return nn_fk


def test_nn_fk(weights_path: str, model=None, data=None, site_id=None, n_test=100):
    """
    Test the CasADi NN FK against ground truth MuJoCo FK.
    
    Args:
        weights_path: Path to NN weights
        model: MuJoCo model (if None, will build one)
        data: MuJoCo data (if None, will create one)
        site_id: EE site ID (if None, will look it up)
        n_test: Number of test samples
        
    Returns:
        dict with error statistics
    """
    import mujoco
    import sys
    from pathlib import Path
    
    # Build NN FK
    nn_fk = build_nn_fk_function(weights_path)
    
    # Build MuJoCo model if needed
    if model is None or data is None:
        MODELS_DIR = Path(__file__).parent.parent.parent / "sim" / "models"
        scene = mujoco.MjSpec.from_file(str(MODELS_DIR / "scene.xml"))
        arm_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml"))
        hand_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robotiq_2f85" / "2f85.xml"))
        arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
        scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")
        model = scene.compile()
        data = mujoco.MjData(model)
    
    if site_id is None:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")
    
    # Test on random configurations
    q_lower = np.array([-2*np.pi] * 6)
    q_upper = np.array([2*np.pi] * 6)
    
    errors = []
    for _ in range(n_test):
        q = q_lower + (q_upper - q_lower) * np.random.rand(6)
        
        # MuJoCo FK (ground truth)
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        p_true = data.site_xpos[site_id].copy()
        
        # NN FK (approximation)
        p_pred = np.array(nn_fk(q)).flatten()
        
        # Error
        error = np.linalg.norm(p_pred - p_true)
        errors.append(error)
    
    errors = np.array(errors)
    
    stats = {
        'mean': errors.mean(),
        'median': np.median(errors),
        'max': errors.max(),
        'std': errors.std(),
        'p95': np.percentile(errors, 95),
        'p99': np.percentile(errors, 99),
    }
    
    return stats


if __name__ == "__main__":
    """Test the NN FK if run as a script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NN FK approximation")
    parser.add_argument("--weights", default="data/models/ur5e_fk_nn.npz", help="Path to NN weights")
    parser.add_argument("--samples", type=int, default=1000, help="Number of test samples")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TESTING NN FK APPROXIMATION")
    print("=" * 70)
    
    print(f"\nLoading NN from {args.weights}...")
    nn_fk = build_nn_fk_function(args.weights)
    print("✓ NN FK loaded successfully")
    
    print(f"\nTesting against MuJoCo FK on {args.samples} random configurations...")
    stats = test_nn_fk(args.weights, n_test=args.samples)
    
    print("\nError statistics (meters):")
    print(f"  Mean:   {stats['mean']:.6f} m ({stats['mean']*1000:.3f} mm)")
    print(f"  Median: {stats['median']:.6f} m ({stats['median']*1000:.3f} mm)")
    print(f"  Std:    {stats['std']:.6f} m ({stats['std']*1000:.3f} mm)")
    print(f"  Max:    {stats['max']:.6f} m ({stats['max']*1000:.3f} mm)")
    print(f"  95th percentile: {stats['p95']:.6f} m ({stats['p95']*1000:.3f} mm)")
    print(f"  99th percentile: {stats['p99']:.6f} m ({stats['p99']*1000:.3f} mm)")
    
    print("\n" + "=" * 70)
    print("✓ Testing complete")
    print("=" * 70)

