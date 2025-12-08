#!/usr/bin/env python3
"""
Optimize FK parameters (H, P) to match MuJoCo arm_hand_pinch site.

This script:
1. Generates random joint configurations
2. Computes ground truth positions from MuJoCo
3. Optimizes H (joint axes) and P (link offsets) to minimize FK error
4. Outputs optimized parameters to update forward_kinematics.py
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.optimize import minimize
import casadi as ca

# Build MuJoCo model
models_dir = Path(__file__).parent.parent / "sim" / "models"
scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))
arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")
model = scene.compile()
data = mujoco.MjData(model)

pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch")

# Generate training data
print("Generating calibration data...")
np.random.seed(42)
n_samples = 200
q_samples = []
x_samples = []

for i in range(n_samples):
    q = np.random.uniform(-np.pi, np.pi, 6)
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    x = data.site_xpos[pinch_site_id].copy()
    
    q_samples.append(q)
    x_samples.append(x)

q_samples = np.array(q_samples)
x_samples = np.array(x_samples)

print(f"Generated {n_samples} samples")
print(f"X range: [{x_samples.min(axis=0)}, {x_samples.max(axis=0)}]")

# Initial parameters (current values from forward_kinematics.py)
# H: joint axes (18 params: 6 joints × 3 components, but normalized so effectively 6 × 2 free params)
# P: link offsets (21 params: 7 links × 3 components)
# For simplicity, we'll optimize P only, keeping H fixed (axes are usually correct)

# Current P values
P_init = np.array([
    [0.0, 0.0, 0.1625],      # p01
    [0.0, 0.0, 0.0],         # p12
    [-0.425, 0.0, 0.0],      # p23
    [-0.3922, 0.0, 0.0],     # p34
    [0.0, -0.1333, -0.0997], # p45
    [0.0, 0.0, 0.0],         # p56
    [0.0, -0.0996, 0.0],     # p6T
])

# Fixed values
H_fixed = np.array([
    [0, 0, 1],      # ez
    [0, -1, 0],     # -ey
    [0, -1, 0],     # -ey
    [0, -1, 0],     # -ey
    [0, 0, -1],     # -ez
    [0, -1, 0],     # -ey
]).T  # 3x6

R_6T_fixed = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
])

TOOL0_TO_PINCH = np.array([0.0, -0.0493, 0.03308])
BASE_TO_WORLD = np.array([0.0002, -0.12382, 0.4495])

def compute_fk_numpy(q, P):
    """Compute FK with given P parameters."""
    # Rodrigues rotation
    def rot_axis(w, theta):
        K = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        I = np.eye(3)
        return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    R = np.eye(3)
    p = P[0]  # p01
    
    for i in range(6):
        R = R @ rot_axis(H_fixed[:, i], q[i])
        if i < 5:
            p = p + R @ P[i + 1]
    
    # Tool frame
    p_0T = p + R @ P[6]
    R_0T = R @ R_6T_fixed
    
    # Pinch offset
    p_pinch_base = p_0T + R_0T @ TOOL0_TO_PINCH
    
    # World frame
    p_pinch_world = p_pinch_base + BASE_TO_WORLD
    
    return p_pinch_world

def objective(params):
    """Compute RMS error over all samples."""
    P = params.reshape(7, 3)
    
    errors = []
    for q, x_true in zip(q_samples, x_samples):
        x_pred = compute_fk_numpy(q, P)
        error = np.linalg.norm(x_pred - x_true)
        errors.append(error)
    
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    return rms_error

# Optimize
print("\nOptimizing P parameters...")
print(f"Initial RMS error: {objective(P_init.flatten()):.6f} m")

result = minimize(
    objective,
    P_init.flatten(),
    method='L-BFGS-B',
    options={'maxiter': 1000, 'disp': True}
)

P_opt = result.x.reshape(7, 3)

print(f"\nFinal RMS error: {result.fun:.6f} m")
print(f"Optimization success: {result.success}")

# Test on all samples
print("\n" + "=" * 70)
print("Testing optimized parameters on all samples...")
errors = []
for q, x_true in zip(q_samples, x_samples):
    x_pred = compute_fk_numpy(q, P_opt)
    error = np.linalg.norm(x_pred - x_true)
    errors.append(error)

errors = np.array(errors)
print(f"Mean error:   {np.mean(errors)*1000:.3f} mm")
print(f"Median error: {np.median(errors)*1000:.3f} mm")
print(f"Max error:    {np.max(errors)*1000:.3f} mm")
print(f"Min error:    {np.min(errors)*1000:.3f} mm")
print(f"Std dev:      {np.std(errors)*1000:.3f} mm")

print("\n" + "=" * 70)
print("Optimized P parameters (copy to forward_kinematics.py):")
print("=" * 70)
print("\nP = ca.SX(3, 7)")
for i, name in enumerate(['p01', 'p12', 'p23', 'p34', 'p45', 'p56', 'p6T']):
    print(f"P[:, {i}] = ca.SX({P_opt[i].tolist()})  # {name}")

print("\n" + "=" * 70)
print("Parameter changes:")
print("=" * 70)
for i, name in enumerate(['p01', 'p12', 'p23', 'p34', 'p45', 'p56', 'p6T']):
    diff = P_opt[i] - P_init[i]
    if np.linalg.norm(diff) > 0.0001:
        print(f"{name}: {P_init[i]} -> {P_opt[i]}")
        print(f"       Change: {diff}")
