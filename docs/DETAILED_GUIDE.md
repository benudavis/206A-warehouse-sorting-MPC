# Detailed Project Guide

This repository contains a single end-to-end demonstration: a UR5e arm picks three
cubes from the table and places them onto an elevated shelf from smallest to
largest. The sections below describe how the demo works and how to extend or debug
it.

---

## 1. Scene & Task

- **Table:** Base surface that holds the objects at `z = 0.52 m`
- **Shelf:** Box geometry positioned at `[0.60, -0.25, 0.70]` (≈28 cm above table)
- **Objects:** Three cubes with half-extent sizes `[0.02, 0.025, 0.03]` meters
- **Goal:** Sort cubes by size (small → medium → large) from left to right on the
  shelf positions `[0.52, 0.60, 0.68]`

All geometry is defined directly in `scripts/demo_sorting.py` using MuJoCo's API.

---

## 2. Control Pipeline (`scripts/demo_sorting.py`)

1. **Inverse Kinematics** (`src/control/inverse_kinematics.py`)
   - Targets the end-effector site `arm_hand_pinch`
   - First solves for a pose 10 cm above the cube, then 3 cm above the surface
   - Uses a tool-down quaternion `[0, 1, 0, 0]`

2. **Model Predictive Control** (`src/control/mpc_controller.py`)
   - Drives the 6 UR5e joints to the IK solution
   - Horizon: 30 steps, dt: 0.01 s, position weight: 500, terminal weight: 1000
   - If the solver fails, the code falls back to simple position interpolation

3. **Manual Attachment for Grasping**
   - When the EE is within 8 cm of the cube, the script records the EE→object
     offset
   - While the gripper is "closed", the object's free joint position follows the
     end-effector (simulating a perfect friction grasp)

4. **Shelf Placement**
   - After lifting ~0.9 radians in shoulder joint, the object is carried to the
     shelf IK target and released
   - A short settle period lets physics place the cube on the platform

---

## 3. Diagnostics & Logging

The diagnostic utilities in `src/diagnostics/` are optional but useful when the
motion needs debugging.

### Quick Usage Example

```python
from src.diagnostics import DiagnosticLogger

logger = DiagnosticLogger(model, data, site_name="arm_hand_pinch")
logger.add_tracked_object("red_small", mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_small"))

# Inside control loop
logger.log_state("red_small", phase="approach", attempt=0)

# After run
logger.generate_report(output_dir="data/diagnostics")
```

**Report contents:** 12-subplot figure (3D trajectory, distances, gripper, joint
traces, lift tests), JSON/NPZ raw logs, and text metrics (IK/MPC convergence,
lift success, distance statistics).

---

## 4. Key Parameters

| Component | Variable | Default | Notes |
|-----------|----------|---------|-------|
| MPC horizon | `horizon` | 30 | Increase for smoother but slower plans |
| MPC dt | `dt` | 0.01 s | Keep stable with Mujoco timestep (`0.0001`) |
| IK tolerance | `tolerance` | 0.02 m | Relax if IK struggles, tighten for precision |
| Grasp distance | `0.08 m` | Determines whether the object attaches |
| Shelf targets | `shelf_positions` | `[0.52, 0.60, 0.68]` | Left → right placement |

---

## 5. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Cube not picked up | EE too far before closing | Adjust approach tolerances or decrease grasp distance threshold |
| Arm oscillates near target | MPC not converging | Increase horizon or terminal weight; fall back to interpolation block |
| Object falls during move | Attachment offset not captured | Ensure `distance < 0.08` before closing; optionally raise threshold |
| Arm clips shelf | Shelf target too close | Modify `shelf_positions` or increase `target_above` height |
| Solver errors (CasADi) | Poor initial guess | Use last joint state as warm start or reduce horizon |

---

## 6. Useful Entry Points

| File | Purpose |
|------|---------|
| `scripts/demo_sorting.py` | Main demo script (scene + control pipeline) |
| `src/control/mpc_controller.py` | MPC implementation |
| `src/control/inverse_kinematics.py` | IK solver |
| `src/perception/sim_state.py` | Extracts robot state for MPC |
| `src/diagnostics/logger.py` | Core logging utilities |

---

## 7. Extending the Demo

- Adjust cube positions or add new shelf slots by editing the arrays near the top
  of `demo_sorting.py`
- Change sorting order (e.g., largest → smallest) by modifying the sort key
- Swap manual attachment with a physics-based grasp by removing the offset logic
  and tuning MuJoCo friction coefficients
- Log additional objects by registering them with `DiagnosticLogger`

---

Enjoy experimenting with the shelf sorting demo! The code is intentionally
compact so that the entire control loop fits in a single script and can be read
from top to bottom.

