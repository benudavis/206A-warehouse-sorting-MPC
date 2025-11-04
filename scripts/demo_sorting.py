#!/usr/bin/env python3
"""
Robust pick-and-place with NO reset between boxes.

Fixes:
- Correct test-lift check (compare object TOP before/after, not COM vs TOP)
- Slightly deeper settle on retries (up to 6 mm)
- Tightened loops to avoid long hangs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from src.control.mpc_controller import MPCController
from src.control.inverse_kinematics import IKSolver
from src.perception.sim_state import SimulationState

EE_SITE = "arm_hand_pinch"

script_dir = Path(__file__).parent.parent / "sim"
models_dir = script_dir / "models"

scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))

# Attach hand to arm, arm to scene anchor
arm_spec.site('attachment_site').attach_body(hand_spec.worldbody, "hand_", "")
scene.site('robot_site').attach_body(arm_spec.worldbody, "arm_", "")

# Create 3 cubes
positions = [
    [0.42, -0.38, 0.52],    # red_small
    [0.47, -0.405, 0.52],   # blue_medium
    [0.52, -0.43, 0.52],    # green_large
]
sizes = [0.02, 0.025, 0.03]                      # MuJoCo box "size" = half-extent
colors = [[1, 0.2, 0.2, 1], [0.2, 0.2, 1, 1], [0.2, 1, 0.2, 1]]
names  = ["red_small", "blue_medium", "green_large"]

for pos, size, color, name in zip(positions, sizes, colors, names):
    cube = scene.worldbody.add_body()
    cube.name = name
    cube.pos = pos
    g = cube.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [size] * 3
    g.rgba = color
    g.mass = 0.05
    g.friction = [1.0, 0.005, 0.0001]
    cube.add_freejoint()

model = scene.compile()
model.opt.timestep = 0.0001
data = mujoco.MjData(model)
data_fk = mujoco.MjData(model)

# Home once at startup (orientation facing workcell)
model.key_qpos[0][model.jnt('arm_shoulder_pan_joint').qposadr] += np.pi
model.key_ctrl[0][model.jnt('arm_shoulder_pan_joint').dofadr] += np.pi
mujoco.mj_resetDataKeyframe(model, data, 0)

print("="*70)
print("WAREHOUSE SORTING WITH ACTUAL PICK AND PLACE")
print("="*70)
print("\nUsing natural gripper friction (no manual constraints)")
print("="*70)

# Settle
for _ in range(300):
    mujoco.mj_step(model, data)

# Object lookup
objects = []
for i, name in enumerate(names):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos = data.xpos[bid].copy()
    objects.append({'id': bid, 'name': name, 'size': sizes[i], 'start_pos': pos.copy()})
    print(f"   {name}: {sizes[i]*1000:.0f}mm at {pos}")

# Init utils
ik = IKSolver(model, data, site_name=EE_SITE)
mpc = MPCController(n_joints=6, horizon=30, dt=0.01)
state = SimulationState(model, data, ee_site_name=EE_SITE)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
viewer = mujoco.viewer.launch_passive(model, data)

# Sort by size (smallest first)
sorted_objs = sorted(objects, key=lambda x: x['size'])
print(f"\n🎯 Sorting plan:")
for i, obj in enumerate(sorted_objs):
    print(f"   {i+1}. {obj['name']} → Bin {i+1}")

print(f"\n{'='*70}\nSTARTING SORT\n{'='*70}")
for _ in range(100):  # tiny settle (no reset)
    mujoco.mj_step(model, data)

# Refresh settled starts
for obj in objects:
    obj['start_pos'] = data.xpos[obj['id']].copy()
    print(f"  {obj['name']} settled at: {obj['start_pos']}")

# ---------- helpers ----------
DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

def fk_site_pos(q):
    data_fk.qpos[:] = data.qpos[:]
    data_fk.qpos[:6] = q[:6]
    mujoco.mj_forward(model, data_fk)
    return data_fk.site_xpos[site_id].copy()

def mpc_go_to(q_target, hold_gripper=0, max_steps=1200, tol=0.045):
    """MPC to joint target from current pose."""
    for step in range(max_steps):
        rs = state.get_robot_state()
        err = np.linalg.norm(rs[:6] - q_target)
        if err < tol:
            return True, step
        try:
            action, _ = mpc.compute_control(rs, q_target)
            data.ctrl[:6] = action
        except Exception:
            data.ctrl[:6] = q_target
        data.ctrl[6] = hold_gripper
        mujoco.mj_step(model, data, nstep=100)
        viewer.sync()
    return False, max_steps

def ik_to_pos(pos_xyz, tol=0.012, iters=600):
    q, ok = ik.solve(pos_xyz, target_quat=DOWN_QUAT_WXYZ, max_iterations=iters, tolerance=tol)
    return q, ok

def ee_cartesian_delta_z(dz, grip, steps=600, tol=0.02):
    """Move straight in Z in Cartesian by solving IK to current_pos + dz."""
    cur = data.site_xpos[site_id].copy()
    tgt = cur + np.array([0.0, 0.0, float(dz)])
    q, _ = ik_to_pos(tgt, tol=0.01, iters=700)
    ok, _ = mpc_go_to(q, hold_gripper=grip, max_steps=steps, tol=tol)
    return ok

def object_top_z(obj_id, half):
    # data.xpos is COM; top surface is COM.z + half-extent
    return float(data.xpos[obj_id][2] + half)

def test_lift_success(obj_id, half, baseline_top_z, lift_m=0.01, thresh=0.004):
    """
    Close and lift by 'lift_m' vertically; success if TOP rises by > thresh.
    NOTE: uses the SAME definition of TOP (COM.z + half) before & after.
    """
    data.ctrl[6] = 255
    ee_cartesian_delta_z(lift_m, grip=255, steps=350, tol=0.02)
    new_top = object_top_z(obj_id, half)  # <-- FIXED: use same 'half'
    return (new_top - baseline_top_z) > thresh, new_top

def robust_pick(obj, pre_above=0.12, pre_clear=0.025, attempts=6):
    """
    Robust pick with retries:
      1) Go to 'above' (pre_above).
      2) Pre-contact at object_top + pre_clear.
      3) Final settle at (top + push), closing near the bottom.
      4) Test-lift 1 cm; if not attached, reopen, back off up, retry with XY offsets.
    """
    obj_id   = obj['id']
    half     = obj['size']

    # 1) go well above object
    com0 = data.xpos[obj_id].copy()
    top0 = object_top_z(obj_id, half)
    above = com0.copy(); above[2] = top0 + pre_above
    q_above, _ = ik_to_pos(above, tol=0.02, iters=800)
    mpc_go_to(q_above, hold_gripper=0, max_steps=1000, tol=0.05)

    # spiral XY offsets (meters)
    spiral = [
        (0.00, 0.00),
        (0.006, 0.000), (-0.006, 0.000),
        (0.000, 0.006), (0.000, -0.006),
        (0.006, 0.006), (-0.006, -0.006),
        (0.008, 0.000), (0.000, 0.008),
    ]

    for attempt in range(min(attempts, len(spiral))):
        dx, dy = spiral[attempt]

        # sense again
        com = data.xpos[obj_id].copy()
        top = object_top_z(obj_id, half)

        # 2) pre-contact: slightly above the top
        pre = np.array([com[0] + dx, com[1] + dy, top + pre_clear])
        q_pre, _ = ik_to_pos(pre, tol=0.012, iters=700)
        mpc_go_to(q_pre, hold_gripper=0, max_steps=800, tol=0.05)

        # 3) final settle: tiny push onto the top, while closing
        # a bit deeper each retry, capped at 6 mm
        push = min(0.006, 0.002 + 0.0015*attempt)
        final = np.array([com[0] + dx, com[1] + dy, top + push])
        q_fin, _ = ik_to_pos(final, tol=0.008, iters=800)

        # approach to final while starting to close
        mpc_go_to(q_fin, hold_gripper=180, max_steps=600, tol=0.02)

        # squeeze dwell
        for _ in range(100):
            data.ctrl[:6] = data.qpos[:6]
            data.ctrl[6] = 255
            mujoco.mj_step(model, data, nstep=100)
            viewer.sync()

        # 4) test-lift (1 cm) with CORRECT top comparison
        success, _ = test_lift_success(obj_id, half, baseline_top_z=top, lift_m=0.01, thresh=0.004)
        if success:
            return True

        # Failed grasp → reopen, back up, and retry with next offset
        for _ in range(60):
            data.ctrl[:6] = data.qpos[:6]
            data.ctrl[6] = 0
            mujoco.mj_step(model, data, nstep=100); viewer.sync()
        ee_cartesian_delta_z(0.012, grip=0, steps=300, tol=0.02)

    return False

def safe_lift_up(z_up=0.12):
    grip = int(data.ctrl[6])
    return ee_cartesian_delta_z(z_up, grip=grip, steps=700, tol=0.03)

def safe_retreat_up(z_up=0.06, grip=None):
    if grip is None:
        grip = int(data.ctrl[6])
    return ee_cartesian_delta_z(z_up, grip=grip, steps=500, tol=0.04)

# ---------- main loop ----------
for sort_idx, obj in enumerate(sorted_objs):
    print(f"\n### {obj['name'].upper()} ###")

    # Ensure gripper open before first pick
    if sort_idx == 0:
        for _ in range(50):
            data.ctrl[:6] = data.qpos[:6]
            data.ctrl[6] = 0
            mujoco.mj_step(model, data, nstep=100); viewer.sync()

    picked = robust_pick(obj, pre_above=0.12, pre_clear=0.025, attempts=8)

    if not picked:
        print("  ❌ Could not grasp after retries — skipping to next target.")
        continue

    print("  Lifting object straight up...")
    safe_lift_up(0.12)

    # Move to bin while holding
    bin_joints = np.array([2.8 + sort_idx*0.3, -1.0, 1.2, -1.6, -1.57, 0.0])
    print(f"  Moving to bin {sort_idx+1}...")
    mpc_go_to(bin_joints, hold_gripper=255, max_steps=700, tol=0.055)

    # Pre-place, release, retreat
    print("  Pre-place raise...")
    safe_retreat_up(0.04, grip=255)

    print("  Releasing...")
    for _ in range(100):
        data.ctrl[:6] = data.qpos[:6]
        data.ctrl[6] = 0
        mujoco.mj_step(model, data, nstep=100); viewer.sync()

    print("  Retreating upward from bin...")
    safe_retreat_up(0.06, grip=0)

print(f"\n{'='*70}\nSORTING COMPLETE!\n{'='*70}")

print("\nClose viewer to exit...")
try:
    while viewer.is_running():
        mujoco.mj_step(model, data, nstep=100)
        viewer.sync()
except KeyboardInterrupt:
    pass
