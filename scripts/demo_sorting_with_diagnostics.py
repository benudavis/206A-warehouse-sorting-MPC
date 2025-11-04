#!/usr/bin/env python3
"""
Demo sorting with integrated diagnostic logging.

Shows how easy it is to add comprehensive diagnostics to any demo.
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
from src.diagnostics import DiagnosticLogger  # <-- Just import the logger!

EE_SITE = "arm_hand_pinch"

script_dir = Path(__file__).parent.parent / "sim"
models_dir = script_dir / "models"

scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))

arm_spec.site('attachment_site').attach_body(hand_spec.worldbody, "hand_", "")
scene.site('robot_site').attach_body(arm_spec.worldbody, "arm_", "")

# Create 3 cubes
positions = [
    [0.42, -0.38, 0.52],
    [0.47, -0.405, 0.52],
    [0.52, -0.43, 0.52],
]
sizes = [0.02, 0.025, 0.03]
colors = [[1, 0.2, 0.2, 1], [0.2, 0.2, 1, 1], [0.2, 1, 0.2, 1]]
names = ["red_small", "blue_medium", "green_large"]

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

# Home position
model.key_qpos[0][model.jnt('arm_shoulder_pan_joint').qposadr] += np.pi
model.key_ctrl[0][model.jnt('arm_shoulder_pan_joint').dofadr] += np.pi
mujoco.mj_resetDataKeyframe(model, data, 0)

print("="*70)
print("DEMO WITH INTEGRATED DIAGNOSTICS")
print("="*70)

# Settle
for _ in range(300):
    mujoco.mj_step(model, data)

# Initialize diagnostic logger
logger = DiagnosticLogger(model, data, site_name=EE_SITE)

# Add tracked objects
objects = []
for i, name in enumerate(names):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos = data.xpos[bid].copy()
    objects.append({'id': bid, 'name': name, 'size': sizes[i], 'start_pos': pos.copy()})
    
    # Register with logger
    logger.add_tracked_object(name, bid, size=sizes[i], mass=0.05, initial_pos=pos)
    print(f"   {name}: {sizes[i]*1000:.0f}mm at {pos}")

# Init controllers
ik = IKSolver(model, data, site_name=EE_SITE)
mpc = MPCController(n_joints=6, horizon=30, dt=0.01)
state = SimulationState(model, data, ee_site_name=EE_SITE)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
viewer = mujoco.viewer.launch_passive(model, data)

sorted_objs = sorted(objects, key=lambda x: x['size'])

print(f"\n{'='*70}\nSTARTING SORT\n{'='*70}")
for _ in range(100):
    mujoco.mj_step(model, data)

for obj in objects:
    obj['start_pos'] = data.xpos[obj['id']].copy()

# Helper functions
DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

def fk_site_pos(q):
    data_fk.qpos[:] = data.qpos[:]
    data_fk.qpos[:6] = q[:6]
    mujoco.mj_forward(model, data_fk)
    return data_fk.site_xpos[site_id].copy()

def mpc_go_to(q_target, hold_gripper=0, max_steps=1400, tol=0.045, obj_name=None, phase="move", attempt=0):
    """MPC with integrated logging"""
    for step in range(max_steps):
        rs = state.get_robot_state()
        err = np.linalg.norm(rs[:6] - q_target)
        
        # Log every 20 steps
        if obj_name and step % 20 == 0:
            logger.log_state(obj_name, phase, attempt)
        
        if err < tol:
            if obj_name:
                logger.log_state(obj_name, phase, attempt)
                logger.log_mpc_convergence(obj_name, phase, q_target, rs[:6], 
                                          steps=step, converged=True, final_error=err, tolerance=tol)
            return True, step
        
        try:
            action, _ = mpc.compute_control(rs, q_target)
            data.ctrl[:6] = action
        except Exception:
            data.ctrl[:6] = q_target
        data.ctrl[6] = hold_gripper
        mujoco.mj_step(model, data, nstep=100)
        viewer.sync()
    
    if obj_name:
        logger.log_state(obj_name, phase, attempt)
        logger.log_mpc_convergence(obj_name, phase, q_target, rs[:6],
                                  steps=max_steps, converged=False, final_error=err, tolerance=tol)
    return False, max_steps

def ik_to_pos(pos_target, obj_name=None, phase="ik", tol=0.012, iters=600):
    """IK with integrated logging"""
    q, ok = ik.solve(pos_target, target_quat=DOWN_QUAT_WXYZ, max_iterations=iters, tolerance=tol)
    
    achieved = fk_site_pos(q)
    
    if obj_name:
        logger.log_ik_result(obj_name, phase, pos_target, achieved, 
                           target_quat=DOWN_QUAT_WXYZ, success=ok, 
                           iterations=iters, tolerance=tol)
    
    return q, ok

def object_top_z(obj_id, obj_half):
    return float(data.xpos[obj_id][2] + obj_half)

def test_lift(obj_id, obj_name, baseline_top_z, lift_cm=0.01, attempt=0):
    """Test lift with logging"""
    data.ctrl[6] = 255
    
    logger.log_state(obj_name, f"pre_test_lift", attempt)
    
    # Simple lift motion
    cur = data.site_xpos[site_id].copy()
    tgt = cur + np.array([0.0, 0.0, lift_cm])
    q, _ = ik_to_pos(tgt, obj_name=obj_name, phase=f"test_lift_ik")
    mpc_go_to(q, hold_gripper=255, max_steps=350, tol=0.02,
             obj_name=obj_name, phase=f"test_lift_motion", attempt=attempt)
    
    new_top = object_top_z(obj_id, 0.0)
    lift_delta = new_top - baseline_top_z
    success = lift_delta > 0.006
    
    logger.log_state(obj_name, f"post_test_lift", attempt)
    logger.log_lift_test(obj_name, attempt, baseline_top_z, new_top, threshold=0.006, success=success)
    
    return success, new_top

def robust_pick(obj, attempts=4):
    """Simplified pick routine with full logging"""
    obj_id = obj['id']
    obj_name = obj['name']
    half = obj['size']
    
    print(f"\n  Picking {obj_name}...")
    
    # Go above
    top0 = object_top_z(obj_id, half)
    pos_above = data.xpos[obj_id].copy()
    pos_above[2] = top0 + 0.12
    
    logger.log_state(obj_name, "pre_approach", 0)
    q_above, _ = ik_to_pos(pos_above, obj_name=obj_name, phase="approach_above")
    mpc_go_to(q_above, hold_gripper=0, max_steps=1200, tol=0.05,
             obj_name=obj_name, phase="approach_above", attempt=0)
    
    # Try different XY offsets
    spiral = [(0.0, 0.0), (0.006, 0.0), (-0.006, 0.0), (0.0, 0.006)]
    
    for attempt in range(min(attempts, len(spiral))):
        print(f"    Attempt {attempt+1}...")
        dx, dy = spiral[attempt]
        
        top = object_top_z(obj_id, half)
        com = data.xpos[obj_id].copy()
        
        # Pre-contact
        pre = np.array([com[0] + dx, com[1] + dy, top + 0.025])
        q_pre, _ = ik_to_pos(pre, obj_name=obj_name, phase=f"pre_contact_att{attempt}")
        logger.log_state(obj_name, f"pre_contact", attempt)
        mpc_go_to(q_pre, hold_gripper=0, max_steps=1000, tol=0.05,
                 obj_name=obj_name, phase=f"move_to_precontact", attempt=attempt)
        
        # Final settle
        final = np.array([com[0] + dx, com[1] + dy, top + 0.002])
        q_fin, _ = ik_to_pos(final, obj_name=obj_name, phase=f"settle_att{attempt}")
        logger.log_state(obj_name, f"pre_settle", attempt)
        mpc_go_to(q_fin, hold_gripper=180, max_steps=700, tol=0.02,
                 obj_name=obj_name, phase=f"settle_descent", attempt=attempt)
        
        # Squeeze
        logger.log_state(obj_name, f"pre_squeeze", attempt)
        for _ in range(120):
            data.ctrl[:6] = data.qpos[:6]
            data.ctrl[6] = 255
            mujoco.mj_step(model, data, nstep=100)
            viewer.sync()
        logger.log_state(obj_name, f"post_squeeze", attempt)
        
        # Test lift
        success, new_top = test_lift(obj_id, obj_name, baseline_top_z=top, lift_cm=0.01, attempt=attempt)
        
        # Log grasp attempt
        ee_pos = data.site_xpos[site_id].copy()
        dist = np.linalg.norm(ee_pos - com)
        logger.log_grasp_attempt(obj_name, attempt, contact_distance=dist,
                                gripper_closure=255, success=success,
                                reason="lift_test_passed" if success else "lift_test_failed")
        
        if success:
            print(f"    ✅ Grasped!")
            return True
        
        # Failed: back up
        print(f"    ❌ Failed, retrying...")
        for _ in range(80):
            data.ctrl[:6] = data.qpos[:6]
            data.ctrl[6] = 0
            mujoco.mj_step(model, data, nstep=100)
            viewer.sync()
    
    return False

# Main sorting loop - try only first object for demo
for sort_idx, obj in enumerate(sorted_objs[:1]):
    print(f"\n{'='*70}")
    print(f"### {obj['name'].upper()} ###")
    print(f"{'='*70}")
    
    # Open gripper
    for _ in range(60):
        data.ctrl[:6] = data.qpos[:6]
        data.ctrl[6] = 0
        mujoco.mj_step(model, data, nstep=100)
        viewer.sync()
    
    # Try to pick
    picked = robust_pick(obj, attempts=4)
    
    if not picked:
        logger.add_error(f"Failed to pick {obj['name']} after all attempts")
        print(f"  ❌ Could not grasp {obj['name']}")
        continue
    
    print(f"  ✅ Successfully grasped {obj['name']}")
    
    # Move to bin (simplified)
    bin_joints = np.array([2.8, -1.0, 1.2, -1.6, -1.57, 0.0])
    logger.log_state(obj['name'], "pre_transport", 0)
    mpc_go_to(bin_joints, hold_gripper=255, max_steps=800, tol=0.055,
             obj_name=obj['name'], phase="transport_to_bin", attempt=0)
    
    # Release
    logger.log_state(obj['name'], "pre_release", 0)
    for _ in range(120):
        data.ctrl[:6] = data.qpos[:6]
        data.ctrl[6] = 0
        mujoco.mj_step(model, data, nstep=100)
        viewer.sync()
    logger.log_state(obj['name'], "post_release", 0)

print(f"\n{'='*70}")
print("GENERATING DIAGNOSTIC REPORT")
print(f"{'='*70}")

# Generate comprehensive report - this is all you need!
output_dir = logger.generate_report(
    output_dir="data/diagnostics",
    show_plots=True,
    save_plots=True
)

print(f"\n{'='*70}")
print(f"Report saved to: {output_dir}")
print(f"{'='*70}")
print("\nClose viewer to exit...")

try:
    while viewer.is_running():
        mujoco.mj_step(model, data, nstep=100)
        viewer.sync()
except KeyboardInterrupt:
    pass

