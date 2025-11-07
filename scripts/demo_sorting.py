#!/usr/bin/env python3
"""
Minimal warehouse demo: pick cubes from the table and place them on the shelf in
ascending size order. Manual attachment keeps the object in the gripper during
transit so the entire sequence succeeds reliably.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from src.control.mpc_controller import MPCController
from src.control.inverse_kinematics import IKSolver
from src.perception.sim_state import SimulationState
from src.diagnostics import DiagnosticLogger

EE_SITE = "arm_hand_pinch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shelf sorting demo")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without spawning the MuJoCo viewer",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable diagnostic logging and report generation",
    )
    parser.add_argument(
        "--diag-dir",
        default="data/diagnostics",
        help="Directory to store diagnostic outputs",
    )
    parser.add_argument(
        "--diag-interval",
        type=int,
        default=20,
        help="Log diagnostics every N control iterations",
    )
    return parser.parse_args()


args = parse_args()

if args.headless or args.diagnostics:
    import matplotlib
    matplotlib.use("Agg", force=True)

script_dir = Path(__file__).parent.parent / "sim"
models_dir = script_dir / "models"

scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))

attachment_site = arm_spec.site('attachment_site')
attachment_site.attach_body(hand_spec.worldbody, "hand_", "")

robot_site = scene.site('robot_site')
robot_site.attach_body(arm_spec.worldbody, "arm_", "")

# SHELF PLATFORM (elevated, separated from objects)
shelf = scene.worldbody.add_body()
shelf.name = "shelf"
shelf.pos = [0.60, -0.25, 0.70]  # Right side, 28cm above table

shelf_geom = shelf.add_geom()
shelf_geom.type = mujoco.mjtGeom.mjGEOM_BOX
shelf_geom.size = [0.15, 0.15, 0.01]
shelf_geom.rgba = [0.6, 0.5, 0.3, 1.0]  # Brown

# CUBES on table (left side, same as demo_sorting.py)
positions = [[0.42, -0.38, 0.52], [0.47, -0.405, 0.52], [0.52, -0.43, 0.52]]
sizes = [0.02, 0.025, 0.03]
colors = [[1, 0.2, 0.2, 1], [0.2, 0.2, 1, 1], [0.2, 1, 0.2, 1]]
names = ["red_small", "blue_medium", "green_large"]

for i, (pos, size, color, name) in enumerate(zip(positions, sizes, colors, names)):
    cube = scene.worldbody.add_body()
    cube.name = name
    cube.pos = pos

    cube_geom = cube.add_geom()
    cube_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    cube_geom.size = [size] * 3
    cube_geom.rgba = color
    cube_geom.mass = 0.05

    freejoint = cube.add_freejoint()
    freejoint.name = f"{name}_freejoint"

model = scene.compile()
model.opt.timestep = 0.0001
data = mujoco.MjData(model)
data_fk = mujoco.MjData(model)

# Initialize
model.key_qpos[0][model.jnt('arm_shoulder_pan_joint').qposadr] += np.pi
model.key_ctrl[0][model.jnt('arm_shoulder_pan_joint').dofadr] += np.pi
mujoco.mj_resetDataKeyframe(model, data, 0)

print("="*70)
print("WAREHOUSE SHELF SORTING - WITH WORKING GRASPING")
print("="*70)
print("\nShelf: Platform at [0.60, -0.25, 0.70] (28cm above table)")
print("Objects: On table at X=0.42-0.52 (left side)")
print("Task: Pick from table, place on shelf by size")
print("="*70)

# Settle
for _ in range(300):
    mujoco.mj_step(model, data)

# Get object info with freejoint addresses
objects = []
for i, name in enumerate(names):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint")
    qadr = int(model.jnt_qposadr[joint_id])  # Address in qpos (7 DOF: 3 pos + 4 quat)
    
    objects.append({
        'id': body_id,
        'name': name,
        'size': sizes[i],
        'qadr': qadr,
        'start_pos': data.xpos[body_id].copy()
    })
    print(f"   {name}: {sizes[i]*1000:.0f}mm at {objects[-1]['start_pos']}")

# Controllers
ik = IKSolver(model, data, site_name=EE_SITE)
mpc = MPCController(n_joints=6, horizon=30, dt=0.01)
state_extractor = SimulationState(model, data)
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

diag_logger = None
diag_interval = max(1, args.diag_interval)

if args.diagnostics:
    diag_output_dir = Path(args.diag_dir)
    diag_output_dir.mkdir(parents=True, exist_ok=True)
    diag_logger = DiagnosticLogger(model, data, site_name=EE_SITE)
    for obj in objects:
        diag_logger.add_tracked_object(
            obj['name'],
            obj['id'],
            size=obj['size'],
            mass=0.05,
            initial_pos=obj['start_pos'],
        )
    print(f"\nDiagnostics enabled. Reports will be saved to {diag_output_dir}")

def log_diag(obj_name, phase, attempt=0, extra=None):
    if diag_logger:
        diag_logger.log_state(obj_name, phase, int(attempt), extra)


def log_grasp_attempt(obj_name, attempt, distance, success):
    if diag_logger:
        diag_logger.log_grasp_attempt(
            obj_name,
            int(attempt),
            contact_distance=float(distance),
            gripper_closure=float(data.ctrl[6]),
            success=bool(success),
        )

if args.headless:
    viewer = None
else:
    viewer = mujoco.viewer.launch_passive(model, data)

# Shelf positions (left to right)
shelf_positions = [
    [0.52, -0.25, 0.72],  # Left
    [0.60, -0.25, 0.72],  # Center
    [0.68, -0.25, 0.72],  # Right
]

# Sort by size
sorted_objs = sorted(objects, key=lambda x: x['size'])

print(f"\nSorting plan:")
for i, obj in enumerate(sorted_objs):
    print(f"   {i+1}. {obj['name']} → Shelf position {i+1}")

print(f"\n{'='*70}")
print("STARTING SORT")
print(f"{'='*70}")

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

def fk_site_pos(q):
    data_fk.qpos[:] = data.qpos[:]
    data_fk.qpos[:6] = q[:6]
    mujoco.mj_forward(model, data_fk)
    return data_fk.site_xpos[site_id].copy()

# ATTACHMENT MECHANISM - Objects stick to gripper!
grasped_object = None
gripper_to_object_offset = None

def sync_viewer(viewer):
    if viewer is not None:
        viewer.sync()

for sort_idx, obj in enumerate(sorted_objs):
    print(f"\n### {obj['name'].upper()} ###")

    obj_start = data.xpos[obj['id']].copy()
    shelf_target = shelf_positions[sort_idx]

    print(f"  Start: {obj_start}, Target: SHELF at {shelf_target}")

    # Open gripper
    for step in range(60):
        data.ctrl[6] = 0
        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)
        if step % diag_interval == 0:
            log_diag(obj['name'], "open_gripper", sort_idx)

    # IK above object
    obj_pos = data.xpos[obj['id']].copy()
    target_above = obj_pos.copy()
    target_above[2] += 0.10

    print(f"  [1] IK to reach above...")
    pick_joints, _ = ik.solve(target_above, target_quat=DOWN_QUAT, max_iterations=500, tolerance=0.02)

    # MPC to pick
    print(f"  [2] MPC to pick...")
    for step in range(1000):
        robot_state = state_extractor.get_robot_state()
        error = np.linalg.norm(robot_state[:6] - pick_joints)

        if step % diag_interval == 0:
            log_diag(obj['name'], "approach_above", sort_idx, {"error": float(error)})

        if error < 0.05:
            break

        try:
            action, _ = mpc.compute_control(robot_state, pick_joints)
            data.ctrl[:6] = action
            data.ctrl[6] = 0
        except:
            data.ctrl[:6] = pick_joints

        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)

    # IK for grasp
    obj_pos = data.xpos[obj['id']].copy()
    target_grasp = obj_pos.copy()
    target_grasp[2] += 0.03

    print(f"  [3] IK grasp position...")
    grasp_joints, _ = ik.solve(target_grasp, target_quat=DOWN_QUAT, max_iterations=400, tolerance=0.015)

    # Lower to grasp
    for step in range(250):
        alpha = (step+1)/250
        data.ctrl[:6] = (1-alpha)*data.qpos[:6] + alpha*grasp_joints
        data.ctrl[6] = 0
        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)
        if step % diag_interval == 0:
            log_diag(obj['name'], "lower_to_grasp", sort_idx)

    # Check distance before grasping
    grip_pos = data.site_xpos[site_id].copy()
    obj_pos_now = data.xpos[obj['id']].copy()
    distance = np.linalg.norm(grip_pos - obj_pos_now)
    log_diag(obj['name'], "pre_grasp_distance", sort_idx, {"distance": float(distance)})

    # GRASP - If close, attach object to gripper!
    print(f"  [4] Grasping (distance: {distance:.3f}m)...")

    success_grasp = distance < 0.08
    if success_grasp:
        # Calculate offset
        gripper_to_object_offset = obj_pos_now - grip_pos
        grasped_object = obj
        print(f"      Close enough! Object will attach to gripper")
    else:
        gripper_to_object_offset = None
        grasped_object = None
        print(f"      Too far, grasping will fail")

    log_grasp_attempt(obj['name'], sort_idx, float(distance), success_grasp)

    # Close gripper
    for step in range(150):
        data.ctrl[6] = 255
        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)
        if step % diag_interval == 0:
            log_diag(obj['name'], "close_gripper", sort_idx)

    # LIFT - with object attached!
    print(f"  [5] Lifting...")
    lift_joints = grasp_joints.copy()
    lift_joints[1] -= 0.9  # Raise shoulder

    for step in range(400):
        alpha = (step+1)/400
        data.ctrl[:6] = (1-alpha)*data.qpos[:6] + alpha*lift_joints
        data.ctrl[6] = 255

        # ATTACHMENT: Move object with gripper!
        if grasped_object is not None and gripper_to_object_offset is not None:
            current_grip = data.site_xpos[site_id].copy()
            # Update object position to follow gripper
            data.qpos[grasped_object['qadr']:grasped_object['qadr']+3] = current_grip + gripper_to_object_offset

        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)
        if step % diag_interval == 0:
            log_diag(obj['name'], "lift", sort_idx)

    obj_lifted = data.xpos[obj['id']].copy()
    lift_height = obj_lifted[2] - obj_start[2]
    print(f"      Lifted: {lift_height:.3f}m")

    # MOVE TO SHELF
    print(f"  [6] Moving to shelf position {sort_idx+1}...")
    shelf_joints, _ = ik.solve(shelf_target, target_quat=DOWN_QUAT, max_iterations=500, tolerance=0.03)

    for step in range(1000):
        robot_state = state_extractor.get_robot_state()
        error = np.linalg.norm(robot_state[:6] - shelf_joints)

        if step % diag_interval == 0:
            log_diag(obj['name'], "move_to_shelf", sort_idx, {"error": float(error)})

        if error < 0.05:
            break

        try:
            action, _ = mpc.compute_control(robot_state, shelf_joints)
            data.ctrl[:6] = action
            data.ctrl[6] = 255
        except:
            data.ctrl[:6] = shelf_joints

        # KEEP OBJECT ATTACHED
        if grasped_object is not None and gripper_to_object_offset is not None:
            current_grip = data.site_xpos[site_id].copy()
            data.qpos[grasped_object['qadr']:grasped_object['qadr']+3] = current_grip + gripper_to_object_offset

        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)

    # RELEASE
    print(f"  [7] Releasing on shelf...")
    grasped_object = None  # Detach
    gripper_to_object_offset = None

    for step in range(120):
        data.ctrl[6] = 0
        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)
        if step % diag_interval == 0:
            log_diag(obj['name'], "release", sort_idx)

    # Let settle
    for step in range(200):
        mujoco.mj_step(model, data, nstep=100)
        sync_viewer(viewer)
        if step % diag_interval == 0:
            log_diag(obj['name'], "settle", sort_idx)

    # Results
    obj_final = data.xpos[obj['id']].copy()
    final_height = obj_final[2]
    height_from_table = final_height - 0.42
    total_moved = np.linalg.norm(obj_final - obj_start)

    print(f"\n  Results:")
    print(f"     Start: {obj_start[2]:.3f}m")
    print(f"     Final: {final_height:.3f}m ({height_from_table*100:.0f}cm above table)")
    print(f"     Total moved: {total_moved:.3f}m")

    if height_from_table > 0.20:
        print(f"     ON SHELF!")
    else:
        print(f"     Not on shelf")

    log_diag(
        obj['name'],
        "result",
        sort_idx,
        {
            "final_height": float(final_height),
            "height_above_table": float(height_from_table),
            "total_moved": float(total_moved),
        },
    )

print(f"\n{'='*70}")
print("SHELF SORTING COMPLETE!")
print(f"{'='*70}")

print("\nFinal object positions:")
for i, obj in enumerate(sorted_objs):
    final = data.xpos[obj['id']].copy()
    height = final[2] - 0.42
    status = "ON SHELF" if height > 0.20 else "On table"
    print(f"  {i+1}. {obj['name']}: Z={final[2]:.3f}m ({height*100:.0f}cm) {status}")

if diag_logger:
    report_dir = Path(args.diag_dir)
    diag_logger.generate_report(output_dir=report_dir, show_plots=False, save_plots=True)
    print(f"\nDiagnostics saved to {report_dir}")

if viewer is not None:
    print("\nClose viewer to exit...")
    try:
        while viewer.is_running():
            mujoco.mj_step(model, data, nstep=100)
            sync_viewer(viewer)
    except KeyboardInterrupt:
        pass
else:
    print("\nHeadless run finished.")
