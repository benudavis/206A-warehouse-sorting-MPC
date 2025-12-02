#!/usr/bin/env python3
"""Color-based warehouse sorting demo with obstacle avoidance."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

try:
    import mujoco.viewer
    HAS_VIEWER = True
except Exception:
    HAS_VIEWER = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.control.inverse_kinematics import IKSolver
from src.control.mpc_controller import MPCController
from src.diagnostics import DiagnosticLogger

EE_SITE = "arm_hand_pinch"
MODELS_DIR = Path(__file__).parent.parent / "sim" / "models"

# Box configurations: All boxes start in the middle
# Red boxes go to left basket, Blue boxes go to right basket
BOXES = [
    # Red boxes (go to left basket)
    {"name": "red_1",  "pos": [0.30, -0.30, 0.52], "size": 0.030, "rgba": [0.9, 0.1, 0.1, 1.0], "color": "red"},
    {"name": "red_2",  "pos": [0.35, -0.35, 0.52], "size": 0.028, "rgba": [0.8, 0.0, 0.0, 1.0], "color": "red"},
    {"name": "red_3",  "pos": [0.40, -0.25, 0.52], "size": 0.032, "rgba": [1.0, 0.2, 0.2, 1.0], "color": "red"},
    # Blue boxes (go to right basket)
    {"name": "blue_1", "pos": [0.30, -0.40, 0.52], "size": 0.030, "rgba": [0.1, 0.1, 0.9, 1.0], "color": "blue"},
    {"name": "blue_2", "pos": [0.40, -0.45, 0.52], "size": 0.028, "rgba": [0.0, 0.0, 0.8, 1.0], "color": "blue"},
    {"name": "blue_3", "pos": [0.35, -0.40, 0.52], "size": 0.032, "rgba": [0.2, 0.2, 1.0, 1.0], "color": "blue"},
]

# Basket positions: left and right sides
BASKETS = {
    "red":  {"pos": [-0.30, -0.10, 0.48], "rgba": [0.8, 0.2, 0.2, 0.5]},  # Left side for red
    "blue": {"pos": [ -0.30, -0.60, 0.48], "rgba": [0.2, 0.2, 0.8, 0.5]},  # Right side for blue
}


# -----------------------------
# Quaternion helper functions
# -----------------------------

def quat_mul(q1, q2):
    """Multiply two quaternions (w, x, y, z) in MuJoCo convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conj(q):
    """Conjugate of quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


class FullscreenEnforcer:
    def __init__(self, viewer, retries=30, interval=0.1):
        self.viewer = viewer
        self.retries = max(1, retries)
        self.interval = max(0.01, interval)
        self.attempts = 0
        self.last_attempt = 0.0
        self.done = False

    def __call__(self):
        if self.done:
            return
        now = time.monotonic()
        if self.attempts and now - self.last_attempt < self.interval:
            return
        self.attempts += 1
        self.last_attempt = now

        try:
            if hasattr(self.viewer, "set_fullscreen"):
                self.viewer.set_fullscreen(True)
                self.done = True
                return
        except Exception:
            pass

        if self.attempts >= self.retries:
            self.done = True


def parse_args():
    parser = argparse.ArgumentParser(description="Color-based warehouse sorting demo")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--diagnostics", action="store_true", help="Record diagnostics")
    parser.add_argument("--diag-dir", default="data/diagnostics", help="Diagnostics output directory")
    parser.add_argument("--diag-interval", type=int, default=20, help="Log diagnostics every N steps")
    return parser.parse_args()


def build_world():
    """Build world with boxes, baskets, and (optional) obstacles."""
    scene = mujoco.MjSpec.from_file(str(MODELS_DIR / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robotiq_2f85" / "2f85.xml"))

    arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
    scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")

    # Create baskets (bins)
    for basket_name, basket_config in BASKETS.items():
        basket = scene.worldbody.add_body()
        basket.name = f"basket_{basket_name}"
        basket.pos = basket_config["pos"]
        # Basket is a shallow box
        basket_geom = basket.add_geom()
        basket_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        basket_geom.size = [0.15, 0.15, 0.02]
        basket_geom.rgba = basket_config["rgba"]

    # Vertical wall obstacle between boxes and baskets
    # Positioned well away from robot to avoid initial collision
    obstacle = scene.worldbody.add_body()
    obstacle.name = "obstacle_center_wall"
    obstacle.pos = [-0.05, -0.35, 0.68]  # Between boxes and baskets, safe from robot
    obs_geom = obstacle.add_geom()
    obs_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    obs_geom.size = [0.015, 0.15, 0.10]  # Thin vertical wall: 1.5cm thick, 30cm wide, 20cm tall
    obs_geom.rgba = [0.9, 0.5, 0.1, 0.85]  # Orange, semi-transparent

    # Small shelf above boxes (overhead obstacle)
    # Compute position to avoid initial collision with boxes
    max_box_size = max(cfg["size"] for cfg in BOXES)  # Largest box half-size
    max_box_z = max(cfg["pos"][2] for cfg in BOXES)   # Highest box center z
    max_box_top = max_box_z + max_box_size            # Top of tallest box
    
    shelf_thickness = 0.015  # Half-thickness (3cm total)
    clearance = 0.03  # 3cm gap between box top and shelf bottom
    shelf_z = max_box_top + clearance + shelf_thickness
    
    shelf = scene.worldbody.add_body()
    shelf.name = "shelf_overhead"
    shelf.pos = [0.35, -0.35, shelf_z]  # Above boxes with proper clearance
    shelf_geom = shelf.add_geom()
    shelf_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    shelf_geom.size = [0.15, 0.15, shelf_thickness]  # Horizontal shelf: 30cm x 30cm x 3cm thick
    shelf_geom.rgba = [0.6, 0.45, 0.3, 0.85]  # Brown, semi-transparent

    # Create boxes
    for box_config in BOXES:
        box = scene.worldbody.add_body()
        box.name = box_config["name"]
        box.pos = box_config["pos"]
        geom = box.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [box_config["size"]] * 3
        geom.rgba = box_config["rgba"]
        geom.mass = 0.05
        geom.friction = [1.0, 0.005, 0.0001]
        box.add_freejoint()

    model = scene.compile()
    model.opt.timestep = 0.0005
    data = mujoco.MjData(model)
    return model, data


def settle(model, data, steps=400):
    """Let physics settle."""
    for _ in range(steps):
        mujoco.mj_step(model, data)


def step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook=None):
    """Step simulation with welded attachment handling."""
    # Gripper command (position-controlled gripper index 6)
    data.ctrl[6] = grip

    if attachment["active"] and attachment["qadr"] >= 0:
        # Gripper pose
        grip_pos = data.site_xpos[site_id].copy()
        ee_body_id = model.site_bodyid[site_id]
        grip_quat = data.xquat[ee_body_id].copy()

        # Box pose indices
        qadr = attachment["qadr"]
        dadr = attachment["dadr"]
        box_offset = attachment["offset"]
        q_rel = attachment["q_rel"]

        # Update box position
        data.qpos[qadr : qadr + 3] = grip_pos + box_offset

        # Update box orientation: q_box = q_grip * q_rel
        box_quat = quat_mul(grip_quat, q_rel)
        data.qpos[qadr + 3 : qadr + 7] = box_quat

        # Zero freejoint velocities (6 DoFs: 3 linear + 3 angular)
        if dadr >= 0:
            data.qvel[dadr : dadr + 6] = 0.0

    mujoco.mj_step(model, data, nstep=100)
    if viewer is not None:
        viewer.sync()
        if fullscreen_hook is not None:
            fullscreen_hook()


def move_to(
    model,
    data,
    target,
    steps,
    grip,
    attachment,
    site_id,
    viewer,
    diag=None,
    phase="move",
    fullscreen_hook=None,
    mpc_controller=None,
):
    """
    Move to target joint configuration.

    If mpc_controller is provided, use MPC for long-distance motions and
    visualize the MPC-predicted path. If None, use simple linear interpolation.
    """
    start = data.qpos[:6].copy()

    if mpc_controller is not None:
        data_scratch = mujoco.MjData(model)
        obstacle_warning_shown = False
        mpc_replan_interval = 50  # Recompute MPC every 50 steps (NN FK is expensive!)
        planned_trajectory = None

        for i in range(steps):
            current_q = data.qpos[:6].copy()
            current_dq = data.qvel[:6].copy()
            current_state = np.concatenate([current_q, current_dq])

            # Check for nearby obstacles (if any configured)
            if not obstacle_warning_shown and mpc_controller.obstacles:
                ee_pos = data.site_xpos[site_id].copy()
                min_dist = float("inf")
                for obs_pos, obs_size in mpc_controller.obstacles:
                    dist = mpc_controller._point_box_distance(ee_pos, obs_pos, obs_size)
                    min_dist = min(min_dist, dist)
                if min_dist < mpc_controller.safety_margin * 2:
                    print(f"  ⚠️  Obstacle nearby (dist={min_dist:.3f}m), MPC planning avoidance...")
                    obstacle_warning_shown = True

            # Recompute MPC only every N steps (not every step!)
            if i % mpc_replan_interval == 0:
                try:
                    next_q, planned_trajectory = mpc_controller.compute_control(
                        current_state,
                        target,
                        model=model,
                        data_scratch=data_scratch,
                        site_id=site_id,
                    )
                    data.ctrl[:6] = next_q
                    
                    # Print info only at start
                    if i == 0:
                        data_scratch.qpos[:6] = planned_trajectory[0]
                        mujoco.mj_forward(model, data_scratch)
                        start_ee = data_scratch.site_xpos[site_id].copy()
                        data_scratch.qpos[:6] = planned_trajectory[-1]
                        mujoco.mj_forward(model, data_scratch)
                        end_ee = data_scratch.site_xpos[site_id].copy()
                        dist = np.linalg.norm(end_ee - start_ee)
                        print(f"    MPC planning {dist:.3f}m path ({len(planned_trajectory)} waypoints)")

                except Exception:
                    # Fallback: simple interpolation step
                    alpha = (i + 1) / steps
                    desired = (1.0 - alpha) * start + alpha * target
                    data.ctrl[:6] = desired
            else:
                # Between MPC replans, use simple interpolation
                alpha = (i + 1) / steps
                desired = (1.0 - alpha) * start + alpha * target
                data.ctrl[:6] = desired

            step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", phase, extra_data={"step": i})
    else:
        # Linear interpolation (precise & simple, used for local moves)
        for i in range(steps):
            alpha = (i + 1) / steps
            desired = (1.0 - alpha) * start + alpha * target
            data.ctrl[:6] = desired
            step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", phase, extra_data={"alpha": float(alpha)})


def main():
    args = parse_args()

    if args.headless or args.diagnostics:
        import matplotlib
        matplotlib.use("Agg", force=True)

    print("=" * 70)
    print("COLOR-BASED WAREHOUSE SORTING WITH OBSTACLE AVOIDANCE")
    print("=" * 70)
    print("Layout: Boxes in middle, Baskets on left/right sides")
    print("  Left (Red basket)  ←  [Obstacles]  ←  Boxes  →  [Obstacles]  →  Right (Blue basket)")
    print("Obstacles force MPC to plan intelligent paths around barriers")
    print("=" * 70)

    model, data = build_world()

    # Flip shoulder to face scene
    model.key_qpos[0][model.jnt("arm_shoulder_pan_joint").qposadr] += np.pi
    model.key_ctrl[0][model.jnt("arm_shoulder_pan_joint").dofadr] += np.pi
    mujoco.mj_resetDataKeyframe(model, data, 0)

    settle(model, data)

    ik = IKSolver(model, data, site_name=EE_SITE)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    # Initialize MPC with obstacle avoidance (NN FK may or may not be present)
    print("\nInitializing MPC controller with obstacle avoidance...")
    mpc_controller = MPCController(n_joints=6, horizon=10, dt=0.01)  # Shorter horizon for speed

    # Add obstacles to MPC
    for obstacle_name in ["obstacle_center_wall", "shelf_overhead"]:
        try:
            obs_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle_name)
            if obs_body_id >= 0:
                obs_pos = data.xpos[obs_body_id].copy()
                for gid in range(model.ngeom):
                    if model.geom_bodyid[gid] == obs_body_id:
                        obs_size = model.geom_size[gid].copy()
                        mpc_controller.add_obstacle(obs_pos, obs_size)
                        print(f"  Added obstacle: {obstacle_name}")
                        break
        except Exception as e:
            print(f"  Warning: Could not add obstacle {obstacle_name}: {e}")

    print(f"MPC ready with {len(mpc_controller.obstacles)} obstacles\n" + "=" * 70)

    viewer = None
    fullscreen_hook = None
    if not args.headless and HAS_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data)
        fullscreen_hook = FullscreenEnforcer(viewer)
        fullscreen_hook()

    diag = None
    if args.diagnostics:
        diag_logger = DiagnosticLogger(model, data, site_name=EE_SITE)
        diag = {"logger": diag_logger, "interval": max(1, args.diag_interval)}

    # Collect box information
    boxes = []
    for cfg in BOXES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["name"])
        if body_id < 0:
            continue

        jnt_adr = model.body_jntadr[body_id]
        joint_id = jnt_adr
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])

        boxes.append(
            {
                "name": cfg["name"],
                "body_id": body_id,
                "joint_id": joint_id,
                "qadr": qadr,
                "dadr": dadr,
                "size": cfg["size"],
                "color": cfg["color"],
            }
        )

    # Get basket info
    baskets = {}
    for color in ["red", "blue"]:
        basket_name = f"basket_{color}"
        basket_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, basket_name)
        basket_geom_id = None
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == basket_body:
                basket_geom_id = gid
                break
        baskets[color] = {
            "body_id": basket_body,
            "geom_id": basket_geom_id,
            "thickness": model.geom_size[basket_geom_id][2] if basket_geom_id is not None else 0.02,
        }

    # Common poses
    down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    # Wall obstacle: z=[0.58, 0.78], x=[-0.065, -0.035]
    # Shelf overhead: z=[0.582, 0.612], x=[0.20, 0.50]
    # mid_clear must avoid both - go high and to the side
    mid_clear = np.array([0.55, -0.35, 0.85])  # High above both obstacles, near boxes
    mid_clear_joints, _ = ik.solve(mid_clear, target_quat=down_quat, max_iterations=400)
    home_joints = data.qpos[:6].copy()

    # Attachment state (extended to store pose + freejoint indices)
    attachment = {
        "active": False,
        "qadr": -1,
        "dadr": -1,
        "body_id": -1,
        "offset": np.zeros(3),
        "q_rel": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    }

    # Main sorting loop
    for idx, box in enumerate(boxes):
        box_name = box["name"]
        box_color = box["color"]
        box_body = box["body_id"]
        box_qadr = box["qadr"]
        box_dadr = box["dadr"]
        box_size = box["size"]

        print(f"\n{'='*70}")
        print(f"Sorting {box_name} ({box_color}) → {box_color} basket ({idx+1}/{len(boxes)})")
        print(f"{'='*70}")

        # Get current box position (in case it moved)
        box_pos = data.xpos[box_body].copy()

        # Waypoints relative to sensed box pose (box_pos is the center of the box)
        above = box_pos.copy()
        above[2] += 0.15  # 15cm above box center
        align = box_pos.copy()
        align[2] += 0.05  # 5cm above box center
        contact = box_pos.copy()
        # contact is exactly at box center - gripper wraps around middle of box

        above_joints, _ = ik.solve(
            above, target_quat=down_quat, max_iterations=400, tolerance=0.02
        )
        align_joints, _ = ik.solve(
            align, target_quat=down_quat, max_iterations=400, tolerance=0.015
        )
        contact_joints, _ = ik.solve(
            contact, target_quat=down_quat, max_iterations=400, tolerance=0.01
        )

        # 1) Move above box WITH MPC (longer, obstacle-aware path)
        print("Moving above box...")
        move_to(
            model,
            data,
            above_joints[:6],
            600,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "approach_above",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )

        # 2) Local lowering moves WITHOUT MPC (precise straight-line joint moves)
        print("Lowering to box...")
        move_to(
            model,
            data,
            align_joints[:6],
            300,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "align",
            fullscreen_hook,
            mpc_controller=None,  # <- use linear interpolation for precision
        )
        move_to(
            model,
            data,
            contact_joints[:6],
            200,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "lower",
            fullscreen_hook,
            mpc_controller=None,  # <- use linear interpolation for precision
        )

        # Compute grasp offset and activate attachment
        grip_pos = data.site_xpos[site_id].copy()
        box_now = data.xpos[box_body].copy()
        dist = np.linalg.norm(grip_pos - box_now)

        if dist < 0.07:
            ee_body_id = model.site_bodyid[site_id]
            grip_quat = data.xquat[ee_body_id].copy()
            box_quat = data.xquat[box_body].copy()
            # Relative orientation: q_rel = conj(q_grip) * q_box
            q_rel = quat_mul(quat_conj(grip_quat), box_quat)

            attachment["active"] = True
            attachment["qadr"] = box_qadr
            attachment["dadr"] = box_dadr
            attachment["body_id"] = box_body
            attachment["offset"] = box_now - grip_pos
            attachment["q_rel"] = q_rel

            print(f"✓ {box_name} attached to gripper (distance {dist:.3f}m)")
        else:
            attachment["active"] = False
            attachment["qadr"] = -1
            attachment["dadr"] = -1
            attachment["body_id"] = -1
            attachment["offset"] = np.zeros(3)
            attachment["q_rel"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            print(f"✗ {box_name} out of reach (distance {dist:.3f}m); skipping")
            continue

        # Close gripper
        print("Closing gripper...")
        for i in range(150):
            step_sim(model, data, attachment, site_id, 255, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", "close", extra_data={"step": i})

        if not attachment["active"]:
            print("Attachment failed; skipping placement for this box.")
            continue

        # Get target basket position
        basket_info = baskets[box_color]
        basket_pos = data.xpos[basket_info["body_id"]].copy()
        basket_top_z = basket_pos[2] + basket_info["thickness"]

        # Place position (slightly above basket to drop)
        drop_margin = 0.015
        place_pos = basket_pos.copy()
        place_pos[2] = basket_top_z + box_size + drop_margin

        # Convert to EE target using grasp offset
        place_ee = place_pos - attachment["offset"]
        approach_ee = place_ee.copy()
        approach_ee[2] += 0.10  # approach from above

        approach_joints, _ = ik.solve(
            approach_ee, target_quat=down_quat, max_iterations=400, tolerance=0.02
        )
        place_joints, _ = ik.solve(
            place_ee, target_quat=down_quat, max_iterations=400, tolerance=0.01
        )

        # Transport WITH MPC (obstacle-aware) to mid_clear and approach above basket
        print(f"Transporting to {box_color} basket...")
        move_to(
            model,
            data,
            mid_clear_joints[:6],
            400,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "lift_high",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )
        move_to(
            model,
            data,
            approach_joints[:6],
            400,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "approach_basket",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )

        # Local precise lowering into basket WITHOUT MPC
        move_to(
            model,
            data,
            place_joints[:6],
            300,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "pre_place",
            fullscreen_hook,
            mpc_controller=None,
        )

        # Release box above basket (drop)
        print(f"Releasing into {box_color} basket...")
        attachment["active"] = False
        attachment["qadr"] = -1
        attachment["dadr"] = -1
        attachment["body_id"] = -1
        attachment["q_rel"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for i in range(250):
            step_sim(model, data, attachment, site_id, 0, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", "release", extra_data={"step": i})

        # Back away from basket
        print("Retreating from basket...")
        move_to(
            model,
            data,
            approach_joints[:6],
            300,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "retreat_from_basket",
            fullscreen_hook,
            mpc_controller=None,
        )
        move_to(
            model,
            data,
            mid_clear_joints[:6],
            300,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "retreat_clear",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )
        
        # Only return home at the end, not between each box

    # Return to home position after all boxes are sorted
    print("\nReturning to home position...")
    move_to(
        model,
        data,
        home_joints,
        500,
        0,
        attachment,
        site_id,
        viewer,
        diag,
        "return_home",
        fullscreen_hook,
        mpc_controller=mpc_controller,
    )

    print("\n" + "=" * 70)
    print("✓ SORTING COMPLETE!")
    print("=" * 70)

    # Final positions
    for box in boxes:
        final_pos = data.xpos[box["body_id"]].copy()
        print(f"{box['name']} ({box['color']}): {final_pos}")

    if diag:
        output_dir = Path(args.diag_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        diag["logger"].generate_report(
            output_dir=output_dir,
            show_plots=not args.headless,
            save_plots=True,
        )

    if viewer is not None:
        print("\nClose viewer to exit...")
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data, nstep=100)
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
