#!/usr/bin/env python3
"""Three-cube pick → stack on shelf demo with optional diagnostics."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

try:
    import mujoco.viewer  # noqa: F401
    HAS_VIEWER = True
except Exception:  # pragma: no cover
    HAS_VIEWER = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.control.inverse_kinematics import IKSolver
from src.control.mpc_controller import MPCController
from src.diagnostics import DiagnosticLogger

EE_SITE = "arm_hand_pinch"
MODELS_DIR = Path(__file__).parent.parent / "sim" / "models"

# Three cubes with different sizes and separated starting positions (not stacked).
# They will be stacked on the shelf from largest (bottom) to smallest (top).
CUBE_CONFIGS = [
    {"name": "cube_large", "pos": [0.55, -0.45, 0.52], "size": 0.035, "rgba": [1.0, 0.0, 0.0, 1.0]},  # Red
    {"name": "cube_medium", "pos": [0.70, -0.30, 0.52], "size": 0.030, "rgba": [0.0, 0.0, 1.0, 1.0]},  # Blue
    {"name": "cube_small", "pos": [0.40, -0.55, 0.52], "size": 0.025, "rgba": [0.0, 1.0, 0.0, 1.0]},  # Green
]


def _ensure_viewer_fullscreen(viewer) -> bool:
    """Attempt to force the MuJoCo viewer window into fullscreen."""

    if viewer is None:
        return False

    try:
        from mujoco import glfw
    except ImportError:
        return False

    window_ctx = getattr(viewer, "window", None)
    glfw_window = getattr(viewer, "_window", None)

    if window_ctx is not None and glfw_window is None:
        glfw_window = getattr(window_ctx, "window", None) or window_ctx

    # Prefer the public MuJoCo API (available in >=3.2).
    for method_name, args in (
        ("set_fullscreen", (True,)),
        ("toggle_fullscreen", tuple()),
        ("maximize", tuple()),
    ):
        method = getattr(viewer, method_name, None)
        if callable(method):
            try:
                method(*args)
            except Exception:
                continue
            else:
                return True

    # Fall back to window-scoped helper if exposed (older MuJoCo).
    window_obj = window_ctx or getattr(viewer, "_window", None)
    if window_obj is not None:
        for window_method in ("make_fullscreen", "set_fullscreen", "maximise", "maximize"):
            method = getattr(window_obj, window_method, None)
            if callable(method):
                try:
                    method(True)
                except Exception:
                    continue
                else:
                    return True

    if glfw_window is None:
        return False

    try:
        monitor = glfw.get_primary_monitor()
        if not monitor:
            return False
        mode = glfw.get_video_mode(monitor)
        if not mode:
            return False

        size = getattr(mode, "size", None)
        if size is not None:
            width = getattr(size, "width", None)
            height = getattr(size, "height", None)
            if width is None or height is None:
                try:
                    width, height = size[0], size[1]
                except Exception:
                    return False
        else:
            width = getattr(mode, "width", None)
            height = getattr(mode, "height", None)

        if width is None or height is None:
            return False

        refresh = getattr(mode, "refresh_rate", None)
        if refresh is None:
            refresh = glfw.DONT_CARE

        glfw.set_window_monitor(glfw_window, monitor, 0, 0, int(width), int(height), refresh)
        glfw.focus_window(glfw_window)
        return True
    except Exception:
        return False


class _FullscreenEnforcer:
    def __init__(self, viewer, retries=30, interval=0.1):
        self.viewer = viewer
        self.retries = max(1, retries)
        self.interval = max(0.01, interval)
        self.attempts = 0
        self.last_attempt = 0.0
        self.done = False

    def __call__(self) -> None:
        if self.done:
            return
        now = time.monotonic()
        if self.attempts and now - self.last_attempt < self.interval:
            return
        self.attempts += 1
        self.last_attempt = now
        success = _ensure_viewer_fullscreen(self.viewer)
        if success:
            print("Viewer fullscreen enabled.")
            self.done = True
            return
        if self.attempts >= self.retries:
            print("Viewer fullscreen could not be enforced automatically; toggle manually if needed.")
            self.done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-cube shelf stacking demo")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer")
    parser.add_argument("--diagnostics", action="store_true", help="Record diagnostics and plots")
    parser.add_argument("--diag-dir", default="data/diagnostics", help="Diagnostics output directory")
    parser.add_argument("--diag-interval", type=int, default=20, help="Log diagnostics every N steps")
    return parser.parse_args()


def build_world():
    scene = mujoco.MjSpec.from_file(str(MODELS_DIR / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml"))
    hand_spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "robotiq_2f85" / "2f85.xml"))

    arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
    scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")

    # Shelf
    shelf = scene.worldbody.add_body()
    shelf.name = "shelf"
    shelf.pos = [-0.32, -0.25, 0.70]
    shelf_geom = shelf.add_geom()
    shelf_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    shelf_geom.size = [0.12, 0.12, 0.01]
    shelf_geom.rgba = [0.6, 0.45, 0.3, 1.0]

    # Obstacles between boxes and shelf to force MPC path planning
    # Vertical wall obstacle - blocks direct path from boxes to shelf
    obstacle1 = scene.worldbody.add_body()
    obstacle1.name = "obstacle_wall"
    obstacle1.pos = [0.10, -0.35, 0.75]
    obs1_geom = obstacle1.add_geom()
    obs1_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    obs1_geom.size = [0.25, 0.02, 0.20]  # Wide vertical wall
    obs1_geom.rgba = [0.8, 0.2, 0.2, 0.8]  # Semi-transparent red
    
    # Overhead obstacle - blocks high direct paths
    obstacle2 = scene.worldbody.add_body()
    obstacle2.name = "obstacle_ceiling"
    obstacle2.pos = [0.0, -0.30, 0.85]
    obs2_geom = obstacle2.add_geom()
    obs2_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    obs2_geom.size = [0.20, 0.15, 0.02]  # Horizontal ceiling panel
    obs2_geom.rgba = [0.8, 0.5, 0.2, 0.8]  # Semi-transparent orange

    # Three cubes with different sizes and initial positions
    for cfg in CUBE_CONFIGS:
        cube = scene.worldbody.add_body()
        cube.name = cfg["name"]
        cube.pos = cfg["pos"]
        geom = cube.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [cfg["size"], cfg["size"], cfg["size"]]
        geom.rgba = cfg["rgba"]
        geom.mass = 0.05
        geom.friction = [1.0, 0.005, 0.0001]
        cube.add_freejoint()

    model = scene.compile()
    model.opt.timestep = 0.0005
    data = mujoco.MjData(model)
    return model, data


def settle(model, data, steps=400):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook=None):
    # arm controls [0:6], gripper assumed at index 6
    data.ctrl[6] = grip
    if attachment["active"] and attachment["qadr"] >= 0:
        current_grip = data.site_xpos[site_id].copy()
        # Teleport attached cube to follow the gripper with fixed offset
        qadr = attachment["qadr"]
        data.qpos[qadr : qadr + 3] = current_grip + attachment["offset"]
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
    Move to target using MPC if provided, otherwise linear interpolation.
    
    Args:
        mpc_controller: MPCController instance (if None, uses linear interpolation)
    """
    start = data.qpos[:6].copy()
    
    if mpc_controller is not None:
        # Use MPC for trajectory planning with obstacle avoidance
        data_scratch = mujoco.MjData(model)  # Scratch buffer for MPC FK
        obstacle_warning_shown = False
        
        for i in range(steps):
            # Get current state
            current_q = data.qpos[:6].copy()
            current_dq = data.qvel[:6].copy()
            current_state = np.concatenate([current_q, current_dq])
            
            # Show warning if near obstacles (only once per move)
            if not obstacle_warning_shown and mpc_controller.obstacles:
                # Check current EE position against obstacles
                ee_pos = data.site_xpos[site_id].copy()
                min_dist = float('inf')
                for obs_pos, obs_size in mpc_controller.obstacles:
                    dist = mpc_controller._point_box_distance(ee_pos, obs_pos, obs_size)
                    min_dist = min(min_dist, dist)
                
                if min_dist < mpc_controller.safety_margin * 2:  # Within 2x safety margin
                    print(f"  ⚠️  MPC: Obstacle detected nearby (dist={min_dist:.3f}m), planning avoidance trajectory...")
                    obstacle_warning_shown = True
            
            # Compute MPC control
            try:
                next_q, trajectory = mpc_controller.compute_control(
                    current_state,
                    target,
                    model=model,
                    data_scratch=data_scratch,
                    site_id=site_id,
                )
                data.ctrl[:6] = next_q
            except Exception as e:
                # Fallback to linear interpolation if MPC fails
                alpha = (i + 1) / steps
                desired = (1 - alpha) * start + alpha * target
                data.ctrl[:6] = desired
            
            step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("cube", phase, extra_data={"step": i, "using_mpc": mpc_controller is not None})
    else:
        # Original linear interpolation
        for i in range(steps):
            alpha = (i + 1) / steps
            desired = (1 - alpha) * start + alpha * target
            data.ctrl[:6] = desired
            step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("cube", phase, extra_data={"alpha": float(alpha)})


def main():
    args = parse_args()

    if args.headless or args.diagnostics:
        import matplotlib
        matplotlib.use("Agg", force=True)

    print("=" * 70)
    print("THREE-CUBE STACKING DEMO WITH MPC OBSTACLE AVOIDANCE")
    print("=" * 70)
    print("This demo showcases MPC-based trajectory planning with obstacle avoidance.")
    print("Obstacles are placed between the cubes and shelf to force the MPC")
    print("to plan non-trivial paths. Simple linear interpolation would collide.")
    print("=" * 70)

    model, data = build_world()

    # Flip shoulder to face scene (matches your original code)
    model.key_qpos[0][model.jnt("arm_shoulder_pan_joint").qposadr] += np.pi
    model.key_ctrl[0][model.jnt("arm_shoulder_pan_joint").dofadr] += np.pi
    mujoco.mj_resetDataKeyframe(model, data, 0)

    settle(model, data)

    ik = IKSolver(model, data, site_name=EE_SITE)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    
    # Initialize MPC controller
    print("Initializing MPC controller...")
    mpc_controller = MPCController(
        n_joints=6,
        horizon=20,  # Shorter horizon for faster computation
        dt=0.01,
    )
    
    # Add obstacles to MPC controller for avoidance
    # Extract obstacle positions and sizes from the model
    print("\nConfiguring obstacle avoidance:")
    for obstacle_name in ["obstacle_wall", "obstacle_ceiling"]:
        try:
            obs_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle_name)
            if obs_body_id >= 0:
                # Get obstacle position from body
                obs_pos = data.xpos[obs_body_id].copy()
                
                # Get obstacle size from first geom of this body
                for gid in range(model.ngeom):
                    if model.geom_bodyid[gid] == obs_body_id:
                        obs_size = model.geom_size[gid].copy()
                        mpc_controller.add_obstacle(obs_pos, obs_size)
                        print(f"  - {obstacle_name}: position={obs_pos}, size={obs_size}")
                        break
        except Exception as e:
            print(f"Warning: Could not add obstacle {obstacle_name}: {e}")
    
    print(f"\nMPC controller ready with {len(mpc_controller.obstacles)} obstacles")
    print(f"Safety margin: {mpc_controller.safety_margin}m")
    print("=" * 70)

    viewer = None
    fullscreen_hook = None
    if not args.headless and HAS_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data)
        fullscreen_hook = _FullscreenEnforcer(viewer)
        fullscreen_hook()

    diag = None
    if args.diagnostics:
        diag_logger = DiagnosticLogger(model, data, site_name=EE_SITE)
        diag = {"logger": diag_logger, "interval": max(1, args.diag_interval)}
        print(f"Diagnostics enabled -> outputs in {args.diag_dir}")

    # Collect cube info (body id, freejoint qpos address, size)
    cubes = []
    for cfg in CUBE_CONFIGS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["name"])
        if body_id < 0:
            raise RuntimeError(f"Body '{cfg['name']}' not found in model")

        # Get the first joint of this body (its freejoint)
        jnt_adr = model.body_jntadr[body_id]
        jnt_num = model.body_jntnum[body_id]
        if jnt_num < 1:
            raise RuntimeError(f"Body '{cfg['name']}' has no joints (expected freejoint)")

        joint_id = jnt_adr  # first joint of this body
        qadr = int(model.jnt_qposadr[joint_id])

        cubes.append(
            {
                "name": cfg["name"],
                "body_id": body_id,
                "joint_id": joint_id,
                "qadr": qadr,
                "size": cfg["size"],
            }
        )

    # For diagnostics, track the first cube under generic name "cube"
    if diag and cubes:
        first = cubes[0]
        diag["logger"].add_tracked_object(
            "cube",
            first["body_id"],
            size=first["size"],
            mass=0.05,
            initial_pos=data.xpos[first["body_id"]].copy(),
        )

    # Shelf geometry info (body id & thickness stay constant;
    # we will recompute current shelf pose from data each time).
    shelf_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shelf")

    shelf_geom_id = None
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == shelf_body:
            shelf_geom_id = gid
            break
    if shelf_geom_id is None:
        raise RuntimeError("Shelf geom not found")

    shelf_thickness = model.geom_size[shelf_geom_id][2]

    # Sort cubes by size -> largest bottom, smallest top
    cubes_sorted = sorted(cubes, key=lambda c: c["size"], reverse=True)

    # Common IK targets
    down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    mid_clear = np.array([0.0, -0.30, 0.95])
    mid_clear_joints, _ = ik.solve(mid_clear, target_quat=down_quat, max_iterations=400, tolerance=0.05)

    # We'll recompute shelf position each time for robustness
    shelf_pos_initial = data.xpos[shelf_body].copy()
    shelf_xy = shelf_pos_initial[:2]

    high_stack = np.array([shelf_xy[0], shelf_xy[1], 1.0])
    high_stack_joints, _ = ik.solve(high_stack, target_quat=down_quat, max_iterations=400, tolerance=0.05)

    home_joints = data.qpos[:6].copy()

    # Shared attachment state
    attachment = {"active": False, "qadr": -1, "offset": np.zeros(3)}

    # ---- Main loop: pick and place each cube into the stack ----
    for idx, cube in enumerate(cubes_sorted):
        cube_body = cube["body_id"]
        cube_qadr = cube["qadr"]
        size = cube["size"]

        # *** Sense current cube position (in case it moved from initial config) ***
        cube_pos = data.xpos[cube_body].copy()

        # Pick waypoints (relative to current sensed pose)
        above = cube_pos.copy();   above[2]   += size + 0.12
        align = cube_pos.copy();   align[2]   += size + 0.03
        contact = cube_pos.copy(); contact[2] += size + 0.01
        high_pick = cube_pos.copy(); high_pick[2] = 1.0

        above_joints, _ = ik.solve(above, target_quat=down_quat, max_iterations=400, tolerance=0.02)
        align_joints, _ = ik.solve(align, target_quat=down_quat, max_iterations=400, tolerance=0.015)
        contact_joints, _ = ik.solve(contact, target_quat=down_quat, max_iterations=400, tolerance=0.01)
        high_pick_joints, _ = ik.solve(high_pick, target_quat=down_quat, max_iterations=400, tolerance=0.05)

        print(f"\n=== Handling {cube['name']} ({idx + 1}/{len(cubes_sorted)}) ===")

        # Move above cube
        print("Moving above cube...")
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

        # Lower and make contact
        print("Lowering to cube...")
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
            mpc_controller=mpc_controller,
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
            mpc_controller=mpc_controller,
        )

        # Compute grasp offset and activate attachment
        grip_pos = data.site_xpos[site_id].copy()
        cube_now = data.xpos[cube_body].copy()
        dist = np.linalg.norm(grip_pos - cube_now)
        if dist < 0.07:
            attachment["active"] = True
            attachment["qadr"] = cube_qadr
            attachment["offset"] = cube_now - grip_pos
            print(f"{cube['name']} attached to gripper (distance {dist:.3f}).")
        else:
            attachment["active"] = False
            attachment["qadr"] = -1
            attachment["offset"] = np.zeros(3)
            print(f"{cube['name']} out of reach (distance {dist:.3f}); continuing without attachment.")

        if diag:
            diag["logger"].log_state("cube", "pre_grasp", extra_data={"distance": float(dist)})

        # Close gripper
        print("Closing gripper...")
        for i in range(150):
            step_sim(model, data, attachment, site_id, 255, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("cube", "close", extra_data={"step": i})

        if not attachment["active"]:
            print("Attachment failed; skipping placement for this cube.")
            continue

        # =========================
        #  LIVE STACK STATE LOGIC
        # =========================
        #
        # Re-sense shelf and already placed cubes to compute where the stack
        # actually is. We only use cubes that are near the shelf center.
        shelf_pos = data.xpos[shelf_body].copy()
        shelf_top_z = shelf_pos[2] + shelf_thickness

        stack_top_z = shelf_top_z
        xy_radius = 0.15  # accept cubes within this radius of shelf center

        for prev in cubes_sorted[:idx]:
            prev_body = prev["body_id"]
            prev_size = prev["size"]
            prev_pos = data.xpos[prev_body].copy()
            # Only consider cubes that are actually on/near the stack region
            if np.linalg.norm(prev_pos[:2] - shelf_xy) < xy_radius:
                top = prev_pos[2] + prev_size
                if top > stack_top_z:
                    stack_top_z = top

        # Desired center for this cube based on current stack top
        desired_cube_center = np.array(
            [shelf_xy[0], shelf_xy[1], stack_top_z + size]
        )

        # Place slightly above the desired center so we drop instead of press
        drop_margin = 0.015  # 1.5 cm above ideal center
        pre_place_cube_center = desired_cube_center.copy()
        pre_place_cube_center[2] += drop_margin

        # Convert pre-place cube pose to EE targets using the grasp offset
        pre_place_ee = pre_place_cube_center - attachment["offset"]
        shelf_above_ee = pre_place_ee.copy()
        shelf_above_ee[2] += 0.10  # come in from above

        # IK for approach-above and pre-place poses
        shelf_above_joints, _ = ik.solve(
            shelf_above_ee,
            target_quat=down_quat,
            max_iterations=400,
            tolerance=0.02,
        )
        pre_place_joints, _ = ik.solve(
            pre_place_ee,
            target_quat=down_quat,
            max_iterations=400,
            tolerance=0.01,
        )

        print("Moving to shelf / stack position...")
        move_to(
            model,
            data,
            high_pick_joints[:6],
            300,
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
            mid_clear_joints[:6],
            400,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "clear",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )
        move_to(
            model,
            data,
            high_stack_joints[:6],
            400,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "approach_high",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )
        move_to(
            model,
            data,
            shelf_above_joints[:6],
            300,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "approach_shelf",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )
        move_to(
            model,
            data,
            pre_place_joints[:6],
            300,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "pre_place",
            fullscreen_hook,
            mpc_controller=mpc_controller,
        )

        # Release cube above the live-computed stack so it can drop and settle
        print("Releasing on stack (drop from pre-place)...")
        attachment["active"] = False
        attachment["qadr"] = -1
        for i in range(250):  # longer to let it settle after the drop
            step_sim(model, data, attachment, site_id, 0, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("cube", "release", extra_data={"step": i})

        # Back away
        print("Backing away from stack...")
        move_to(
            model,
            data,
            high_stack_joints[:6],
            300,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "retreat_high",
            fullscreen_hook,
            mpc_controller=mpc_controller,
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

    # Print final cube positions
    for cube in cubes_sorted:
        final_pos = data.xpos[cube["body_id"]].copy()
        print(f"{cube['name']} final position: {final_pos}")
    print("Stacking demo complete.")

    if diag:
        output_dir = Path(args.diag_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        diag["logger"].generate_report(
            output_dir=output_dir,
            show_plots=not args.headless,
            save_plots=True,
        )

    if viewer is not None:
        print("Close the viewer to exit...")
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data, nstep=100)
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
