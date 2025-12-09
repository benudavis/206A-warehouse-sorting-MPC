#!/usr/bin/env python3
"""
Color-based warehouse sorting demo with MPC obstacle avoidance.

Goal:
  - Pick boxes from the central region,
  - Route them around a vertical wall,
  - Place them into left/right baskets based on color,
  - Avoid collisions with the wall as much as possible.
"""

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

# Make sure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.control.inverse_kinematics import IKSolver
from src.control.mpc_controller import MPCController
from src.diagnostics import DiagnosticLogger

EE_SITE = "arm_hand_pinch"
MODELS_DIR = Path(__file__).parent.parent / "sim" / "models"

# Safety margins [m]
BASE_SAFETY_MARGIN = 0.05   # robot empty-handed
OBJECT_SAFETY_MARGIN = 0.08  # robot holding a box

# Safe zones for pick and place operations
# Boxes are around x=0.35-0.45, y=-0.28 to -0.42, z=0.52
BOX_SAFE_ZONE = {
    "center": [0.40, -0.35, 0.60],   # Center of box region, slightly above boxes
    "half_size": [0.15, 0.15, 0.20],  # Encompasses all boxes with margin
}

# Baskets are at x=-0.45, y=-0.10 (red) and y=-0.60 (blue), z=0.48
BASKET_SAFE_ZONE = {
    "center": [-0.45, -0.35, 0.65],   # Center above both baskets
    "half_size": [0.20, 0.35, 0.25],  # Covers area above both baskets
}

# Moving obstacle parameters
OBSTACLE_X_MIN = -0.25   # Near basket safe zone edge
OBSTACLE_X_MAX = 0.15    # Near box safe zone edge  
OBSTACLE_PERIOD = 8.0    # Seconds for one full back-and-forth cycle
OBSTACLE_BASE_POS = [-0.05, -0.35, 0.70]  # Base position (center of oscillation)

# Box configurations
BOXES = [
    # Red boxes → red basket
    {"name": "red_1",  "pos": [0.35, -0.28, 0.52], "size": 0.030, "rgba": [0.9, 0.1, 0.1, 1.0], "color": "red"},
    {"name": "red_2",  "pos": [0.40, -0.32, 0.52], "size": 0.028, "rgba": [0.8, 0.0, 0.0, 1.0], "color": "red"},
    {"name": "red_3",  "pos": [0.45, -0.28, 0.52], "size": 0.032, "rgba": [1.0, 0.2, 0.2, 1.0], "color": "red"},
    # Blue boxes → blue basket
    {"name": "blue_1", "pos": [0.35, -0.42, 0.52], "size": 0.030, "rgba": [0.1, 0.1, 0.9, 1.0], "color": "blue"},
    {"name": "blue_2", "pos": [0.40, -0.38, 0.52], "size": 0.028, "rgba": [0.0, 0.0, 0.8, 1.0], "color": "blue"},
    {"name": "blue_3", "pos": [0.45, -0.42, 0.52], "size": 0.032, "rgba": [0.2, 0.2, 1.0, 1.0], "color": "blue"},
]

# Basket positions (left side)
BASKETS = {
    "red":  {"pos": [-0.45, -0.10, 0.48], "rgba": [0.8, 0.2, 0.2, 0.5]},
    "blue": {"pos": [-0.45, -0.60, 0.48], "rgba": [0.2, 0.2, 0.8, 0.5]},
}

# ----------------------------------------------------------------------
# Quaternion helpers
# ----------------------------------------------------------------------


def quat_mul(q1, q2):
    """Multiply two quaternions (w, x, y, z), MuJoCo convention."""
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


def is_in_safe_zone(pos, safe_zone):
    """Check if a position is within a safe zone (axis-aligned box)."""
    center = np.array(safe_zone["center"])
    half_size = np.array(safe_zone["half_size"])
    diff = np.abs(pos - center)
    return np.all(diff <= half_size)


def get_current_safe_zone(pos):
    """Return the name of the safe zone the position is in, or None."""
    if is_in_safe_zone(pos, BOX_SAFE_ZONE):
        return "box"
    if is_in_safe_zone(pos, BASKET_SAFE_ZONE):
        return "basket"
    return None


def compute_obstacle_position(t):
    """
    Compute obstacle position at time t.
    
    The obstacle oscillates in the x direction using a sinusoidal motion.
    Returns the full [x, y, z] position.
    """
    # Sinusoidal oscillation in x
    x_amplitude = (OBSTACLE_X_MAX - OBSTACLE_X_MIN) / 2.0
    x_center = (OBSTACLE_X_MAX + OBSTACLE_X_MIN) / 2.0
    x = x_center + x_amplitude * np.sin(2.0 * np.pi * t / OBSTACLE_PERIOD)
    
    return np.array([x, OBSTACLE_BASE_POS[1], OBSTACLE_BASE_POS[2]], dtype=float)


def update_moving_obstacle(model, data, t, mpc_controller=None, obstacle_mocap_ids=None):
    """
    Update the position of the moving obstacle and its visualizations.
    Also updates the MPC obstacle position if provided.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        t: Current simulation time
        mpc_controller: Optional MPC controller to update obstacle position
        obstacle_mocap_ids: Dict with mocap IDs for obstacle and its visualizations
    """
    new_pos = compute_obstacle_position(t)
    
    # Update mocap positions
    if obstacle_mocap_ids is not None:
        for name, mocap_id in obstacle_mocap_ids.items():
            if mocap_id >= 0:
                data.mocap_pos[mocap_id] = new_pos
    
    # Update MPC obstacle
    if mpc_controller is not None and len(mpc_controller.obstacles) > 0:
        # Update the first (and only) obstacle position
        mpc_controller.obstacles[0] = (new_pos.copy(), mpc_controller.obstacles[0][1])


class FullscreenEnforcer:
    """Helper to force fullscreen viewer a few times after launch."""

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
    parser.add_argument(
        "--diagnostics", action="store_true", help="Record diagnostics logs"
    )
    parser.add_argument(
        "--diag-dir", default="data/diagnostics", help="Diagnostics output directory"
    )
    parser.add_argument(
        "--diag-interval", type=int, default=20, help="Log diagnostics every N steps"
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# World construction
# ----------------------------------------------------------------------


def build_world():
    """Build world with boxes, baskets, and a central obstacle wall."""
    scene = mujoco.MjSpec.from_file(str(MODELS_DIR / "scene.xml"))
    arm_spec = mujoco.MjSpec.from_file(
        str(MODELS_DIR / "universal_robots_ur5e" / "ur5e.xml")
    )
    hand_spec = mujoco.MjSpec.from_file(
        str(MODELS_DIR / "robotiq_2f85" / "2f85.xml")
    )

    # Attach hand to robot, robot to scene
    arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
    scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")

    # Baskets and their safety visualization
    for basket_name, cfg in BASKETS.items():
        # Actual basket
        basket = scene.worldbody.add_body()
        basket.name = f"basket_{basket_name}"
        basket.pos = cfg["pos"]
        basket_geom = basket.add_geom()
        basket_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        basket_size = [0.15, 0.15, 0.02]
        basket_geom.size = basket_size
        basket_geom.rgba = cfg["rgba"]

        # Safety visualization only (no collisions)
        basket_safety = scene.worldbody.add_body()
        basket_safety.name = f"basket_{basket_name}_safety"
        basket_safety.pos = cfg["pos"]
        basket_safety_geom = basket_safety.add_geom()
        basket_safety_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        basket_safety_geom.size = [
            basket_size[0] + BASE_SAFETY_MARGIN,
            basket_size[1] + BASE_SAFETY_MARGIN,
            basket_size[2] + BASE_SAFETY_MARGIN,
        ]
        basket_safety_geom.rgba = [0.5, 0.5, 0.5, 0.15]
        basket_safety_geom.contype = 0
        basket_safety_geom.conaffinity = 0

    # Center wall obstacle (z shorter, y longer) - MOVING obstacle (mocap body)
    obstacle = scene.worldbody.add_body()
    obstacle.name = "obstacle_center_wall"
    obstacle.pos = OBSTACLE_BASE_POS
    obstacle.mocap = True  # Make it a mocap body so we can move it
    obs_geom = obstacle.add_geom()
    obs_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    obs_geom.size = [0.01, 0.25, 0.04]  # thin wall: shorter z, longer y
    obs_geom.rgba = [0.9, 0.5, 0.1, 0.85]
    obs_geom.contype = 0  # No collision (mocap bodies can't collide normally)
    obs_geom.conaffinity = 0

    # Safety margin visualizations for wall (also mocap to move with obstacle)
    safety_viz = scene.worldbody.add_body()
    safety_viz.name = "obstacle_safety_margin"
    safety_viz.pos = OBSTACLE_BASE_POS
    safety_viz.mocap = True
    safety_margin_geom = safety_viz.add_geom()
    safety_margin_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    safety_margin_geom.size = [
        0.01 + BASE_SAFETY_MARGIN,
        0.25 + BASE_SAFETY_MARGIN,
        0.04 + BASE_SAFETY_MARGIN,
    ]
    safety_margin_geom.rgba = [1.0, 0.0, 0.0, 0.2]
    safety_margin_geom.contype = 0
    safety_margin_geom.conaffinity = 0

    holding_margin_viz = scene.worldbody.add_body()
    holding_margin_viz.name = "obstacle_holding_margin"
    holding_margin_viz.pos = OBSTACLE_BASE_POS
    holding_margin_viz.mocap = True
    holding_margin_geom = holding_margin_viz.add_geom()
    holding_margin_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    holding_margin_geom.size = [
        0.01 + OBJECT_SAFETY_MARGIN,
        0.25 + OBJECT_SAFETY_MARGIN,
        0.04 + OBJECT_SAFETY_MARGIN,
    ]
    holding_margin_geom.rgba = [1.0, 1.0, 0.0, 0.12]
    holding_margin_geom.contype = 0
    holding_margin_geom.conaffinity = 0

    # Safe zone visualizations (transparent boxes showing where pick/place can happen)
    # Box safe zone (green)
    box_zone_viz = scene.worldbody.add_body()
    box_zone_viz.name = "box_safe_zone"
    box_zone_viz.pos = BOX_SAFE_ZONE["center"]
    box_zone_geom = box_zone_viz.add_geom()
    box_zone_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    box_zone_geom.size = BOX_SAFE_ZONE["half_size"]
    box_zone_geom.rgba = [0.0, 0.8, 0.0, 0.08]  # Green, very transparent
    box_zone_geom.contype = 0
    box_zone_geom.conaffinity = 0

    # Basket safe zone (cyan)
    basket_zone_viz = scene.worldbody.add_body()
    basket_zone_viz.name = "basket_safe_zone"
    basket_zone_viz.pos = BASKET_SAFE_ZONE["center"]
    basket_zone_geom = basket_zone_viz.add_geom()
    basket_zone_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    basket_zone_geom.size = BASKET_SAFE_ZONE["half_size"]
    basket_zone_geom.rgba = [0.0, 0.8, 0.8, 0.08]  # Cyan, very transparent
    basket_zone_geom.contype = 0
    basket_zone_geom.conaffinity = 0

    # Boxes
    for cfg in BOXES:
        box = scene.worldbody.add_body()
        box.name = cfg["name"]
        box.pos = cfg["pos"]
        geom = box.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [cfg["size"]] * 3
        geom.rgba = cfg["rgba"]
        geom.mass = 0.05
        geom.friction = [2.0, 0.01, 0.0005]
        box.add_freejoint()

    # Pre-allocated trajectory visualization geoms
    MAX_TRAJ_WAYPOINTS = 12
    JOINT_COLORS = [
        [1.0, 0.0, 0.0, 0.7],  # shoulder pan
        [1.0, 0.5, 0.0, 0.7],  # shoulder lift
        [1.0, 1.0, 0.0, 0.7],  # elbow
        [0.0, 1.0, 0.0, 0.7],  # wrist 1
        [0.0, 0.0, 1.0, 0.7],  # wrist 2
        [0.5, 0.0, 1.0, 0.7],  # wrist 3
    ]

    # Joint trajectory spheres
    for joint_idx in range(6):
        for waypoint_idx in range(MAX_TRAJ_WAYPOINTS):
            marker_body = scene.worldbody.add_body()
            marker_body.name = f"traj_joint{joint_idx}_wp{waypoint_idx}"
            marker_body.pos = [0, 0, -10]
            marker_body.mocap = True

            marker_geom = marker_body.add_geom()
            marker_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            marker_geom.size = [0.015, 0.0, 0.0]
            marker_geom.rgba = JOINT_COLORS[joint_idx]
            marker_geom.contype = 0
            marker_geom.conaffinity = 0

    MAX_TRAJ_SEGMENTS = MAX_TRAJ_WAYPOINTS - 1

    # Joint trajectory line segments (capsules)
    for joint_idx in range(6):
        for seg_idx in range(MAX_TRAJ_SEGMENTS):
            line_body = scene.worldbody.add_body()
            line_body.name = f"traj_line_j{joint_idx}_s{seg_idx}"
            line_body.pos = [0, 0, -10]
            line_body.mocap = True

            line_geom = line_body.add_geom()
            line_geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            line_geom.size = [0.004, 0.01, 0.0]
            line_geom.rgba = JOINT_COLORS[joint_idx]
            line_geom.contype = 0
            line_geom.conaffinity = 0

    # End-effector trajectory (white, thicker)
    for waypoint_idx in range(MAX_TRAJ_WAYPOINTS):
        ee_marker = scene.worldbody.add_body()
        ee_marker.name = f"traj_ee_wp{waypoint_idx}"
        ee_marker.pos = [0, 0, -10]
        ee_marker.mocap = True

        ee_geom = ee_marker.add_geom()
        ee_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        ee_geom.size = [0.020, 0.0, 0.0]
        ee_geom.rgba = [1.0, 1.0, 1.0, 1.0]
        ee_geom.contype = 0
        ee_geom.conaffinity = 0

    for seg_idx in range(MAX_TRAJ_SEGMENTS):
        ee_line = scene.worldbody.add_body()
        ee_line.name = f"traj_ee_line{seg_idx}"
        ee_line.pos = [0, 0, -10]
        ee_line.mocap = True

        ee_line_geom = ee_line.add_geom()
        ee_line_geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        ee_line_geom.size = [0.008, 0.01, 0.0]
        ee_line_geom.rgba = [1.0, 1.0, 1.0, 1.0]
        ee_line_geom.contype = 0
        ee_line_geom.conaffinity = 0

    # Compile model
    model = scene.compile()
    model.opt.timestep = 0.0002  # physics timestep

    # Add joint damping
    for i in range(6):
        model.dof_damping[i] = 10.0

    data = mujoco.MjData(model)
    return model, data


# ----------------------------------------------------------------------
# Collision checking & trajectory visualization
# ----------------------------------------------------------------------


def check_collision(model, data, obstacle_geom_ids):
    """Check if any arm link is colliding with specified obstacle geoms."""
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2

        if geom1 in obstacle_geom_ids or geom2 in obstacle_geom_ids:
            body1 = model.geom_bodyid[geom1]
            body2 = model.geom_bodyid[geom2]
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)

            if body1_name and ("arm_" in body1_name or (body2_name and "arm_" in body2_name)):
                if "box" not in body1_name and "box" not in body2_name:
                    return True, (body1_name, body2_name)

    return False, None


def _mat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def visualize_mpc_trajectory(viewer, model, data_viz, q_trajectory, site_id, data_main=None):
    """
    Visualize MPC trajectory for all 6 joint positions and EE.

    If q_trajectory is None, hides all markers.
    """
    data_mocap = data_main if data_main is not None else data_viz
    MAX_WAYPOINTS = 12
    MAX_SEGMENTS = MAX_WAYPOINTS - 1

    def hide_all():
        for joint_idx in range(6):
            for wp_idx in range(MAX_WAYPOINTS):
                try:
                    body_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, f"traj_joint{joint_idx}_wp{wp_idx}"
                    )
                    if body_id >= 0:
                        mocap_id = model.body_mocapid[body_id]
                        if mocap_id >= 0:
                            data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
                except Exception:
                    pass
            for seg_idx in range(MAX_SEGMENTS):
                try:
                    body_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, f"traj_line_j{joint_idx}_s{seg_idx}"
                    )
                    if body_id >= 0:
                        mocap_id = model.body_mocapid[body_id]
                        if mocap_id >= 0:
                            data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
                except Exception:
                    pass

        for wp_idx in range(MAX_WAYPOINTS):
            try:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_wp{wp_idx}"
                )
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except Exception:
                pass

        for seg_idx in range(MAX_SEGMENTS):
            try:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_line{seg_idx}"
                )
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except Exception:
                pass

    if q_trajectory is None or len(q_trajectory) < 2:
        hide_all()
        return

    # Joint body IDs
    joint_body_names = [
        "arm_shoulder_link",
        "arm_upper_arm_link",
        "arm_forearm_link",
        "arm_wrist_1_link",
        "arm_wrist_2_link",
        "arm_wrist_3_link",
    ]
    joint_body_ids = []
    for name in joint_body_names:
        try:
            joint_body_ids.append(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            )
        except Exception:
            joint_body_ids.append(-1)

    all_joint_positions = []
    ee_positions = []

    for q in q_trajectory:
        data_viz.qpos[:6] = q
        mujoco.mj_forward(model, data_viz)

        joint_positions = []
        for bid in joint_body_ids:
            if bid >= 0:
                joint_positions.append(data_viz.xpos[bid].copy())
            else:
                joint_positions.append(np.array([0.0, 0.0, -10.0]))
        all_joint_positions.append(joint_positions)
        ee_positions.append(data_viz.site_xpos[site_id].copy())

    all_joint_positions = np.array(all_joint_positions)  # (N, 6, 3)
    ee_positions = np.array(ee_positions)                # (N, 3)
    n_waypoints = min(len(q_trajectory), MAX_WAYPOINTS)

    # Joint markers
    for joint_idx in range(6):
        # Waypoint spheres
        for wp_idx in range(n_waypoints):
            pos = all_joint_positions[wp_idx, joint_idx, :]
            try:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"traj_joint{joint_idx}_wp{wp_idx}"
                )
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = pos
            except Exception:
                pass
        # Hide unused
        for wp_idx in range(n_waypoints, MAX_WAYPOINTS):
            try:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"traj_joint{joint_idx}_wp{wp_idx}"
                )
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except Exception:
                pass

        # Line segments
        for seg_idx in range(min(n_waypoints - 1, MAX_SEGMENTS)):
            p1 = all_joint_positions[seg_idx, joint_idx, :]
            p2 = all_joint_positions[seg_idx + 1, joint_idx, :]
            center = (p1 + p2) / 2.0
            direction = p2 - p1
            length = np.linalg.norm(direction)
            try:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"traj_line_j{joint_idx}_s{seg_idx}"
                )
                if body_id < 0:
                    continue
                mocap_id = model.body_mocapid[body_id]
                if length < 1e-6 or mocap_id < 0:
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
                    continue

                direction = direction / length
                z_axis = direction
                if abs(z_axis[0]) < 0.9:
                    x_axis = np.cross([1, 0, 0], z_axis)
                else:
                    x_axis = np.cross([0, 1, 0], z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                R = np.column_stack([x_axis, y_axis, z_axis])
                quat = _mat_to_quat(R)

                data_mocap.mocap_pos[mocap_id] = center
                data_mocap.mocap_quat[mocap_id] = quat

                geom_id = model.body_geomadr[body_id]
                if geom_id >= 0:
                    model.geom_size[geom_id] = [0.004, length / 2.0, 0.0]
            except Exception:
                pass

        # Hide unused segments
        for seg_idx in range(n_waypoints - 1, MAX_SEGMENTS):
            try:
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"traj_line_j{joint_idx}_s{seg_idx}"
                )
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except Exception:
                pass

    # EE markers
    for wp_idx in range(n_waypoints):
        pos = ee_positions[wp_idx]
        try:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_wp{wp_idx}"
            )
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = pos
        except Exception:
            pass
    for wp_idx in range(n_waypoints, MAX_WAYPOINTS):
        try:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_wp{wp_idx}"
            )
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
        except Exception:
            pass

    for seg_idx in range(min(n_waypoints - 1, MAX_SEGMENTS)):
        p1 = ee_positions[seg_idx]
        p2 = ee_positions[seg_idx + 1]
        center = (p1 + p2) / 2.0
        direction = p2 - p1
        length = np.linalg.norm(direction)
        try:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_line{seg_idx}"
            )
            if body_id < 0:
                continue
            mocap_id = model.body_mocapid[body_id]
            if length < 1e-6 or mocap_id < 0:
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
                continue

            direction = direction / length
            z_axis = direction
            if abs(z_axis[0]) < 0.9:
                x_axis = np.cross([1, 0, 0], z_axis)
            else:
                x_axis = np.cross([0, 1, 0], z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            R = np.column_stack([x_axis, y_axis, z_axis])
            quat = _mat_to_quat(R)

            data_mocap.mocap_pos[mocap_id] = center
            data_mocap.mocap_quat[mocap_id] = quat

            geom_id = model.body_geomadr[body_id]
            if geom_id >= 0:
                model.geom_size[geom_id] = [0.008, length / 2.0, 0.0]
        except Exception:
            pass

    for seg_idx in range(n_waypoints - 1, MAX_SEGMENTS):
        try:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_line{seg_idx}"
            )
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
        except Exception:
            pass


# ----------------------------------------------------------------------
# Simulation stepping & motion primitives
# ----------------------------------------------------------------------


def settle(model, data, steps=400):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook=None,
              sim_time=None, obstacle_mocap_ids=None, mpc_controller=None):
    """Step simulation with welded box attachment and moving obstacle."""
    # Update moving obstacle position
    if sim_time is not None and obstacle_mocap_ids is not None:
        update_moving_obstacle(model, data, sim_time, mpc_controller, obstacle_mocap_ids)
    
    # Gripper command
    data.ctrl[6] = grip

    # Welded attachment for box (if active)
    if attachment["active"] and attachment["qadr"] >= 0:
        grip_pos = data.site_xpos[site_id].copy()
        ee_body_id = model.site_bodyid[site_id]
        grip_quat = data.xquat[ee_body_id].copy()

        qadr = attachment["qadr"]
        dadr = attachment["dadr"]
        box_offset = attachment["offset"]
        q_rel = attachment["q_rel"]

        data.qpos[qadr : qadr + 3] = grip_pos + box_offset
        data.qpos[qadr + 3 : qadr + 7] = quat_mul(grip_quat, q_rel)

        if dadr >= 0:
            data.qvel[dadr : dadr + 6] = 0.0

    # Integrate physics for one control interval: 250 * 0.0002 = 0.05 s
    mujoco.mj_step(model, data, nstep=250)

    # Display
    if viewer is not None:
        viewer.sync()
        if fullscreen_hook is not None:
            fullscreen_hook()
    
    # Return updated time
    dt_step = 250 * model.opt.timestep
    return (sim_time + dt_step) if sim_time is not None else None


def move_to_waypoints(
    model,
    data,
    waypoints,
    steps_per_segment,
    grip,
    attachment,
    site_id,
    viewer,
    diag=None,
    phase="move",
    fullscreen_hook=None,
    mpc_controller: MPCController | None = None,
    obstacle_geom_ids=None,
    collision_log=None,
    viz_data=None,
    sim_time=0.0,
    obstacle_mocap_ids=None,
):
    """
    Move through multiple waypoints continuously.
    
    MPC plans toward the FINAL waypoint but executes smoothly through intermediate waypoints.
    Returns updated simulation time.
    """
    if len(waypoints) == 0:
        return sim_time
    
    final_target = np.asarray(waypoints[-1][:6], dtype=float)
    total_steps = len(waypoints) * steps_per_segment
    collision_count = 0
    
    original_margin = None
    if mpc_controller is not None and attachment.get("active", False):
        original_margin = mpc_controller.safety_margin
        mpc_controller.safety_margin = OBJECT_SAFETY_MARGIN
        print(f"    Increased safety margin: {mpc_controller.safety_margin:.3f}m (holding object)")
    
    if mpc_controller is not None:
        data_scratch = mujoco.MjData(model)
        
        if total_steps > 50:
            print(f"    🔄 Continuous MPC through {len(waypoints)} waypoints ({total_steps} steps)")
        
        for i in range(total_steps):
            current_q = data.qpos[:6].copy()
            current_dq = data.qvel[:6].copy()
            current_state = np.concatenate([current_q, current_dq])
            
            # Always plan toward FINAL waypoint (not intermediate)
            # This allows MPC to see the full path and plan accordingly
            try:
                q_cmd, q_traj = mpc_controller.compute_control(
                    current_state=current_state,
                    target_state=final_target,
                    model=model,
                    data_scratch=data_scratch,
                    site_id=site_id,
                )
            except Exception as e:
                if i == 0:
                    print(f"  ⚠️  MPC failed: {e}")
                # Fallback: interpolate toward final target
                alpha = (i + 1) / total_steps
                q_cmd = (1.0 - alpha) * data.qpos[:6].copy() + alpha * final_target
                q_traj = None
            
            if viz_data is not None and q_traj is not None:
                visualize_mpc_trajectory(viewer, model, viz_data, q_traj, site_id, data_main=data)
            
            data.ctrl[:6] = q_cmd
            sim_time = step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook,
                                sim_time, obstacle_mocap_ids, mpc_controller)
            
            if obstacle_geom_ids is not None:
                is_collision, bodies = check_collision(model, data, obstacle_geom_ids)
                if is_collision:
                    collision_count += 1
                    if collision_count == 1:
                        print(f"    ⚠️  COLLISION in '{phase}': {bodies[0]} <-> {bodies[1]}")
                    if collision_log is not None:
                        collision_log.append({
                            "phase": phase, "step": i, "bodies": bodies,
                            "position": data.qpos[:6].copy(),
                        })
            
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", phase, extra_data={"step": i})
        
        if collision_count > 0:
            print(f"    Total collisions: {collision_count}")
        
        if viz_data is not None:
            visualize_mpc_trajectory(viewer, model, viz_data, None, site_id, data_main=data)
    else:
        # Smooth interpolation through waypoints
        for seg_idx, target_joints in enumerate(waypoints):
            target = np.asarray(target_joints[:6], dtype=float)
            start = data.qpos[:6].copy()
            
            for i in range(steps_per_segment):
                alpha = (i + 1) / steps_per_segment
                alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                desired = (1.0 - alpha_smooth) * start + alpha_smooth * target
                data.ctrl[:6] = desired
                sim_time = step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook,
                                    sim_time, obstacle_mocap_ids, mpc_controller)
                
                if diag and i % diag["interval"] == 0:
                    diag["logger"].log_state("box", phase, extra_data={"step": seg_idx * steps_per_segment + i})
    
    if original_margin is not None and mpc_controller is not None:
        mpc_controller.safety_margin = original_margin
    
    return sim_time


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
    mpc_controller: MPCController | None = None,
    obstacle_geom_ids=None,
    collision_log=None,
    viz_data=None,
    sim_time=0.0,
    obstacle_mocap_ids=None,
):
    """Single-target move (wrapper for backwards compatibility). Returns updated sim_time."""
    return move_to_waypoints(
        model, data, [target], steps, grip, attachment, site_id, viewer,
        diag, phase, fullscreen_hook, mpc_controller, obstacle_geom_ids,
        collision_log, viz_data, sim_time, obstacle_mocap_ids
    )


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


def main():
    args = parse_args()

    if args.headless or args.diagnostics:
        import matplotlib

        matplotlib.use("Agg", force=True)

    print("=" * 70)
    print("COLOR-BASED WAREHOUSE SORTING DEMO")
    print("=" * 70)
    print("Layout: Boxes (center) → Wall (middle) → Baskets (left)")
    print("Red boxes → red basket | Blue boxes → blue basket")
    print("-" * 70)
    print("SAFE ZONES (pick/place operations only allowed in these regions):")
    print(f"  Box zone:    center={BOX_SAFE_ZONE['center']}, half_size={BOX_SAFE_ZONE['half_size']}")
    print(f"  Basket zone: center={BASKET_SAFE_ZONE['center']}, half_size={BASKET_SAFE_ZONE['half_size']}")
    print("=" * 70)

    model, data = build_world()

    # Get mocap IDs for moving obstacle and its visualizations
    obstacle_mocap_ids = {}
    for name in ["obstacle_center_wall", "obstacle_safety_margin", "obstacle_holding_margin"]:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                obstacle_mocap_ids[name] = mocap_id
        except Exception:
            pass
    print(f"\nMoving obstacle mocap IDs: {obstacle_mocap_ids}")
    print(f"Obstacle oscillates in x from {OBSTACLE_X_MIN:.2f} to {OBSTACLE_X_MAX:.2f} with period {OBSTACLE_PERIOD:.1f}s")

    # Rotate shoulder to face the scene
    shoulder_joint = model.jnt("arm_shoulder_pan_joint")
    model.key_qpos[0][shoulder_joint.qposadr] += np.pi
    model.key_ctrl[0][shoulder_joint.dofadr] += np.pi
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Simulation time tracking
    sim_time = 0.0

    # Let everything settle
    settle(model, data, steps=500)
    sim_time += 500 * model.opt.timestep * 250  # Approximate time for settle

    # IK for end-effector
    ik = IKSolver(model, data, site_name=EE_SITE)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    # Separate data for visualization FK (never rendered directly)
    viz_data = mujoco.MjData(model)

    # Initialize MPC
    print("\nInitializing MPC...")
    mpc_controller = MPCController(
        n_joints=6,
        horizon=10,   # Fixed horizon: 10 * 0.05s = 0.5s lookahead
        dt=0.05,      # Control period (one step_sim call)
    )
    mpc_controller.safety_margin = BASE_SAFETY_MARGIN
    print(f"  TRUE receding horizon: MPC solves at EVERY control step")
    print(f"  Horizon fixed at H={mpc_controller.horizon} (planning {mpc_controller.horizon * mpc_controller.dt:.2f}s ahead)")
    print(f"  MPC will solve at EVERY control step (true receding horizon)")
    print(f"  Horizon always H={mpc_controller.horizon}, planning {mpc_controller.horizon * mpc_controller.dt:.2f}s ahead")

    # Add only the wall as an obstacle to MPC & for collision detection
    obstacle_geom_ids = []
    try:
        obs_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_center_wall"
        )
        if obs_body_id >= 0:
            obs_pos = data.xpos[obs_body_id].copy()
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == obs_body_id:
                    obs_size = model.geom_size[gid].copy()
                    mpc_controller.add_obstacle(obs_pos, obs_size)
                    obstacle_geom_ids.append(gid)
                    print(f"  Added wall obstacle for MPC and collision checks (geom_id={gid})")
                    break
    except Exception as e:
        print(f"  Warning: could not add wall obstacle: {e}")

    # Initialize which arm links to use in collision checks
    mpc_controller.initialize_link_bodies(model)

    print(f"✓ MPC ready with {len(mpc_controller.obstacles)} obstacles")
    print("\nObstacle Safety Margins:")
    print(f"  Wall half-size: from XML.")
    print(f"  Soft cost margin (no box): {BASE_SAFETY_MARGIN*100:.1f} cm")
    print(f"  Soft cost margin (with box): {OBJECT_SAFETY_MARGIN*100:.1f} cm")
    print("=" * 70)

    # Viewer
    viewer = None
    fullscreen_hook = None
    if not args.headless and HAS_VIEWER:
        viewer = mujoco.viewer.launch_passive(model, data)
        fullscreen_hook = FullscreenEnforcer(viewer)
        fullscreen_hook()

    # Diagnostics
    diag = None
    if args.diagnostics:
        diag_logger = DiagnosticLogger(model, data, site_name=EE_SITE)
        diag = {"logger": diag_logger, "interval": max(1, args.diag_interval)}

    # Collect box info (free joints)
    boxes = []
    for cfg in BOXES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["name"])
        if body_id < 0:
            continue
        jnt_adr = model.body_jntadr[body_id]
        qadr = int(model.jnt_qposadr[jnt_adr])
        dadr = int(model.jnt_dofadr[jnt_adr])

        boxes.append(
            {
                "name": cfg["name"],
                "body_id": body_id,
                "joint_id": jnt_adr,
                "qadr": qadr,
                "dadr": dadr,
                "size": cfg["size"],
                "color": cfg["color"],
            }
        )

    # Basket info
    baskets = {}
    for color in ["red", "blue"]:
        basket_name = f"basket_{color}"
        basket_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, basket_name)
        basket_geom_id = None
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == basket_body:
                basket_geom_id = gid
                break
        thickness = model.geom_size[basket_geom_id][2] if basket_geom_id is not None else 0.02
        baskets[color] = {
            "body_id": basket_body,
            "geom_id": basket_geom_id,
            "thickness": thickness,
        }

    # Common poses
    down_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    mid_clear = np.array([0.50, -0.35, 0.85], dtype=float)
    mid_clear_joints, _ = ik.solve(
        mid_clear,
        target_quat=down_quat,
        max_iterations=500,
        tolerance=0.01,
    )
    home_joints = data.qpos[:6].copy()

    # Attachment state
    attachment = {
        "active": False,
        "qadr": -1,
        "dadr": -1,
        "body_id": -1,
        "offset": np.zeros(3, dtype=float),
        "q_rel": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    }

    collision_log = []

    # ------------------------------------------------------------------
    # Main sorting loop
    # ------------------------------------------------------------------
    for idx, box in enumerate(boxes):
        box_name = box["name"]
        box_color = box["color"]
        box_body = box["body_id"]
        box_qadr = box["qadr"]
        box_dadr = box["dadr"]
        box_size = box["size"]

        print("\n" + "=" * 70)
        print(f"[{idx+1}/{len(boxes)}] {box_name} ({box_color}) → {box_color} basket")
        print("=" * 70)

        # Current box COM
        box_pos = data.xpos[box_body].copy()
        print(f"  Box position: {box_pos}")

        # Skip if box fell off table
        if box_pos[2] < 0.40:
            print(f"  ⚠️  Box {box_name} fell off table (z={box_pos[2]:.3f}), skipping.")
            continue

        # Check if box is in the box safe zone (required for picking)
        if not is_in_safe_zone(box_pos, BOX_SAFE_ZONE):
            print(f"  ⚠️  Box {box_name} is outside box safe zone, skipping.")
            print(f"      Box pos: {box_pos}")
            print(f"      Safe zone center: {BOX_SAFE_ZONE['center']}, half_size: {BOX_SAFE_ZONE['half_size']}")
            continue

        # Approach and contact poses
        above = box_pos.copy()
        above[2] += 0.12

        contact = box_pos.copy()
        contact[2] += box_size * 0.5  # move toward top face
        contact[2] -= 0.005           # small downward offset

        above_joints, above_success = ik.solve(
            above, target_quat=down_quat, max_iterations=500, tolerance=0.003
        )
        contact_joints, contact_success = ik.solve(
            contact, target_quat=down_quat, max_iterations=800, tolerance=0.002
        )

        if not above_success:
            print(f"  ⚠️  IK failed for 'above' pose at {above}")
        if not contact_success:
            print(f"  ⚠️  IK failed for 'contact' pose at {contact}")

        # Make sure gripper is open
        print("Opening gripper...")
        for _ in range(50):
            sim_time = step_sim(model, data, attachment, site_id, grip=0, viewer=viewer, fullscreen_hook=fullscreen_hook,
                                sim_time=sim_time, obstacle_mocap_ids=obstacle_mocap_ids, mpc_controller=mpc_controller)

        # Move above box (no MPC, direct)
        print("Approaching above box...")
        sim_time = move_to(
            model,
            data,
            above_joints[:6],
            steps=300,
            grip=0,
            attachment=attachment,
            site_id=site_id,
            viewer=viewer,
            diag=diag,
            phase="approach_above",
            fullscreen_hook=fullscreen_hook,
            mpc_controller=None,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
            sim_time=sim_time,
            obstacle_mocap_ids=obstacle_mocap_ids,
        )

        # Lower to contact (no MPC, precise)
        print("Lowering to box...")
        sim_time = move_to(
            model,
            data,
            contact_joints[:6],
            steps=300,
            grip=0,
            attachment=attachment,
            site_id=site_id,
            viewer=viewer,
            diag=diag,
            phase="lower",
            fullscreen_hook=fullscreen_hook,
            mpc_controller=None,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
            sim_time=sim_time,
            obstacle_mocap_ids=obstacle_mocap_ids,
        )

        # Let physics settle
        for _ in range(50):
            sim_time = step_sim(model, data, attachment, site_id, grip=0, viewer=viewer, fullscreen_hook=fullscreen_hook,
                                sim_time=sim_time, obstacle_mocap_ids=obstacle_mocap_ids, mpc_controller=mpc_controller)

        # Check distance EE ↔ box COM
        GRASP_THRESH = 0.050
        grip_pos = data.site_xpos[site_id].copy()
        box_now = data.xpos[box_body].copy()
        dist = float(np.linalg.norm(grip_pos - box_now))
        print(f"  Distance to box: {dist*1000:.1f} mm (threshold {GRASP_THRESH*1000:.1f} mm)")

        # Optional refinement
        if dist >= GRASP_THRESH:
            refine_joints, success = ik.solve(
                box_now,
                target_quat=down_quat,
                max_iterations=200,
                tolerance=0.0015,
            )
            if success:
                print("  Refining grasp pose...")
                sim_time = move_to(
                    model,
                    data,
                    refine_joints[:6],
                    steps=150,
                    grip=0,
                    attachment=attachment,
                    site_id=site_id,
                    viewer=viewer,
                    diag=diag,
                    phase="refine_grasp",
                    fullscreen_hook=fullscreen_hook,
                    mpc_controller=None,
                    obstacle_geom_ids=obstacle_geom_ids,
                    collision_log=collision_log,
                    viz_data=viz_data,
                    sim_time=sim_time,
                    obstacle_mocap_ids=obstacle_mocap_ids,
                )
                for _ in range(30):
                    sim_time = step_sim(
                        model,
                        data,
                        attachment,
                        site_id,
                        grip=0,
                        viewer=viewer,
                        fullscreen_hook=fullscreen_hook,
                        sim_time=sim_time,
                        obstacle_mocap_ids=obstacle_mocap_ids,
                        mpc_controller=mpc_controller,
                    )
                grip_pos = data.site_xpos[site_id].copy()
                box_now = data.xpos[box_body].copy()
                dist = float(np.linalg.norm(grip_pos - box_now))
                print(f"  Distance after refine: {dist*1000:.1f} mm")
            else:
                print("  IK refinement failed.")

        # Attach box if close enough
        if dist < GRASP_THRESH:
            # Verify pick is happening in box safe zone
            current_zone = get_current_safe_zone(grip_pos)
            if current_zone != "box":
                print(f"  ⚠️  Pick operation at {grip_pos} is outside box safe zone!")
                print(f"      Current zone: {current_zone}")
            else:
                print(f"  ✓ Pick operation is within box safe zone")

            ee_body_id = model.site_bodyid[site_id]
            grip_quat = data.xquat[ee_body_id].copy()
            box_quat = data.xquat[box_body].copy()
            q_rel = quat_mul(quat_conj(grip_quat), box_quat)

            # Snap box COM exactly to gripper site
            data.qpos[box_qadr : box_qadr + 3] = grip_pos
            data.qpos[box_qadr + 3 : box_qadr + 7] = box_quat
            mujoco.mj_forward(model, data)

            attachment["active"] = True
            attachment["qadr"] = box_qadr
            attachment["dadr"] = box_dadr
            attachment["body_id"] = box_body
            attachment["offset"] = np.zeros(3, dtype=float)
            attachment["q_rel"] = q_rel

            print(f"✓ {box_name} attached (distance {dist:.3f} m)")
        else:
            attachment["active"] = False
            attachment["qadr"] = -1
            attachment["dadr"] = -1
            attachment["body_id"] = -1
            attachment["offset"] = np.zeros(3, dtype=float)
            attachment["q_rel"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            print(f"✗ {box_name} out of reach (distance {dist:.3f} m); skipping")
            continue

        # Close gripper
        print("Closing gripper...")
        for i in range(150):
            sim_time = step_sim(model, data, attachment, site_id, grip=255, viewer=viewer, fullscreen_hook=fullscreen_hook,
                                sim_time=sim_time, obstacle_mocap_ids=obstacle_mocap_ids, mpc_controller=mpc_controller)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", "close", extra_data={"step": i})

        if not attachment["active"]:
            print("Attachment failed; skipping placement for this box.")
            continue

        # Basket placement target
        basket_info = baskets[box_color]
        basket_pos = data.xpos[basket_info["body_id"]].copy()
        basket_top_z = basket_pos[2] + basket_info["thickness"]

        drop_margin = 0.005
        place_pos = basket_pos.copy()
        place_pos[2] = basket_top_z + box_size + drop_margin

        # Check if placement position is in the basket safe zone
        if not is_in_safe_zone(place_pos, BASKET_SAFE_ZONE):
            print(f"  ⚠️  Basket placement position is outside basket safe zone!")
            print(f"      Place pos: {place_pos}")
            print(f"      Safe zone center: {BASKET_SAFE_ZONE['center']}, half_size: {BASKET_SAFE_ZONE['half_size']}")
            # Adjust place position to be within safe zone if possible
            # For now, we'll continue but warn the user
            print(f"      Continuing anyway, but placement may be unsafe.")

        approach_ee = place_pos.copy()
        approach_ee[2] += 0.10

        approach_joints, _ = ik.solve(
            approach_ee,
            target_quat=down_quat,
            max_iterations=500,
            tolerance=0.01,
        )
        place_joints, _ = ik.solve(
            place_pos,
            target_quat=down_quat,
            max_iterations=600,
            tolerance=0.005,
        )

        # Transport with MPC (ONE continuous motion through waypoints)
        print(f"Transporting {box_name} → {box_color} basket...")
        sim_time = move_to_waypoints(
            model,
            data,
            waypoints=[mid_clear_joints[:6], approach_joints[:6]],  # Two waypoints
            steps_per_segment=300,
            grip=255,
            attachment=attachment,
            site_id=site_id,
            viewer=viewer,
            diag=diag,
            phase="transport",
            fullscreen_hook=fullscreen_hook,
            mpc_controller=mpc_controller,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
            sim_time=sim_time,
            obstacle_mocap_ids=obstacle_mocap_ids,
        )

        # Precise lowering into basket (no MPC)
        sim_time = move_to(
            model,
            data,
            place_joints[:6],
            steps=200,
            grip=255,
            attachment=attachment,
            site_id=site_id,
            viewer=viewer,
            diag=diag,
            phase="pre_place",
            fullscreen_hook=fullscreen_hook,
            mpc_controller=None,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
            sim_time=sim_time,
            obstacle_mocap_ids=obstacle_mocap_ids,
        )

        # Release box
        print(f"Releasing into {box_color} basket...")
        
        # Verify release is happening in basket safe zone
        release_pos = data.site_xpos[site_id].copy()
        current_zone = get_current_safe_zone(release_pos)
        if current_zone != "basket":
            print(f"  ⚠️  Release operation at {release_pos} is outside basket safe zone!")
            print(f"      Current zone: {current_zone}")
        else:
            print(f"  ✓ Release operation is within basket safe zone")
        
        attachment["active"] = False
        attachment["qadr"] = -1
        attachment["dadr"] = -1
        attachment["body_id"] = -1
        attachment["q_rel"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for i in range(200):
            sim_time = step_sim(model, data, attachment, site_id, grip=0, viewer=viewer, fullscreen_hook=fullscreen_hook,
                                sim_time=sim_time, obstacle_mocap_ids=obstacle_mocap_ids, mpc_controller=mpc_controller)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", "release", extra_data={"step": i})

        # Retreat with MPC (ONE continuous motion through waypoints)
        print("Retreating...")
        sim_time = move_to_waypoints(
            model,
            data,
            waypoints=[approach_joints[:6], mid_clear_joints[:6]],  # Back through waypoints
            steps_per_segment=200,
            grip=0,
            attachment=attachment,
            site_id=site_id,
            viewer=viewer,
            diag=diag,
            phase="retreat",
            fullscreen_hook=fullscreen_hook,
            mpc_controller=mpc_controller,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
            sim_time=sim_time,
            obstacle_mocap_ids=obstacle_mocap_ids,
        )

    # Return home using MPC
    print("\nReturning to home position...")
    sim_time = move_to(
        model,
        data,
        home_joints,
        steps=300,
        grip=0,
        attachment=attachment,
        site_id=site_id,
        viewer=viewer,
        diag=diag,
        phase="return_home",
        fullscreen_hook=fullscreen_hook,
        mpc_controller=mpc_controller,
        obstacle_geom_ids=obstacle_geom_ids,
        collision_log=collision_log,
        viz_data=viz_data,
        sim_time=sim_time,
        obstacle_mocap_ids=obstacle_mocap_ids,
    )

    # Summary
    print("\n" + "=" * 70)
    print("✓ SORTING COMPLETE")
    print("=" * 70)

    for box in boxes:
        final_pos = data.xpos[box["body_id"]].copy()
        print(f"{box['name']} ({box['color']}): final position {final_pos}")

    print("\n" + "=" * 70)
    print("COLLISION REPORT")
    print("=" * 70)
    if not collision_log:
        print("✓ No collisions detected.")
    else:
        print(f"⚠️  Total collisions: {len(collision_log)}")
        from collections import defaultdict

        by_phase = defaultdict(int)
        for entry in collision_log:
            by_phase[entry["phase"]] += 1

        print("\nCollisions by phase:")
        for phase, count in sorted(by_phase.items()):
            print(f"  {phase}: {count}")
    print("=" * 70)

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
        print(f"(Obstacle continues to move - watch it oscillate!)")
        try:
            while viewer.is_running():
                # Keep updating obstacle position
                update_moving_obstacle(model, data, sim_time, mpc_controller, obstacle_mocap_ids)
                mujoco.mj_step(model, data, nstep=250)
                sim_time += 250 * model.opt.timestep
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
