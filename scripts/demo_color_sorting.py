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

# Safety margin constants (meters) - used by MPC and visualization
BASE_SAFETY_MARGIN = 0.05   # 5cm clearance when robot is empty
OBJECT_SAFETY_MARGIN = 0.08  # 8cm clearance when robot holds an object

# Box configurations: All boxes start in the middle, easily reachable
# Red boxes go to left basket, Blue boxes go to right basket
BOXES = [
    # Red boxes (go to left basket)
    {"name": "red_1",  "pos": [0.35, -0.28, 0.52], "size": 0.030, "rgba": [0.9, 0.1, 0.1, 1.0], "color": "red"},
    {"name": "red_2",  "pos": [0.40, -0.32, 0.52], "size": 0.028, "rgba": [0.8, 0.0, 0.0, 1.0], "color": "red"},
    {"name": "red_3",  "pos": [0.45, -0.28, 0.52], "size": 0.032, "rgba": [1.0, 0.2, 0.2, 1.0], "color": "red"},
    # Blue boxes (go to right basket)
    {"name": "blue_1", "pos": [0.35, -0.42, 0.52], "size": 0.030, "rgba": [0.1, 0.1, 0.9, 1.0], "color": "blue"},
    {"name": "blue_2", "pos": [0.40, -0.38, 0.52], "size": 0.028, "rgba": [0.0, 0.0, 0.8, 1.0], "color": "blue"},
    {"name": "blue_3", "pos": [0.45, -0.42, 0.52], "size": 0.032, "rgba": [0.2, 0.2, 1.0, 1.0], "color": "blue"},
]

# Basket positions: left side, further from boxes
BASKETS = {
    "red":  {"pos": [-0.45, -0.10, 0.48], "rgba": [0.8, 0.2, 0.2, 0.5]},   # Left side for red
    "blue": {"pos": [-0.45, -0.60, 0.48], "rgba": [0.2, 0.2, 0.8, 0.5]},   # Left side for blue
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
    """Build world with boxes, baskets, and obstacles."""
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
        basket_geom = basket.add_geom()
        basket_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        basket_geom.size = [0.15, 0.15, 0.02]
        basket_geom.rgba = basket_config["rgba"]

    # Vertical wall obstacle between boxes and baskets (moved further from boxes)
    obstacle = scene.worldbody.add_body()
    obstacle.name = "obstacle_center_wall"
    obstacle.pos = [-0.15, -0.35, 0.70]  # Wall positioned between boxes and baskets
    obs_geom = obstacle.add_geom()
    obs_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    obs_geom.size = [0.01, 0.12, 0.08]  # Actual obstacle: 2cm thick, 24cm wide, 16cm tall
    obs_geom.rgba = [0.9, 0.5, 0.1, 0.85]  # Orange, semi-transparent
    
    # Visualize safety margins (MPC keeps robot these distances away)
    # 
    # Layered Safety Zones:
    #   [Orange Box]  = Actual obstacle (2cm × 24cm × 16cm)
    #   [Red Zone]    = +5cm margin all around (robot empty-handed)
    #   [Yellow Zone] = +8cm margin all around (robot holding object)
    #
    # MPC Cost Function: Penalty increases as robot enters these zones
    #
    safety_viz = scene.worldbody.add_body()
    safety_viz.name = "obstacle_safety_margin"
    safety_viz.pos = [-0.15, -0.35, 0.70]  # Same center as obstacle
    safety_margin_geom = safety_viz.add_geom()
    safety_margin_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    # Add base safety margin to each dimension
    safety_margin_geom.size = [0.01 + BASE_SAFETY_MARGIN, 0.12 + BASE_SAFETY_MARGIN, 0.08 + BASE_SAFETY_MARGIN]
    safety_margin_geom.rgba = [1.0, 0.0, 0.0, 0.2]  # Red, semi-transparent
    safety_margin_geom.contype = 0  # No collision
    safety_margin_geom.conaffinity = 0
    
    # Visualize extended margin when holding objects
    holding_margin_viz = scene.worldbody.add_body()
    holding_margin_viz.name = "obstacle_holding_margin"
    holding_margin_viz.pos = [-0.15, -0.35, 0.70]  # Same center
    holding_margin_geom = holding_margin_viz.add_geom()
    holding_margin_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    # Add object safety margin when holding objects
    holding_margin_geom.size = [0.01 + OBJECT_SAFETY_MARGIN, 0.12 + OBJECT_SAFETY_MARGIN, 0.08 + OBJECT_SAFETY_MARGIN]
    holding_margin_geom.rgba = [1.0, 1.0, 0.0, 0.12]  # Yellow, very transparent
    holding_margin_geom.contype = 0  # No collision
    holding_margin_geom.conaffinity = 0

    # Boxes
    for box_config in BOXES:
        box = scene.worldbody.add_body()
        box.name = box_config["name"]
        box.pos = box_config["pos"]
        geom = box.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [box_config["size"]] * 3
        geom.rgba = box_config["rgba"]
        geom.mass = 0.05
        geom.friction = [2.0, 0.01, 0.0005]  # Higher friction for stable grasping
        box.add_freejoint()

    # Pre-allocate trajectory visualization geoms
    # For each joint in the arm (6 joints), create visualization markers
    MAX_TRAJ_WAYPOINTS = 12  # Show up to 12 waypoints along trajectory
    JOINT_NAMES = [
        "arm_shoulder_pan_joint",
        "arm_shoulder_lift_joint", 
        "arm_elbow_joint",
        "arm_wrist_1_joint",
        "arm_wrist_2_joint",
        "arm_wrist_3_joint"
    ]
    
    # Colors for each joint (rainbow gradient)
    JOINT_COLORS = [
        [1.0, 0.0, 0.0, 0.7],  # Red - shoulder pan
        [1.0, 0.5, 0.0, 0.7],  # Orange - shoulder lift
        [1.0, 1.0, 0.0, 0.7],  # Yellow - elbow
        [0.0, 1.0, 0.0, 0.7],  # Green - wrist 1
        [0.0, 0.0, 1.0, 0.7],  # Blue - wrist 2
        [0.5, 0.0, 1.0, 0.7],  # Purple - wrist 3
    ]
    
    # Create spheres for each joint at each waypoint
    for joint_idx in range(6):
        for waypoint_idx in range(MAX_TRAJ_WAYPOINTS):
            marker_body = scene.worldbody.add_body()
            marker_body.name = f"traj_joint{joint_idx}_wp{waypoint_idx}"
            marker_body.pos = [0, 0, -10]  # Hide initially
            marker_body.mocap = True
            
            marker_geom = marker_body.add_geom()
            marker_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            marker_geom.size = [0.015, 0, 0]  # 15mm radius sphere
            marker_geom.rgba = JOINT_COLORS[joint_idx]
            marker_geom.contype = 0
            marker_geom.conaffinity = 0
    
    # Create line segments connecting waypoints for each joint
    MAX_TRAJ_SEGMENTS = 11  # MAX_WAYPOINTS - 1
    for joint_idx in range(6):
        for seg_idx in range(MAX_TRAJ_SEGMENTS):
            line_body = scene.worldbody.add_body()
            line_body.name = f"traj_line_j{joint_idx}_s{seg_idx}"
            line_body.pos = [0, 0, -10]
            line_body.mocap = True
            
            line_geom = line_body.add_geom()
            line_geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            line_geom.size = [0.004, 0.01, 0]  # Thin line
            line_geom.rgba = JOINT_COLORS[joint_idx]
            line_geom.contype = 0
            line_geom.conaffinity = 0
    
    # Create END-EFFECTOR trajectory visualization (bright white, thicker)
    for waypoint_idx in range(MAX_TRAJ_WAYPOINTS):
        ee_marker = scene.worldbody.add_body()
        ee_marker.name = f"traj_ee_wp{waypoint_idx}"
        ee_marker.pos = [0, 0, -10]
        ee_marker.mocap = True
        
        ee_geom = ee_marker.add_geom()
        ee_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        ee_geom.size = [0.020, 0, 0]  # 20mm radius - BIGGER than joints
        ee_geom.rgba = [1.0, 1.0, 1.0, 1.0]  # Bright white, fully opaque
        ee_geom.contype = 0
        ee_geom.conaffinity = 0
    
    for seg_idx in range(MAX_TRAJ_SEGMENTS):
        ee_line = scene.worldbody.add_body()
        ee_line.name = f"traj_ee_line{seg_idx}"
        ee_line.pos = [0, 0, -10]
        ee_line.mocap = True
        
        ee_line_geom = ee_line.add_geom()
        ee_line_geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        ee_line_geom.size = [0.008, 0.01, 0]  # THICKER than joint lines
        ee_line_geom.rgba = [1.0, 1.0, 1.0, 1.0]  # Bright white
        ee_line_geom.contype = 0
        ee_line_geom.conaffinity = 0

    model = scene.compile()
    model.opt.timestep = 0.0002  # Smaller timestep for stability

    # Add joint damping for smooth, stable control
    for i in range(6):  # First 6 joints (arm)
        model.dof_damping[i] = 10.0

    data = mujoco.MjData(model)
    return model, data


def check_collision(model, data, obstacle_geom_ids):
    """Check if any arm links are colliding with obstacles."""
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        
        # Check if either geometry is an obstacle
        if geom1 in obstacle_geom_ids or geom2 in obstacle_geom_ids:
            # Get body IDs for both geometries
            body1 = model.geom_bodyid[geom1]
            body2 = model.geom_bodyid[geom2]
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
            
            # Check if it's an arm collision (not gripper-box contact)
            if body1_name and ("arm_" in body1_name or body2_name and "arm_" in body2_name):
                if "box" not in body1_name and "box" not in body2_name:
                    return True, (body1_name, body2_name)
    
    return False, None


def visualize_mpc_trajectory(viewer, model, data_viz, q_trajectory, site_id, data_main=None):
    """
    Visualize MPC trajectory showing all 6 joint positions along the planned path.
    
    Args:
        data_viz: Scratch data for computing FK (won't be rendered)
        data_main: Main simulation data (actually rendered) - mocap will be set here
    """
    # Use main data for mocap if provided, otherwise use viz data
    data_mocap = data_main if data_main is not None else data_viz
    
    MAX_WAYPOINTS = 12
    MAX_SEGMENTS = 11
    
    if q_trajectory is None or len(q_trajectory) < 2:
        # Hide all trajectory markers (joints + EE)
        for joint_idx in range(6):
            for wp_idx in range(MAX_WAYPOINTS):
                try:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_joint{joint_idx}_wp{wp_idx}")
                    if body_id >= 0:
                        mocap_id = model.body_mocapid[body_id]
                        if mocap_id >= 0:
                            data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
                except:
                    pass
            
            for seg_idx in range(MAX_SEGMENTS):
                try:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_line_j{joint_idx}_s{seg_idx}")
                    if body_id >= 0:
                        mocap_id = model.body_mocapid[body_id]
                        if mocap_id >= 0:
                            data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
                except:
                    pass
        
        # Hide EE trajectory
        for wp_idx in range(MAX_WAYPOINTS):
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_wp{wp_idx}")
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except:
                pass
        
        for seg_idx in range(MAX_SEGMENTS):
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_line{seg_idx}")
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except:
                pass
        
        return
    
    # Get body IDs for each joint
    joint_body_names = [
        "arm_shoulder_link",
        "arm_upper_arm_link",
        "arm_forearm_link",
        "arm_wrist_1_link",
        "arm_wrist_2_link",
        "arm_wrist_3_link"
    ]
    
    joint_body_ids = []
    for name in joint_body_names:
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            joint_body_ids.append(bid)
        except:
            joint_body_ids.append(-1)
    
    # Compute positions for all joints + EE site at each waypoint
    all_joint_positions = []  # Shape: (n_waypoints, 6_joints, 3_xyz)
    ee_positions = []  # Shape: (n_waypoints, 3_xyz) - actual gripper position
    
    for q in q_trajectory:
        data_viz.qpos[:6] = q
        mujoco.mj_forward(model, data_viz)
        
        # Get joint body positions
        joint_positions = []
        for body_id in joint_body_ids:
            if body_id >= 0:
                pos = data_viz.xpos[body_id].copy()
                joint_positions.append(pos)
            else:
                joint_positions.append(np.array([0, 0, -10]))
        
        all_joint_positions.append(joint_positions)
        
        # Get actual EE site position (gripper pinch point)
        ee_pos = data_viz.site_xpos[site_id].copy()
        ee_positions.append(ee_pos)
    
    all_joint_positions = np.array(all_joint_positions)  # (n_waypoints, 6, 3)
    ee_positions = np.array(ee_positions)  # (n_waypoints, 3)
    
    # Compute total path length (using actual end-effector site)
    path_length = sum(np.linalg.norm(ee_positions[i+1] - ee_positions[i]) 
                      for i in range(len(ee_positions)-1))
    
    print(f"    → MPC planned {len(q_trajectory)} waypoints, EE path length: {path_length:.3f}m")
    print(f"    → Visualizing ALL 6 joint trajectories + End-Effector:")
    print(f"       🔴 Red=Shoulder Pan | 🟠 Orange=Shoulder Lift | 🟡 Yellow=Elbow")
    print(f"       🟢 Green=Wrist 1 | 🔵 Blue=Wrist 2 | 🟣 Purple=Wrist 3")
    print(f"       ⚪ BRIGHT WHITE = End-Effector (gripper) - THE MAIN PATH!")
    
    # Visualize each joint's trajectory
    n_waypoints = min(len(q_trajectory), MAX_WAYPOINTS)
    
    for joint_idx in range(6):
        # Place spheres at each waypoint for this joint
        for wp_idx in range(n_waypoints):
            pos = all_joint_positions[wp_idx, joint_idx, :]
            
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_joint{joint_idx}_wp{wp_idx}")
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = pos
            except:
                pass
        
        # Hide unused waypoint markers
        for wp_idx in range(n_waypoints, MAX_WAYPOINTS):
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_joint{joint_idx}_wp{wp_idx}")
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except:
                pass
        
        # Draw lines connecting consecutive waypoints for this joint
        for seg_idx in range(n_waypoints - 1):
            p1 = all_joint_positions[seg_idx, joint_idx, :]
            p2 = all_joint_positions[seg_idx + 1, joint_idx, :]
            
            center = (p1 + p2) / 2.0
            direction = p2 - p1
            length = np.linalg.norm(direction)
            
            if length < 1e-6:
                # Hide zero-length segments
                try:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_line_j{joint_idx}_s{seg_idx}")
                    if body_id >= 0:
                        mocap_id = model.body_mocapid[body_id]
                        if mocap_id >= 0:
                            data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
                except:
                    pass
                continue
            
            direction = direction / length
            
            # Rotation matrix to align capsule with direction
            z_axis = direction
            if abs(z_axis[0]) < 0.9:
                x_axis = np.cross([1, 0, 0], z_axis)
            else:
                x_axis = np.cross([0, 1, 0], z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            rot_mat = np.column_stack([x_axis, y_axis, z_axis])
            quat = _mat_to_quat(rot_mat)
            
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_line_j{joint_idx}_s{seg_idx}")
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = center
                        data_mocap.mocap_quat[mocap_id] = quat
                    
                    # Update capsule length
                    geom_id = model.body_geomadr[body_id]
                    if geom_id >= 0:
                        model.geom_size[geom_id] = [0.004, length / 2.0, 0]
            except:
                pass
        
        # Hide unused line segments
        for seg_idx in range(n_waypoints - 1, MAX_SEGMENTS):
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_line_j{joint_idx}_s{seg_idx}")
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except:
                pass
    
    # ===== VISUALIZE END-EFFECTOR TRAJECTORY (BRIGHT WHITE) =====
    # Place spheres at each EE waypoint
    for wp_idx in range(n_waypoints):
        pos = ee_positions[wp_idx]
        
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_wp{wp_idx}")
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = pos
        except:
            pass
    
    # Hide unused EE waypoints
    for wp_idx in range(n_waypoints, MAX_WAYPOINTS):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_wp{wp_idx}")
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
        except:
            pass
    
    # Draw lines connecting consecutive EE waypoints
    for seg_idx in range(n_waypoints - 1):
        p1 = ee_positions[seg_idx]
        p2 = ee_positions[seg_idx + 1]
        
        center = (p1 + p2) / 2.0
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_line{seg_idx}")
                if body_id >= 0:
                    mocap_id = model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
            except:
                pass
            continue
        
        direction = direction / length
        
        # Rotation matrix
        z_axis = direction
        if abs(z_axis[0]) < 0.9:
            x_axis = np.cross([1, 0, 0], z_axis)
        else:
            x_axis = np.cross([0, 1, 0], z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        rot_mat = np.column_stack([x_axis, y_axis, z_axis])
        quat = _mat_to_quat(rot_mat)
        
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_line{seg_idx}")
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = center
                    data_mocap.mocap_quat[mocap_id] = quat
                
                # Update capsule length
                geom_id = model.body_geomadr[body_id]
                if geom_id >= 0:
                    model.geom_size[geom_id] = [0.008, length / 2.0, 0]
        except:
            pass
    
    # Hide unused EE line segments
    for seg_idx in range(n_waypoints - 1, MAX_SEGMENTS):
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"traj_ee_line{seg_idx}")
            if body_id >= 0:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id >= 0:
                    data_mocap.mocap_pos[mocap_id] = [0, 0, -10]
        except:
            pass


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


def settle(model, data, steps=400):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook=None):
    """Step simulation with welded attachment handling."""
    # Gripper command
    data.ctrl[6] = grip

    # Welded attachment of box to gripper
    if attachment["active"] and attachment["qadr"] >= 0:
        grip_pos = data.site_xpos[site_id].copy()
        ee_body_id = model.site_bodyid[site_id]
        grip_quat = data.xquat[ee_body_id].copy()

        qadr = attachment["qadr"]
        dadr = attachment["dadr"]
        box_offset = attachment["offset"]
        q_rel = attachment["q_rel"]

        # Position (box COM rigidly attached to gripper site + offset)
        data.qpos[qadr: qadr + 3] = grip_pos + box_offset
        # Orientation: q_box = q_grip * q_rel
        box_quat = quat_mul(grip_quat, q_rel)
        data.qpos[qadr + 3: qadr + 7] = box_quat

        # Zero free-joint velocities for stability
        if dadr >= 0:
            data.qvel[dadr: dadr + 6] = 0.0

    # Advance physics: 250 * 0.0002 = 0.05 s control interval
    mujoco.mj_step(model, data, nstep=250)

    # Viewer sync
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
    obstacle_geom_ids=None,
    collision_log=None,
    viz_data=None,
):
    """
    Move to target joint configuration.

    If mpc_controller is provided, use MPC to generate obstacle-aware joint
    commands. Otherwise use straight-line interpolation in joint space.
    """
    start = data.qpos[:6].copy()

    collision_count = 0
    
    # Adjust safety margin if holding an object
    original_margin = None
    if mpc_controller is not None and attachment.get("active", False):
        original_margin = mpc_controller.safety_margin
        mpc_controller.safety_margin = OBJECT_SAFETY_MARGIN  # 12cm clearance when holding object
        print(f"    Using increased safety margin: {mpc_controller.safety_margin}m (holding object)")
    
    if mpc_controller is not None:
        # MPC with receding horizon: replan periodically, execute trajectory smoothly
        data_scratch = mujoco.MjData(model)
        
        i = 0
        traj_buffer = None
        traj_index = 0
        prev_cmd = start.copy()
        steps_since_plan = 999  # Force replan on first iteration
        REPLAN_INTERVAL = 8  # Replan every 8 steps (regardless of horizon)
        BLEND_FACTOR = 0.3  # Smooth blending between old and new commands
        
        while i < steps:
            # Replan when buffer is empty OR when interval is reached
            if traj_buffer is None or traj_index >= len(traj_buffer) or steps_since_plan >= REPLAN_INTERVAL:
                current_q = data.qpos[:6].copy()
                current_dq = data.qvel[:6].copy()
                current_state = np.concatenate([current_q, current_dq])
                
                try:
                    # Time the MPC computation (this is where pauses occur!)
                    mpc_start = time.time()
                    _, q_trajectory = mpc_controller.compute_control(
                        current_state=current_state,
                        target_state=target[:6],
                        model=model,
                        data_scratch=data_scratch,
                        site_id=site_id,
                    )
                    mpc_time = time.time() - mpc_start
                    if i == 0:  # Only print on first replan
                        print(f"    ⏱️  MPC computation took {mpc_time:.2f}s")
                    
                    # Visualize EVERY MPC plan (updates as robot moves!)
                    if viz_data is not None:
                        visualize_mpc_trajectory(viewer, model, viz_data, q_trajectory, site_id, data_main=data)
                        if i == 0:
                            print(f"    📍 Visualization will update every {REPLAN_INTERVAL} steps (receding horizon)")
                    
                    # Store trajectory for execution (skip first point which is current state)
                    traj_buffer = q_trajectory[1:]
                    traj_index = 0
                    steps_since_plan = 0
                    
                except Exception as e:
                    print(f"  ⚠️  MPC failed at step {i}: {e}")
                    traj_buffer = None
            
            # Get command from trajectory or fallback to interpolation
            if traj_buffer is not None and traj_index < len(traj_buffer):
                raw_cmd = traj_buffer[traj_index]
                traj_index += 1
            else:
                # Fallback: simple interpolation
                alpha = (i + 1) / steps
                raw_cmd = (1.0 - alpha) * start + alpha * target
            
            # Smooth blending with previous command to avoid discontinuities
            smoothed_cmd = (1 - BLEND_FACTOR) * prev_cmd + BLEND_FACTOR * raw_cmd
            
            data.ctrl[:6] = smoothed_cmd
            prev_cmd = smoothed_cmd.copy()
            
            step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook)
            
            # Check for collisions
            if obstacle_geom_ids is not None:
                is_collision, bodies = check_collision(model, data, obstacle_geom_ids)
                if is_collision:
                    collision_count += 1
                    if collision_count == 1:  # Report first collision
                        print(f"    ⚠️  COLLISION detected at step {i}: {bodies[0]} <-> {bodies[1]}")
                    if collision_log is not None:
                        collision_log.append({
                            "phase": phase,
                            "step": i,
                            "bodies": bodies,
                            "position": data.qpos[:6].copy()
                        })
            
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", phase, extra_data={"step": i})
            
            i += 1
            steps_since_plan += 1
        
        if collision_count > 0:
            print(f"    Total collisions in phase '{phase}': {collision_count}")
        
        # Clear visualization after move completes
        if viz_data is not None:
            visualize_mpc_trajectory(viewer, model, viz_data, None, site_id, data_main=data)
    else:
        # Smooth joint-space interpolation (no obstacle awareness)
        for i in range(steps):
            alpha = (i + 1) / steps
            alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
            desired = (1.0 - alpha_smooth) * start + alpha_smooth * target
            data.ctrl[:6] = desired

            step_sim(model, data, attachment, site_id, grip, viewer, fullscreen_hook)
            
            # Check for collisions
            if obstacle_geom_ids is not None:
                is_collision, bodies = check_collision(model, data, obstacle_geom_ids)
                if is_collision:
                    collision_count += 1
                    if collision_count == 1:  # Report first collision
                        print(f"    ⚠️  COLLISION detected at step {i}: {bodies[0]} <-> {bodies[1]}")
                    if collision_log is not None:
                        collision_log.append({
                            "phase": phase,
                            "step": i,
                            "bodies": bodies,
                            "position": data.qpos[:6].copy()
                        })

            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", phase, extra_data={"alpha": float(alpha)})
        
        if collision_count > 0:
            print(f"    Total collisions in phase '{phase}': {collision_count}")
    
    # Restore original safety margin if it was changed
    if original_margin is not None and mpc_controller is not None:
        mpc_controller.safety_margin = original_margin


def main():
    args = parse_args()

    if args.headless or args.diagnostics:
        import matplotlib
        matplotlib.use("Agg", force=True)

    print("=" * 70)
    print("COLOR-BASED WAREHOUSE SORTING DEMO")
    print("=" * 70)
    print("Layout:")
    print("  Baskets (top/bottom) ← Boxes (far side) → Obstacle (middle)")
    print("  Red boxes → Top basket | Blue boxes → Bottom basket")
    print("=" * 70)

    model, data = build_world()

    # Flip shoulder to face scene
    model.key_qpos[0][model.jnt("arm_shoulder_pan_joint").qposadr] += np.pi
    model.key_ctrl[0][model.jnt("arm_shoulder_pan_joint").dofadr] += np.pi
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Let everything settle
    settle(model, data, steps=500)

    ik = IKSolver(model, data, site_name=EE_SITE)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    
    # Create scratch data for visualization (never touch the real sim state!)
    viz_data = mujoco.MjData(model)

    # Initialize MPC
    print("\nInitializing MPC controller with obstacle avoidance...")
    # dt must match control loop: 250 * model.opt.timestep = 0.05
    mpc_controller = MPCController(
        n_joints=6,
        horizon=10,
        dt=0.05,
    )
    
    # Set safety margin to match visualization
    mpc_controller.safety_margin = BASE_SAFETY_MARGIN

    # Add obstacles to MPC and collect geometry IDs for collision detection
    obstacle_geom_ids = []
    for obstacle_name in ["obstacle_center_wall"]:
        try:
            obs_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obstacle_name)
            if obs_body_id >= 0:
                obs_pos = data.xpos[obs_body_id].copy()
                for gid in range(model.ngeom):
                    if model.geom_bodyid[gid] == obs_body_id:
                        obs_size = model.geom_size[gid].copy()
                        mpc_controller.add_obstacle(obs_pos, obs_size)
                        obstacle_geom_ids.append(gid)  # Track for collision detection
                        print(f"  Added obstacle: {obstacle_name} (geom_id={gid})")
                        break
        except Exception as e:
            print(f"  Warning: Could not add obstacle {obstacle_name}: {e}")

    # Initialize which arm links to use in collision checks
    mpc_controller.initialize_link_bodies(model)

    print(f"MPC ready with {len(mpc_controller.obstacles)} obstacles")
    print("\nObstacle Safety Margins Visualized:")
    print(f"  🟧 Orange Box = Actual obstacle (hard collision)")
    print(f"  🟥 Red Zone = Base safety margin (5cm) - MPC avoids this")
    print(f"  🟨 Yellow Zone = Extended margin (8cm) - used when holding objects")
    print("=" * 70)

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
        baskets[color] = {
            "body_id": basket_body,
            "geom_id": basket_geom_id,
            "thickness": model.geom_size[basket_geom_id][2] if basket_geom_id is not None else 0.02,
        }

    # Common poses
    down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    mid_clear = np.array([0.50, -0.35, 0.85])  # clear position high above workspace
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
        "offset": np.zeros(3),
        "q_rel": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    }

    # Collision tracking
    collision_log = []
    
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

        # Current box COM (tracks even if box moved!)
        box_pos = data.xpos[box_body].copy()
        print(f"  Box {box_name} current position: {box_pos}")
        print(f"  Box original position: {BOXES[idx]['pos']}")
        
        # Check if box fell off table or moved too far
        if box_pos[2] < 0.40:  # Below table height
            print(f"  ⚠️  Box {box_name} fell off table (z={box_pos[2]:.3f}m), skipping")
            continue
        
        # Waypoints for picking
        above = box_pos.copy()
        above[2] += 0.12  # 12 cm above box center

        # Target contact position: slightly above the box COM (toward top face)
        contact = box_pos.copy()
        contact[2] += box_size * 0.5  # move toward top face
        contact[2] -= 0.005           # small downward offset so gripper "sits" on box

        above_joints, above_success = ik.solve(
            above, target_quat=down_quat, max_iterations=500, tolerance=0.003
        )
        contact_joints, contact_success = ik.solve(
            contact, target_quat=down_quat, max_iterations=800, tolerance=0.002
        )
        
        if not above_success:
            print(f"  ⚠️  IK failed for 'above' position at {above}")
        if not contact_success:
            print(f"  ⚠️  IK failed for 'contact' position at {contact}")

        # Ensure gripper is fully open before approaching
        print("Opening gripper...")
        for _ in range(50):
            step_sim(model, data, attachment, site_id, 0, viewer, fullscreen_hook)

        # 1) Move above box (simple interpolation)
        print("Moving above box...")
        move_to(
            model,
            data,
            above_joints[:6],
            300,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "approach_above",
            fullscreen_hook,
            mpc_controller=None,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
        )

        # 2) Lower to contact position
        print("Lowering to box...")
        move_to(
            model,
            data,
            contact_joints[:6],
            300,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "lower",
            fullscreen_hook,
            mpc_controller=None,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
        )

        # Let physics settle after lowering
        for _ in range(50):
            step_sim(model, data, attachment, site_id, 0, viewer, fullscreen_hook)

        # Recompute positions after settling
        GRASP_THRESH = 0.050  # 5 cm threshold (increased from 1.5cm for reliability)
        grip_pos = data.site_xpos[site_id].copy()
        box_now = data.xpos[box_body].copy()
        dist = np.linalg.norm(grip_pos - box_now)

        print(f"  Distance to box: {dist*1000:.1f}mm (threshold: {GRASP_THRESH*1000:.1f}mm)")
        print(f"  Gripper at: {grip_pos}")
        print(f"  Box at: {box_now}")

        if dist >= GRASP_THRESH:
            # Optional: one more local IK refinement directly to box COM
            refine_joints, success = ik.solve(
                box_now,
                target_quat=down_quat,
                max_iterations=200,
                tolerance=0.0015,
            )
            if success:
                print("  Refining grasp pose to better align with box center...")
                move_to(
                    model,
                    data,
                    refine_joints[:6],
                    150,
                    0,
                    attachment,
                    site_id,
                    viewer,
                    diag,
                    "refine_grasp",
                    fullscreen_hook,
                    mpc_controller=None,
                    obstacle_geom_ids=obstacle_geom_ids,
                    collision_log=collision_log,
                    viz_data=viz_data,
                )
                for _ in range(30):
                    step_sim(model, data, attachment, site_id, 0, viewer, fullscreen_hook)
                grip_pos = data.site_xpos[site_id].copy()
                box_now = data.xpos[box_body].copy()
                dist = np.linalg.norm(grip_pos - box_now)
                print(f"  Distance after refine: {dist*1000:.1f}mm")
            else:
                print(f"  ⚠️  IK refinement failed, keeping original distance {dist*1000:.1f}mm")

        if dist < GRASP_THRESH:
            ee_body_id = model.site_bodyid[site_id]
            grip_quat = data.xquat[ee_body_id].copy()
            box_quat = data.xquat[box_body].copy()
            q_rel = quat_mul(quat_conj(grip_quat), box_quat)

            # SNAP BOX COM EXACTLY TO GRIPPER SITE
            data.qpos[box_qadr: box_qadr + 3] = grip_pos
            data.qpos[box_qadr + 3: box_qadr + 7] = box_quat
            mujoco.mj_forward(model, data)

            attachment["active"] = True
            attachment["qadr"] = box_qadr
            attachment["dadr"] = box_dadr
            attachment["body_id"] = box_body
            attachment["offset"] = np.zeros(3)  # COM exactly at EE site
            attachment["q_rel"] = q_rel

            print(f"✓ {box_name} attached to gripper (distance {dist:.3f}m, snapped to center)")
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

        # Basket placement
        basket_info = baskets[box_color]
        basket_pos = data.xpos[basket_info["body_id"]].copy()
        basket_top_z = basket_pos[2] + basket_info["thickness"]

        drop_margin = 0.005  # 5 mm above basket floor
        place_pos = basket_pos.copy()
        place_pos[2] = basket_top_z + box_size + drop_margin

        # With offset == 0, EE site goes exactly to desired COM place position
        place_ee = place_pos.copy()
        approach_ee = place_ee.copy()
        approach_ee[2] += 0.10

        approach_joints, _ = ik.solve(
            approach_ee, target_quat=down_quat, max_iterations=500, tolerance=0.01
        )
        place_joints, _ = ik.solve(
            place_ee, target_quat=down_quat, max_iterations=600, tolerance=0.005
        )

        # Transport to basket with MPC (obstacle-aware)
        print(f"Transporting to {box_color} basket...")
        move_to(
            model,
            data,
            mid_clear_joints[:6],
            300,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "lift_high",
            fullscreen_hook,
            mpc_controller=mpc_controller,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
        )
        move_to(
            model,
            data,
            approach_joints[:6],
            300,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "approach_basket",
            fullscreen_hook,
            mpc_controller=mpc_controller,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
        )

        # Precise lowering into basket (no MPC for final centimeters)
        move_to(
            model,
            data,
            place_joints[:6],
            200,
            255,
            attachment,
            site_id,
            viewer,
            diag,
            "pre_place",
            fullscreen_hook,
            mpc_controller=None,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
        )

        # Release and open gripper
        print(f"Releasing into {box_color} basket...")
        attachment["active"] = False
        attachment["qadr"] = -1
        attachment["dadr"] = -1
        attachment["body_id"] = -1
        attachment["q_rel"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for i in range(200):
            step_sim(model, data, attachment, site_id, 0, viewer, fullscreen_hook)
            if diag and i % diag["interval"] == 0:
                diag["logger"].log_state("box", "release", extra_data={"step": i})

        # Retreat with MPC (stay away from obstacle)
        print("Retreating from basket...")
        move_to(
            model,
            data,
            approach_joints[:6],
            200,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "retreat_from_basket",
            fullscreen_hook,
            mpc_controller=mpc_controller,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
        )
        move_to(
            model,
            data,
            mid_clear_joints[:6],
            200,
            0,
            attachment,
            site_id,
            viewer,
            diag,
            "retreat_clear",
            fullscreen_hook,
            mpc_controller=mpc_controller,
            obstacle_geom_ids=obstacle_geom_ids,
            collision_log=collision_log,
            viz_data=viz_data,
        )

    # Go home at end with MPC
    print("\nReturning to home position...")
    move_to(
        model,
        data,
        home_joints,
        300,
        0,
        attachment,
        site_id,
        viewer,
        diag,
        "return_home",
        fullscreen_hook,
        mpc_controller=mpc_controller,
        obstacle_geom_ids=obstacle_geom_ids,
        collision_log=collision_log,
        viz_data=viz_data,
    )

    print("\n" + "=" * 70)
    print("✓ SORTING COMPLETE!")
    print("=" * 70)

    for box in boxes:
        final_pos = data.xpos[box["body_id"]].copy()
        print(f"{box['name']} ({box['color']}): {final_pos}")
    
    # Collision summary
    print("\n" + "=" * 70)
    print("COLLISION REPORT")
    print("=" * 70)
    if len(collision_log) == 0:
        print("✓ No collisions detected!")
    else:
        print(f"⚠️  Total collisions: {len(collision_log)}")
        # Group by phase
        from collections import defaultdict
        by_phase = defaultdict(int)
        for entry in collision_log:
            by_phase[entry["phase"]] += 1
        
        print("\nCollisions by phase:")
        for phase, count in sorted(by_phase.items()):
            print(f"  {phase}: {count} collisions")
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
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data, nstep=250)
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
