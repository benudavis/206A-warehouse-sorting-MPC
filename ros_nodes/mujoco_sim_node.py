#!/usr/bin/env python3
"""
MuJoCo Simulation Node for ROS2

This node runs in Docker and:
1. Loads MuJoCo scene with boxes, obstacles, baskets
2. Publishes box positions to /box_coordinates
3. Subscribes to joint trajectories and executes them in simulation
4. Publishes /joint_states from simulation
5. Handles box attachment when grasped
"""

import sys
from pathlib import Path
import numpy as np
import json
import time
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("Warning: MuJoCo not available")

# Add project root to path
sys.path.insert(0, '/ros2_ws/src_project')
sys.path.insert(0, str(Path(__file__).parent.parent))


class MuJoCoSimNode(Node):
    """MuJoCo simulation node that bridges ROS2 and MuJoCo."""

    def __init__(self):
        super().__init__('mujoco_sim')
        
        if not HAS_MUJOCO:
            self.get_logger().error('MuJoCo not available!')
            return
        
        # Parameters
        self.declare_parameter('sim_rate', 100.0)  # Hz
        self.declare_parameter('publish_box_positions', True)
        
        self.sim_rate = self.get_parameter('sim_rate').value
        self.publish_boxes = self.get_parameter('publish_box_positions').value
        
        # Load MuJoCo model
        self.load_mujoco_model()
        
        # State
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.trajectory_index = 0
        self.box_attached = None
        
        # Box positions (from MuJoCo model)
        self.box_positions = self.extract_box_positions()
        
        # Publishers
        self.joint_state_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )
        
        self.box_pub = self.create_publisher(
            String,
            '/box_coordinates',
            10
        )
        
        # Subscribers
        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/mpc/joint_trajectory',
            self.trajectory_callback,
            10
        )
        
        self.gripper_sub = self.create_subscription(
            String,
            '/gripper_events',
            self.gripper_event_callback,
            10
        )
        
        # Simulation timer
        self.sim_timer = self.create_timer(1.0 / self.sim_rate, self.sim_step)
        
        # Box position publisher timer
        if self.publish_boxes:
            self.box_timer = self.create_timer(1.0, self.publish_box_positions)
        
        self.get_logger().info('✓ MuJoCo Simulation Node ready (headless)')
        self.get_logger().info(f'  Sim rate: {self.sim_rate} Hz')
        self.get_logger().info(f'  Boxes: {len(self.box_positions)}')

    def load_mujoco_model(self):
        """Load MuJoCo model with robot, boxes, obstacles, baskets."""
        try:
            models_dir = Path("/ros2_ws/sim/models")
            
            scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
            arm_spec = mujoco.MjSpec.from_file(
                str(models_dir / "universal_robots_ur5e" / "ur5e.xml")
            )
            hand_spec = mujoco.MjSpec.from_file(
                str(models_dir / "robotiq_2f85" / "2f85.xml")
            )
            
            arm_spec.site("attachment_site").attach_body(
                hand_spec.worldbody, "hand_", ""
            )
            scene.site("robot_site").attach_body(
                arm_spec.worldbody, "arm_", ""
            )
            
            self.model = scene.compile()
            self.data = mujoco.MjData(self.model)
            
            # Find joint IDs
            self.joint_names = [
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint'
            ]
            self.joint_ids = []
            for name in self.joint_names:
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_{name}"
                )
                if jid >= 0:
                    self.joint_ids.append(jid)
            
            self.ee_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "arm_hand_pinch"
            )
            
            # Find box body IDs
            self.box_body_ids = {}
            box_names = [
                "red_1", "red_2", "red_3",
                "blue_1", "blue_2", "blue_3"
            ]
            for box_name in box_names:
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, box_name
                )
                if bid >= 0:
                    self.box_body_ids[box_name] = bid
            
            # Set initial joint positions
            home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
            if len(self.joint_ids) == 6:
                self.data.qpos[self.joint_ids] = home_joints
            
            mujoco.mj_forward(self.model, self.data)
            
            self.get_logger().info('✓ MuJoCo model loaded')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load MuJoCo model: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise

    def extract_box_positions(self):
        """Extract box positions from MuJoCo model."""
        boxes = []
        box_defs = [
            {"name": "red_1", "pos": [0.35, -0.28, 0.52], "color": "red"},
            {"name": "red_2", "pos": [0.40, -0.32, 0.52], "color": "red"},
            {"name": "red_3", "pos": [0.45, -0.28, 0.52], "color": "red"},
            {"name": "blue_1", "pos": [0.35, -0.42, 0.52], "color": "blue"},
            {"name": "blue_2", "pos": [0.40, -0.38, 0.52], "color": "blue"},
            {"name": "blue_3", "pos": [0.45, -0.42, 0.52], "color": "blue"},
        ]
        
        for box_def in box_defs:
            box_name = box_def["name"]
            if box_name in self.box_body_ids:
                bid = self.box_body_ids[box_name]
                pos = self.data.xpos[bid].copy()
                boxes.append({
                    "name": box_name,
                    "pos": pos.tolist(),
                    "color": box_def["color"]
                })
            else:
                boxes.append({
                    "name": box_name,
                    "pos": box_def["pos"],
                    "color": box_def["color"]
                })
        
        return boxes

    def trajectory_callback(self, msg):
        """Receive trajectory from MPC."""
        if len(msg.points) == 0:
            return
        
        self.current_trajectory = msg
        self.trajectory_start_time = self.get_clock().now()
        self.trajectory_index = 0
        
        self.get_logger().info(f'Received trajectory: {len(msg.points)} waypoints')

    def gripper_event_callback(self, msg):
        """Handle gripper events (grasp/release)."""
        try:
            parts = msg.data.split(':')
            action = parts[0]
            box_name = parts[1] if len(parts) > 1 else None
            
            if action == 'grasp' and box_name:
                self.box_attached = box_name
                self.get_logger().info(f'Grasped box: {box_name}')
                if box_name in self.box_body_ids:
                    self.attached_box_id = self.box_body_ids[box_name]
                    self.attachment_offset = np.array([0.0, 0.0, 0.05])
            elif action == 'release':
                if self.box_attached:
                    self.get_logger().info(f'Released box: {self.box_attached}')
                    self.box_attached = None
                    self.attached_box_id = None
                    
        except Exception as e:
            self.get_logger().error(f'Error handling gripper event: {e}')

    def sim_step(self):
        """Step simulation and publish joint states."""
        if self.current_trajectory is not None:
            self.execute_trajectory()
        
        if self.box_attached and hasattr(self, 'attached_box_id'):
            self.update_attached_box()
        
        dt = 1.0 / self.sim_rate
        mujoco.mj_step(self.model, self.data, nstep=1)
        
        self.publish_joint_state()

    def execute_trajectory(self):
        """Execute current trajectory step by step."""
        if self.current_trajectory is None:
            return
        
        now = self.get_clock().now()
        if self.trajectory_start_time is None:
            self.trajectory_start_time = now
        
        elapsed = (now - self.trajectory_start_time).nanoseconds / 1e9
        
        while self.trajectory_index < len(self.current_trajectory.points):
            point = self.current_trajectory.points[self.trajectory_index]
            waypoint_time = (
                point.time_from_start.sec +
                point.time_from_start.nanosec / 1e9
            )
            
            if elapsed >= waypoint_time:
                if len(point.positions) >= len(self.joint_ids):
                    for i, jid in enumerate(self.joint_ids):
                        if i < len(point.positions):
                            self.data.qpos[jid] = point.positions[i]
                    mujoco.mj_forward(self.model, self.data)
                self.trajectory_index += 1
            else:
                break
        
        if self.trajectory_index >= len(self.current_trajectory.points):
            self.current_trajectory = None
            self.trajectory_index = 0

    def update_attached_box(self):
        """Update position of attached box to follow end-effector."""
        if not hasattr(self, 'attached_box_id'):
            return
        
        if self.ee_site_id >= 0:
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            ee_quat = self.data.site_xquat[self.ee_site_id].copy()
            
            from scipy.spatial.transform import Rotation as R
            rot = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            offset_world = rot.apply(self.attachment_offset)
            
            box_pos = ee_pos + offset_world
            bid = self.attached_box_id
            self.data.xpos[bid] = box_pos
            self.data.xquat[bid] = ee_quat
            
            for box in self.box_positions:
                if box["name"] == self.box_attached:
                    box["pos"] = box_pos.tolist()
                    break

    def publish_joint_state(self):
        """Publish current joint state."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = self.joint_names
        
        positions = []
        velocities = []
        efforts = []
        
        for jid in self.joint_ids:
            positions.append(float(self.data.qpos[jid]))
            velocities.append(float(self.data.qvel[jid]))
            efforts.append(0.0)
        
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts
        
        self.joint_state_pub.publish(msg)

    def publish_box_positions(self):
        """Publish box positions."""
        if not self.publish_boxes:
            return
        
        for box in self.box_positions:
            if box["name"] != self.box_attached and box["name"] in self.box_body_ids:
                bid = self.box_body_ids[box["name"]]
                pos = self.data.xpos[bid].copy()
                box["pos"] = pos.tolist()
        
        msg = String()
        msg.data = json.dumps(self.box_positions, indent=2)
        self.box_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoSimNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
