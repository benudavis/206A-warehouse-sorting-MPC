#!/usr/bin/env python3
"""
Task Planner Node - Complete Pick & Place Automation

Implements full pick-place cycle for warehouse sorting:
- Red boxes → red basket
- Blue boxes → blue basket
- Avoids wall obstacle
"""

import sys
from pathlib import Path
from enum import Enum
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import String
from builtin_interfaces.msg import Duration

# Add project root to path
sys.path.insert(0, '/ros2_ws/src_project')

from control.inverse_kinematics import IKSolver
import mujoco


class TaskState(Enum):
    """Task states for pick-and-place."""
    IDLE = 0
    MOVING_TO_BOX = 1
    LOWERING = 2
    GRASPING = 3
    LIFTING = 4
    MOVING_TO_BASKET = 5
    PLACING = 6
    RETURNING_HOME = 7


class TaskPlannerNode(Node):
    """Automated pick-and-place task sequencer."""

    def __init__(self):
        super().__init__('task_planner')
        
        # Publishers
        self.gripper_event_pub = self.create_publisher(
            String,
            '/gripper_events',
            10
        )
        
        self.status_pub = self.create_publisher(
            String,
            '/task_status',
            10
        )
        
        # Trajectory action client
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # State
        self.current_joints = np.zeros(6)
        self.current_state = TaskState.IDLE
        
        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Box definitions (from demo)
        self.boxes = [
            {"name": "red_1", "pos": [0.35, -0.28, 0.52], "size": 0.030, "color": "red"},
            {"name": "red_2", "pos": [0.40, -0.32, 0.52], "size": 0.028, "color": "red"},
            {"name": "red_3", "pos": [0.45, -0.28, 0.52], "size": 0.032, "color": "red"},
            {"name": "blue_1", "pos": [0.35, -0.42, 0.52], "size": 0.030, "color": "blue"},
            {"name": "blue_2", "pos": [0.40, -0.38, 0.52], "size": 0.028, "color": "blue"},
            {"name": "blue_3", "pos": [0.45, -0.42, 0.52], "size": 0.032, "color": "blue"},
        ]
        
        # Basket positions
        self.baskets = {
            "red": [-0.45, -0.10, 0.48],
            "blue": [-0.45, -0.60, 0.48],
        }
        
        # Setup IK solver (using MuJoCo model for IK only)
        self.setup_ik_solver()
        
        # Home position
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        
        # Start automation after delay
        self.create_timer(5.0, self.start_automation)
        
        self.get_logger().info('✓ Task Planner ready - will start automation in 5 seconds')

    def setup_ik_solver(self):
        """Initialize IK solver with MuJoCo model."""
        try:
            # Build MuJoCo model for IK
            models_dir = Path("/ros2_ws/sim/models")
            scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
            arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
            hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))
            
            # Attach
            arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
            scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")
            
            # Compile
            model = scene.compile()
            data = mujoco.MjData(model)
            
            # Initialize IK solver
            self.ik_solver = IKSolver(model, data, site_name="arm_hand_pinch")
            self.mujoco_model = model
            self.mujoco_data = data
            
            self.get_logger().info('✓ IK solver initialized')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize IK solver: {e}')
            self.ik_solver = None

    def joint_callback(self, msg):
        """Update current joint positions."""
        if len(msg.position) >= 6:
            self.current_joints = np.array(msg.position[:6])

    def start_automation(self):
        """Start the automated pick-place sequence."""
        self.get_logger().info('🚀 Starting automated pick-and-place sequence!')
        
        # Run automation in a thread to avoid blocking
        import threading
        thread = threading.Thread(target=self.run_pick_place_sequence)
        thread.daemon = True
        thread.start()

    def run_pick_place_sequence(self):
        """Main pick-place automation loop."""
        
        # Process each box
        for idx, box in enumerate(self.boxes):
            box_name = box["name"]
            box_color = box["color"]
            box_pos = np.array(box["pos"])
            box_size = box["size"]
            basket_pos = np.array(self.baskets[box_color])
            
            self.get_logger().info(f'\n{"="*60}')
            self.get_logger().info(f'[{idx+1}/{len(self.boxes)}] {box_name} ({box_color}) → {box_color} basket')
            self.get_logger().info(f'{"="*60}')
            
            # 1. Move above box
            self.get_logger().info(f'  Step 1: Moving above {box_name}...')
            above_pos = box_pos.copy()
            above_pos[2] += 0.15  # 15cm above box
            
            above_joints = self.solve_ik(above_pos)
            if above_joints is not None:
                self.move_to_joints(above_joints, duration=3.0)
                time.sleep(3.5)
            
            # 2. Lower to grasp
            self.get_logger().info(f'  Step 2: Lowering to grasp...')
            grasp_pos = box_pos.copy()
            grasp_pos[2] = box_pos[2] + box_size * 0.5 - 0.005
            
            grasp_joints = self.solve_ik(grasp_pos)
            if grasp_joints is not None:
                self.move_to_joints(grasp_joints, duration=2.0)
                time.sleep(2.5)
            
            # 3. Grasp box
            self.get_logger().info(f'  Step 3: Grasping {box_name}...')
            self.publish_gripper_event(f'grasp:{box_name}')
            time.sleep(0.5)
            
            # 4. Lift
            self.get_logger().info(f'  Step 4: Lifting...')
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.15
            
            lift_joints = self.solve_ik(lift_pos)
            if lift_joints is not None:
                self.move_to_joints(lift_joints, duration=2.0)
                time.sleep(2.5)
            
            # 5. Move to basket (via safe waypoint to avoid wall)
            self.get_logger().info(f'  Step 5: Moving to {box_color} basket...')
            
            # Waypoint above basket
            above_basket = basket_pos.copy()
            above_basket[2] += 0.20
            
            waypoint_joints = self.solve_ik(above_basket)
            if waypoint_joints is not None:
                self.move_to_joints(waypoint_joints, duration=4.0)
                time.sleep(4.5)
            
            # 6. Lower to place
            self.get_logger().info(f'  Step 6: Placing in basket...')
            place_pos = basket_pos.copy()
            place_pos[2] += 0.05  # Just above basket
            
            place_joints = self.solve_ik(place_pos)
            if place_joints is not None:
                self.move_to_joints(place_joints, duration=2.0)
                time.sleep(2.5)
            
            # 7. Release
            self.get_logger().info(f'  Step 7: Releasing...')
            self.publish_gripper_event(f'release:{box_name}:{box_color}_basket')
            time.sleep(0.5)
            
            # 8. Lift away
            self.get_logger().info(f'  Step 8: Retracting...')
            if waypoint_joints is not None:
                self.move_to_joints(waypoint_joints, duration=2.0)
                time.sleep(2.5)
            
            self.get_logger().info(f'✓ {box_name} placed in {box_color} basket!')
        
        # Return home
        self.get_logger().info('\n🏠 Returning home...')
        self.move_to_joints(self.home_joints, duration=4.0)
        time.sleep(4.5)
        
        self.get_logger().info('\n🎉 ALL BOXES SORTED! Demo complete.')

    def solve_ik(self, target_pos, target_quat=None):
        """Solve IK for target position."""
        if self.ik_solver is None:
            self.get_logger().error('IK solver not initialized!')
            return None
        
        try:
            # Use down-facing orientation if not specified
            if target_quat is None:
                target_quat = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing down
            
            # Update MuJoCo data with current joints
            self.mujoco_data.qpos[:6] = self.current_joints
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
            
            # Solve IK
            joints, error = self.ik_solver.solve(
                target_pos,
                target_quat=target_quat,
                max_iterations=500,
                tolerance=0.01
            )
            
            if error < 0.05:  # 5cm tolerance
                return joints
            else:
                self.get_logger().warn(f'IK error too large: {error:.4f}m')
                return joints  # Use anyway
                
        except Exception as e:
            self.get_logger().error(f'IK failed: {e}')
            return None

    def move_to_joints(self, target_joints, duration=3.0):
        """Send joint trajectory to UR driver."""
        if target_joints is None:
            return
        
        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Start point (current position)
        point1 = JointTrajectoryPoint()
        point1.positions = self.current_joints.tolist()
        point1.time_from_start = Duration(sec=0, nanosec=0)
        
        # End point (target)
        point2 = JointTrajectoryPoint()
        point2.positions = target_joints.tolist()
        point2.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        traj.points = [point1, point2]
        
        # Send via action
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj
        
        self._action_client.wait_for_server(timeout_sec=2.0)
        future = self._action_client.send_goal_async(goal_msg)
        
        self.get_logger().info(f'  → Trajectory sent (duration: {duration:.1f}s)')

    def publish_gripper_event(self, event):
        """Publish gripper event to scene manager."""
        msg = String()
        msg.data = event
        self.gripper_event_pub.publish(msg)
        self.get_logger().info(f'  → Gripper event: {event}')


def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
