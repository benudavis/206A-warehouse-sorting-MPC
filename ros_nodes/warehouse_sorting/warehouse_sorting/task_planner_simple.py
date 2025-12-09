#!/usr/bin/env python3
"""
Simple Task Planner - Demonstrates Robot Motion & Pick-Place

Uses predefined joint configurations to demonstrate:
- Robot movement
- Box grasping (visual attachment)
- Box placement in baskets
"""

import numpy as np
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import String
from builtin_interfaces.msg import Duration


class SimpleTaskPlanner(Node):
    """Simple pick-place demonstrator."""

    def __init__(self):
        super().__init__('task_planner')
        
        # Publishers
        self.gripper_event_pub = self.create_publisher(
            String,
            '/gripper_events',
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
        
        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Predefined poses (from your working demo)
        self.poses = {
            'home': np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]),
            'above_red_1': np.array([0.5, -1.2, 1.3, -1.67, -1.57, 0.0]),
            'grasp_red_1': np.array([0.5, -1.0, 1.1, -1.67, -1.57, 0.0]),
            'lift': np.array([0.5, -1.4, 1.5, -1.67, -1.57, 0.0]),
            'above_red_basket': np.array([-0.5, -1.3, 1.4, -1.67, -1.57, 0.0]),
            'place_red': np.array([-0.5, -1.1, 1.2, -1.67, -1.57, 0.0]),
        }
        
        # Start demo after delay
        self.create_timer(5.0, self.start_demo)
        
        self.get_logger().info('✓ Simple Task Planner ready - demo starts in 5 seconds')

    def joint_callback(self, msg):
        """Update current joint positions."""
        if len(msg.position) >= 6:
            self.current_joints = np.array(msg.position[:6])

    def start_demo(self):
        """Start the demonstration."""
        self.get_logger().info('🚀 Starting pick-place demonstration!')
        
        # Run in thread
        thread = threading.Thread(target=self.run_demo)
        thread.daemon = True
        thread.start()

    def run_demo(self):
        """Demonstrate pick-place cycle."""
        time.sleep(2)  # Let system stabilize
        
        # Demo sequence for red_1 box
        self.get_logger().info('\n' + '='*60)
        self.get_logger().info('Demonstrating: red_1 → red basket')
        self.get_logger().info('='*60)
        
        # 1. Move above box
        self.get_logger().info('Step 1: Moving above red_1...')
        self.move_to_pose('above_red_1', duration=3.0)
        time.sleep(4.0)
        
        # 2. Lower to grasp
        self.get_logger().info('Step 2: Lowering to grasp...')
        self.move_to_pose('grasp_red_1', duration=2.0)
        time.sleep(2.5)
        
        # 3. Grasp (visual attachment)
        self.get_logger().info('Step 3: Grasping box...')
        self.publish_gripper_event('grasp:red_1')
        time.sleep(1.0)
        
        # 4. Lift
        self.get_logger().info('Step 4: Lifting...')
        self.move_to_pose('lift', duration=2.0)
        time.sleep(2.5)
        
        # 5. Move to basket
        self.get_logger().info('Step 5: Moving to red basket...')
        self.move_to_pose('above_red_basket', duration=4.0)
        time.sleep(4.5)
        
        # 6. Lower to place
        self.get_logger().info('Step 6: Placing in basket...')
        self.move_to_pose('place_red', duration=2.0)
        time.sleep(2.5)
        
        # 7. Release
        self.get_logger().info('Step 7: Releasing...')
        self.publish_gripper_event('release:red_1:red_basket')
        time.sleep(1.0)
        
        # 8. Return home
        self.get_logger().info('Step 8: Returning home...')
        self.move_to_pose('home', duration=4.0)
        time.sleep(4.5)
        
        self.get_logger().info('\n🎉 Demonstration complete!')
        self.get_logger().info('Red box should now be in the red basket in Foxglove!')

    def move_to_pose(self, pose_name, duration=3.0):
        """Move robot to predefined pose."""
        if pose_name not in self.poses:
            self.get_logger().error(f'Unknown pose: {pose_name}')
            return
        
        target_joints = self.poses[pose_name]
        
        # Create trajectory
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Current position
        point1 = JointTrajectoryPoint()
        point1.positions = self.current_joints.tolist()
        point1.time_from_start = Duration(sec=0, nanosec=0)
        
        # Target position
        point2 = JointTrajectoryPoint()
        point2.positions = target_joints.tolist()
        point2.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        traj.points = [point1, point2]
        
        # Send action
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj
        
        self._action_client.wait_for_server(timeout_sec=2.0)
        self._action_client.send_goal_async(goal_msg)
        
        self.get_logger().info(f'  → Moving to {pose_name} ({duration:.1f}s)')

    def publish_gripper_event(self, event):
        """Publish gripper event."""
        msg = String()
        msg.data = event
        self.gripper_event_pub.publish(msg)
        self.get_logger().info(f'  → Gripper: {event}')


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTaskPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

