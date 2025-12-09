#!/usr/bin/env python3
"""
MPC Controller Node

Provides real-time Model Predictive Control for the UR5e arm.
Subscribes to target poses and current state, publishes optimized joint trajectories.
"""

import sys
from pathlib import Path
import numpy as np
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Vector3
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header

# Add project root to path (Docker: /ros2_ws/src_project)
sys.path.insert(0, '/ros2_ws/src_project')
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from control.mpc_controller import MPCController


class MPCControllerNode(Node):
    """ROS 2 node for MPC-based trajectory optimization."""

    def __init__(self):
        super().__init__('mpc_controller')
        
        # Parameters
        self.declare_parameter('n_joints', 6)
        self.declare_parameter('horizon', 10)
        self.declare_parameter('dt', 0.05)
        self.declare_parameter('enable_fk', True)
        self.declare_parameter('publish_rate', 20.0)  # Hz
        
        n_joints = self.get_parameter('n_joints').value
        horizon = self.get_parameter('horizon').value
        dt = self.get_parameter('dt').value
        enable_fk = self.get_parameter('enable_fk').value
        
        # Initialize MPC controller
        self.mpc = MPCController(
            n_joints=n_joints,
            horizon=horizon,
            dt=dt,
            enable_fk=enable_fk
        )
        
        self.get_logger().info(f'MPC initialized: {n_joints} joints, horizon={horizon}, dt={dt}')
        
        # State
        self.current_state = None
        self.target_state = None
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            JointState,
            '/mpc/target_joint_state',
            self.target_callback,
            10
        )
        
        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/mpc/joint_trajectory',
            10
        )
        
        # Timer for periodic MPC solve
        publish_rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / publish_rate, self.mpc_loop)
        
        # Services
        self.add_obstacle_srv = self.create_service(
            Point,  # Will use custom srv in production
            '~/add_obstacle',
            self.add_obstacle_callback
        )
        
        self.get_logger().info('MPC Controller Node ready')

    def joint_state_callback(self, msg):
        """Receive current joint state."""
        if len(msg.position) >= self.mpc.n_joints and len(msg.velocity) >= self.mpc.n_joints:
            self.current_state = np.concatenate([
                np.array(msg.position[:self.mpc.n_joints]),
                np.array(msg.velocity[:self.mpc.n_joints])
            ])

    def target_callback(self, msg):
        """Receive target joint configuration."""
        if len(msg.position) >= self.mpc.n_joints:
            self.target_state = np.array(msg.position[:self.mpc.n_joints])

    def mpc_loop(self):
        """Main MPC control loop - runs at fixed rate."""
        if self.current_state is None or self.target_state is None:
            return
        
        try:
            # Solve MPC
            q_next, q_traj = self.mpc.compute_control(
                self.current_state,
                self.target_state
            )
            
            # Publish trajectory
            traj_msg = JointTrajectory()
            traj_msg.header = Header()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = [f'shoulder_pan_joint', 'shoulder_lift_joint', 
                                   'elbow_joint', 'wrist_1_joint', 
                                   'wrist_2_joint', 'wrist_3_joint']
            
            # Add trajectory points
            for k in range(q_traj.shape[0]):
                point = JointTrajectoryPoint()
                point.positions = q_traj[k].tolist()
                point.time_from_start.sec = 0
                point.time_from_start.nanosec = int(k * self.mpc.dt * 1e9)
                traj_msg.points.append(point)
            
            self.trajectory_pub.publish(traj_msg)
            
        except Exception as e:
            self.get_logger().error(f'MPC solve failed: {e}')

    def add_obstacle_callback(self, request, response):
        """Service to add obstacle to MPC."""
        # In production, use custom AddObstacle.srv
        # For now, assume request has position and size fields
        try:
            position = [request.x, request.y, request.z]
            size = [0.05, 0.05, 0.05]  # Default size
            self.mpc.add_obstacle(position, size)
            response.success = True
        except Exception as e:
            self.get_logger().error(f'Failed to add obstacle: {e}')
            response.success = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MPCControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
