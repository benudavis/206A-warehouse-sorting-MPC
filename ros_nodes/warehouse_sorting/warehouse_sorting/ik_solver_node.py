#!/usr/bin/env python3
"""
IK Solver Service Node

Provides inverse kinematics solving as a ROS service.
Uses MuJoCo-based Jacobian IK solver.
"""

import sys
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
import mujoco

# Add project root to path (Docker: /ros2_ws/src_project)
sys.path.insert(0, '/ros2_ws/src_project')
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from control.inverse_kinematics import IKSolver


class IKSolverNode(Node):
    """ROS 2 node providing IK solving service."""

    def __init__(self):
        super().__init__('ik_solver')
        
        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('site_name', 'arm_hand_pinch')
        
        model_path = self.get_parameter('model_path').value
        site_name = self.get_parameter('site_name').value
        
        # Load MuJoCo model
        if not model_path:
            # Build default model (same as demos)
            # In Docker: /ros2_ws/sim/models
            models_dir = Path("/ros2_ws/sim/models")
            scene = mujoco.MjSpec.from_file(str(models_dir / "scene.xml"))
            arm_spec = mujoco.MjSpec.from_file(str(models_dir / "universal_robots_ur5e" / "ur5e.xml"))
            hand_spec = mujoco.MjSpec.from_file(str(models_dir / "robotiq_2f85" / "2f85.xml"))
            arm_spec.site("attachment_site").attach_body(hand_spec.worldbody, "hand_", "")
            scene.site("robot_site").attach_body(arm_spec.worldbody, "arm_", "")
            self.model = scene.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        
        self.data = mujoco.MjData(self.model)
        
        # Initialize IK solver
        self.ik_solver = IKSolver(self.model, self.data, site_name=site_name)
        
        self.get_logger().info(f'IK solver initialized with site: {site_name}')
        
        # Subscriber for current joint state
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Service
        # Note: Using basic types here; in production use custom SolveIK.srv
        self.ik_service = self.create_service(
            JointState,  # Placeholder - should use custom SolveIK srv
            '~/solve_ik',
            self.solve_ik_callback
        )
        
        self.get_logger().info('IK Solver Service ready at ~/solve_ik')

    def joint_state_callback(self, msg):
        """Update internal state from joint_states topic."""
        if len(msg.position) >= 6:
            self.data.qpos[:6] = np.array(msg.position[:6])
            mujoco.mj_forward(self.model, self.data)

    def solve_ik_callback(self, request, response):
        """
        Solve IK service callback.
        
        In production, use custom SolveIK.srv with:
          - geometry_msgs/Pose target
          - bool use_orientation
          - float64 tolerance
          - int32 max_iterations
        """
        try:
            # Extract target from request (simplified)
            # In production: target_pos = [request.target.position.x, ...]
            target_pos = np.array([0.4, 0.0, 0.5])  # Placeholder
            
            # Solve IK
            q_solution, success = self.ik_solver.solve(
                target_pos,
                target_quat=None,
                max_iterations=100,
                tolerance=0.01
            )
            
            # Build response
            response.position = q_solution.tolist()
            response.name = [f'joint_{i}' for i in range(len(q_solution))]
            
            if success:
                self.get_logger().info(f'IK solved successfully')
            else:
                self.get_logger().warn(f'IK did not fully converge')
            
        except Exception as e:
            self.get_logger().error(f'IK solve failed: {e}')
            response.position = [0.0] * 6
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = IKSolverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
