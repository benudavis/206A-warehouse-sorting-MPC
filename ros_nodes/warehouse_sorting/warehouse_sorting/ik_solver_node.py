#!/usr/bin/env python3
"""
IK Planner Node - MoveIt-based IK and Planning

Provides IK solving and motion planning using MoveIt services.
Based on the working IKPlanner implementation.
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration


class IKPlannerNode(Node):
    """ROS 2 node providing MoveIt-based IK and planning."""

    def __init__(self):
        super().__init__('ik_planner')
        
        # MoveIt service clients
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        # Wait for services
        for srv, name in [(self.ik_client, 'compute_ik'),
                          (self.plan_client, 'plan_kinematic_path')]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for /{name} service...')
        
        # Subscriber for current joint state (for warm starting)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.current_joint_state = None
        
        self.get_logger().info('✓ IK Planner Node ready (using MoveIt services)')

    def joint_state_callback(self, msg):
        """Update current joint state."""
        if len(msg.position) >= 6:
            self.current_joint_state = msg

    def compute_ik(self, current_joint_state, x, y, z,
                   qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        """
        Compute an IK solution for the UR5e given a target pose.
        
        Args:
            current_joint_state: sensor_msgs/JointState – current robot state
            x, y, z: target position in base_link frame
            qx, qy, qz, qw: target orientation as quaternion
                            (default is 180° about y – end-effector flipped down)
        
        Returns:
            JointState with solution, or None on failure
        """
        # Build the target pose
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        # Build IK request
        ik_req = GetPositionIK.Request()
        ik_req.ik_request = PositionIKRequest()
        
        # MoveIt group name for UR5e arm
        ik_req.ik_request.group_name = 'ur_manipulator'
        
        # Name of the end-effector link
        ik_req.ik_request.ik_link_name = 'tool0'
        
        # Seed state for IK
        ik_req.ik_request.robot_state.joint_state = current_joint_state
        
        # Target pose
        ik_req.ik_request.pose_stamped = pose
        
        # Collision checking and timeout
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.timeout = Duration(sec=2)
        
        # Also set top-level robot_state
        ik_req.robot_state.joint_state = current_joint_state

        # Call service
        future = self.ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('IK service call failed.')
            return None

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, error code: {result.error_code.val}')
            return None

        self.get_logger().info('✓ IK solution found')
        return result.solution.joint_state

    def plan_to_joints(self, target_joint_state):
        """
        Plan motion given a desired joint configuration.
        
        Args:
            target_joint_state: JointState with target joint positions
        
        Returns:
            RobotTrajectory with planned path, or None on failure
        """
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.planner_id = "RRTConnectkConfigDefault"

        # Build goal constraints
        goal_constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0
                )
            )
        req.motion_plan_request.goal_constraints.append(goal_constraints)

        # Call planning service
        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error('Planning service call failed.')
            return None

        result = future.result()
        if result.motion_plan_response.error_code.val != 1:
            self.get_logger().error(f'Planning failed, error code: {result.motion_plan_response.error_code.val}')
            return None

        self.get_logger().info('✓ Motion plan computed successfully')
        return result.motion_plan_response.trajectory


def main(args=None):
    rclpy.init(args=args)
    node = IKPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
