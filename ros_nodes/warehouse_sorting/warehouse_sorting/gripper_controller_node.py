#!/usr/bin/env python3
"""
Gripper Controller Node

Controls the Robotiq 2F-85 gripper.
Provides action server for gripper commands.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand


class GripperControllerNode(Node):
    """ROS 2 node for Robotiq 2F-85 gripper control."""

    def __init__(self):
        super().__init__('gripper_controller')
        
        # Parameters
        self.declare_parameter('gripper_joint_name', 'gripper_finger1_joint')
        self.declare_parameter('max_position', 0.8)  # radians (fully open)
        self.declare_parameter('min_position', 0.0)  # radians (fully closed)
        
        self.gripper_joint = self.get_parameter('gripper_joint_name').value
        self.max_pos = self.get_parameter('max_position').value
        self.min_pos = self.get_parameter('min_position').value
        
        # State
        self.current_position = self.max_pos  # Start open
        self.current_effort = 0.0
        
        # Publishers
        self.gripper_cmd_pub = self.create_publisher(
            Float32,
            '/gripper/command',
            10
        )
        
        self.gripper_state_pub = self.create_publisher(
            JointState,
            '/gripper/state',
            10
        )
        
        # Simple command subscriber (for basic open/close)
        self.simple_cmd_sub = self.create_subscription(
            Float32,
            '/gripper/simple_command',
            self.simple_command_callback,
            10
        )
        
        # Action server for gripper commands
        self._action_server = ActionServer(
            self,
            GripperCommand,
            '~/gripper_command',
            self.execute_gripper_command
        )
        
        # State publisher timer
        self.state_timer = self.create_timer(0.1, self.publish_state)
        
        self.get_logger().info('Gripper Controller Node ready')

    def simple_command_callback(self, msg):
        """Simple gripper command: 0 = open, 255 = closed."""
        position = self.max_pos if msg.data < 128 else self.min_pos
        self.send_gripper_command(position)

    def send_gripper_command(self, position):
        """Send gripper position command."""
        position = np.clip(position, self.min_pos, self.max_pos)
        self.current_position = position
        
        cmd_msg = Float32()
        cmd_msg.data = float(position)
        self.gripper_cmd_pub.publish(cmd_msg)

    def execute_gripper_command(self, goal_handle):
        """Execute gripper action (for full control with force)."""
        self.get_logger().info('Executing gripper command action')
        
        request = goal_handle.request
        target_position = request.command.position
        max_effort = request.command.max_effort
        
        # Send command
        self.send_gripper_command(target_position)
        
        # Simple execution - in production, monitor force and position feedback
        import time
        time.sleep(1.0)  # Wait for gripper to move
        
        # Success feedback
        goal_handle.succeed()
        
        result = GripperCommand.Result()
        result.position = self.current_position
        result.effort = self.current_effort
        result.stalled = False
        result.reached_goal = True
        
        return result

    def publish_state(self):
        """Publish current gripper state."""
        state_msg = JointState()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.name = [self.gripper_joint]
        state_msg.position = [self.current_position]
        state_msg.effort = [self.current_effort]
        
        self.gripper_state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GripperControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
