#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from planning.forward_kinematics import ur7e_forward_kinematics_from_joint_state
import numpy as np


class ForwardKinematicsNode(Node):
    """
    Subscriber node that receives /joint_states messages, computes forward
    kinematics, and displays the resulting transformation matrix.
    """

    def __init__(self):
        super().__init__('forward_kinematics_node')

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info("Forward kinematics node started. Listening to /joint_states...")
        self.get_logger().info("Transformation matrix will be displayed below:")

    def joint_state_callback(self, msg: JointState):
        """
        Callback function that processes joint state messages and computes FK.
        
        Args:
            msg: JointState message from /joint_states topic
        """
        try:
            # Compute forward kinematics transformation matrix
            gst = ur7e_forward_kinematics_from_joint_state(msg)

            # Display the transformation matrix
            self.display_transformation_matrix(gst)

        except Exception as e:
            self.get_logger().error(f"Error computing forward kinematics: {e}")

    def display_transformation_matrix(self, gst: np.ndarray):
        """
        Displays the 4x4 homogeneous transformation matrix.
        
        Args:
            gst: 4x4 numpy array representing the transformation matrix
        """
        # Extract position (translation)
        position = gst[0:3, 3]
        
        # Extract rotation matrix
        rotation = gst[0:3, 0:3]
        
        # Print formatted output
        print("\n" + "="*70)
        print("Forward Kinematics Transformation Matrix (base_link -> wrist_3_link)")
        print("="*70)
        
        # Print the full 4x4 matrix
        print("\nHomogeneous Transformation Matrix:")
        print(gst)
        
        # Print position
        print(f"\nPosition (x, y, z):")
        print(f"  x: {position[0]:.6f}")
        print(f"  y: {position[1]:.6f}")
        print(f"  z: {position[2]:.6f}")
        
        print("="*70 + "\n")

def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematicsNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
