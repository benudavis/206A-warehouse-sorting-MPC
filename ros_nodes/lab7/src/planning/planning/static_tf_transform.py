#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class StaticTFNode(Node):
    """
    Publishes a fixed transform from an AR marker frame to base_link, without SciPy.

    Parameters:
        ar_marker (string): name of the marker frame, e.g. "ar_marker_6" or "ar_marker_8".
    """

    def __init__(self):
        super().__init__('static_tf_node')

        # Parameter: which AR marker to use as parent frame
        self.declare_parameter('ar_marker', 'ar_marker_8')
        marker_frame = self.get_parameter('ar_marker') \
                             .get_parameter_value() \
                             .string_value

        self.br = StaticTransformBroadcaster(self)

        # Precomputed transform: ar_marker -> base_link
        # Translation [m]
        tx = 0.0
        ty = 0.16
        tz = -0.13

        # Quaternion (x, y, z, w) corresponding to:
        # R = [[-1, 0, 0],
        #      [ 0, 0, 1],
        #      [ 0, 1, 0]]
        qx = 0.0
        qy = 0.70710678
        qz = 0.70710678
        qw = 0.0

        # Fill TransformStamped
        self.transform = TransformStamped()
        self.transform.header.frame_id = marker_frame     # parent
        self.transform.child_frame_id = 'base_link'       # child

        self.transform.transform.translation.x = tx
        self.transform.transform.translation.y = ty
        self.transform.transform.translation.z = tz

        self.transform.transform.rotation.x = qx
        self.transform.transform.rotation.y = qy
        self.transform.transform.rotation.z = qz
        self.transform.transform.rotation.w = qw

        self.get_logger().info(
            f"Static TF configured: parent={self.transform.header.frame_id} "
            f"-> child={self.transform.child_frame_id} | "
            f"translation=({tx:.3f}, {ty:.3f}, {tz:.3f}) m, "
            f"quaternion=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})."
        )

        # Broadcast periodically (20 Hz)
        self.timer = self.create_timer(0.05, self.broadcast_tf)

        self.get_logger().info(
            f"StaticTFNode publishing transform {marker_frame} -> base_link "
            f"at 20 Hz."
        )

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)


def main(args=None):
    rclpy.init(args=args)
    node = StaticTFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
