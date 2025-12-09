#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import PointStamped

from tf2_geometry_msgs import do_transform_point

class TransformCubePose(Node):

    def __init__(self):

        super().__init__('transform_cube_pose')

        # TF buffer & listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to cube pose in camera frame
        self.cube_pose_sub = self.create_subscription(
            PointStamped,
            '/cube_pose',
            self.cube_pose_callback,
            10
        )

        # Publish cube pose in base_link frame
        # (you can rename topic if your handout specifies a different one)
        self.cube_pose_pub = self.create_publisher(
            PointStamped,
            '/cube_pose_in_base',
            10
        )

        self.cube_pose = None

        # Give TF a moment to populate (optional but helpful)
        rclpy.spin_once(self, timeout_sec=2.0)
        self.get_logger().info("transform_cube_pose node started")

    def cube_pose_callback(self, msg: PointStamped):

        transformed_point = self.transform_cube_pose(msg)

        if transformed_point is not None:
            self.cube_pose = transformed_point
            self.cube_pose_pub.publish(transformed_point)

    def transform_cube_pose(self, msg: PointStamped):

        """
        Transform point into base_link frame.
        Args:
            msg: PointStamped - The message from /cube_pose, of the position of
                 the cube in camera_depth_optical_frame.
        Returns:
            PointStamped in base_link frame, or None on failure.
        """

        target_frame = 'base_link'
        source_frame = msg.header.frame_id

        try:
            # Look up transform from source_frame -> target_frame
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),   # latest available transform
                timeout=Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(
                f"Failed to lookup transform {source_frame} -> {target_frame}: {e}"
            )
            return None

        # Apply transform to the point
        transformed = do_transform_point(msg, transform)

        # Update header to reflect new frame and time
        transformed.header.stamp = self.get_clock().now().to_msg()
        transformed.header.frame_id = target_frame

        self.get_logger().info(
            f"Cube in {target_frame}: "
            f"({transformed.point.x:.3f}, "
            f"{transformed.point.y:.3f}, "
            f"{transformed.point.z:.3f})"
        )

        return transformed

def main(args=None):

    rclpy.init(args=args)
    node = TransformCubePose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':

    main()