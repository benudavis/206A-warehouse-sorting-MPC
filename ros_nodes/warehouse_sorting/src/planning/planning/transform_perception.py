#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from custom_msgs.msg import LabeledCubeArray, LabeledCube, BoxBounds
import numpy as np


class TransformPerception(Node):
    """
    Transform node for perception messages from camera frame to base_link frame.
    
    Handles:
    - LabeledCubeArray: transforms all cubes from camera frame to base_link
    - BoxBounds: transforms obstacle bounding boxes from camera frame to base_link
      (with z-extrapolation to make obstacles larger)
    """

    def __init__(self):
        super().__init__('transform_perception')

        # TF buffer & listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Track camera frame_id from labeled cubes
        self.camera_frame_id = None

        # Subscribe to perception topics in camera frame
        self.labeled_cubes_sub = self.create_subscription(
            LabeledCubeArray,
            '/labeled_cubes',
            self.labeled_cubes_callback,
            10
        )

        self.obstacles_sub = self.create_subscription(
            BoxBounds,
            '/obstacles',
            self.obstacles_callback,
            10
        )

        # Publish transformed topics in base_link frame
        self.labeled_cubes_base_pub = self.create_publisher(
            LabeledCubeArray,
            '/labeled_cubes_base',
            10
        )

        self.obstacles_base_pub = self.create_publisher(
            BoxBounds,
            '/obstacles_base',
            10
        )

        # Give TF a moment to populate
        rclpy.spin_once(self, timeout_sec=2.0)
        self.get_logger().info("transform_perception node started")

    def labeled_cubes_callback(self, msg: LabeledCubeArray):
        """
        Transform LabeledCubeArray from camera frame to base_link frame.
        
        Args:
            msg: LabeledCubeArray in camera frame
        """
        # Store camera frame_id for obstacle transformation
        if msg.cubes:
            self.camera_frame_id = msg.cubes[0].point.header.frame_id

        # Transform each cube in the array
        transformed_cubes = []
        target_frame = 'base_link'

        for i, labeled_cube in enumerate(msg.cubes):
            # Transform the cube's point from camera frame to base_link
            transformed_point = self._transform_point_to_base(labeled_cube.point, target_frame)
            
            if transformed_point is None:
                self.get_logger().warn(f"Failed to transform cube {i} to base_link")
                continue

            # Create transformed labeled cube
            transformed_cube = LabeledCube()
            transformed_cube.point = transformed_point
            transformed_cube.color_label = labeled_cube.color_label

            transformed_cubes.append(transformed_cube)

        # Publish transformed array
        if transformed_cubes:
            transformed_array = LabeledCubeArray()
            transformed_array.header = msg.header
            transformed_array.header.frame_id = target_frame
            transformed_array.header.stamp = self.get_clock().now().to_msg()
            transformed_array.cubes = transformed_cubes

            self.labeled_cubes_base_pub.publish(transformed_array)
            self.get_logger().debug(
                f"Transformed and published {len(transformed_cubes)} cubes to base_link"
            )

    def obstacles_callback(self, msg: BoxBounds):
        """
        Transform obstacle bounds from camera frame to base_link frame.
        Extrapolates z values in camera frame before transformation.
        
        Args:
            msg: BoxBounds message with x_min, x_max, y_min, y_max, z_min, z_max in camera frame
        """
        # Determine source frame
        if self.camera_frame_id is None:
            # Try common camera frame names
            source_frames = [
                'camera_depth_optical_frame',
                'camera_color_optical_frame',
                'camera_link',
            ]
            source_frame = None
            for frame in source_frames:
                try:
                    self.tf_buffer.lookup_transform(
                        'base_link',
                        frame,
                        Time(),
                        timeout=Duration(seconds=0.1)
                    )
                    source_frame = frame
                    break
                except:
                    continue
            if source_frame is None:
                self.get_logger().warn("Could not determine camera frame for obstacle transformation")
                return
        else:
            source_frame = self.camera_frame_id

        # Extrapolate z_min and z_max in camera frame before transformation
        # Extend the z range to make the obstacle larger in camera's z direction
        z_extrapolation = 0.8  # meters - extend obstacle depth/height in camera z direction
        z_min_extended = msg.z_min - z_extrapolation  # Extend backward in camera z
        z_max_extended = msg.z_max + z_extrapolation  # Extend forward in camera z

        # Transform the 8 corners of the bounding box from camera frame to base_link
        # This is the most accurate way to handle rotations
        corners_camera = np.array([
            [msg.x_min, msg.y_min, z_min_extended],
            [msg.x_max, msg.y_min, z_min_extended],
            [msg.x_min, msg.y_max, z_min_extended],
            [msg.x_max, msg.y_max, z_min_extended],
            [msg.x_min, msg.y_min, z_max_extended],
            [msg.x_max, msg.y_min, z_max_extended],
            [msg.x_min, msg.y_max, z_max_extended],
            [msg.x_max, msg.y_max, z_max_extended],
        ])

        # Get transform from camera frame to base_link
        target_frame = 'base_link'
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(
                f"Could not find transform {source_frame} -> {target_frame} for obstacles: {e}"
            )
            return

        # Transform each corner to base_link
        corners_base = []
        for corner in corners_camera:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(corner[0])
            point_stamped.point.y = float(corner[1])
            point_stamped.point.z = float(corner[2])

            transformed = do_transform_point(point_stamped, transform)
            corners_base.append([
                transformed.point.x,
                transformed.point.y,
                transformed.point.z
            ])

        corners_base = np.array(corners_base)

        # Compute new bounding box in base_link frame
        x_min_base = float(np.min(corners_base[:, 0]))
        x_max_base = float(np.max(corners_base[:, 0]))
        y_min_base = float(np.min(corners_base[:, 1]))
        y_max_base = float(np.max(corners_base[:, 1]))
        z_min_base = float(np.min(corners_base[:, 2]))
        z_max_base = float(np.max(corners_base[:, 2]))

        # Create transformed BoxBounds message
        transformed_obstacle = BoxBounds()
        transformed_obstacle.x_min = x_min_base
        transformed_obstacle.x_max = x_max_base
        transformed_obstacle.y_min = y_min_base
        transformed_obstacle.y_max = y_max_base
        transformed_obstacle.z_min = z_min_base
        transformed_obstacle.z_max = z_max_base

        # Publish transformed obstacle
        self.obstacles_base_pub.publish(transformed_obstacle)

        # Calculate dimensions for logging
        dim_x = (x_max_base - x_min_base)
        dim_y = (y_max_base - y_min_base)
        dim_z = (z_max_base - z_min_base)
        center_x = (x_min_base + x_max_base) / 2.0
        center_y = (y_min_base + y_max_base) / 2.0
        center_z = (z_min_base + z_max_base) / 2.0

        self.get_logger().info(
            f"Obstacle transformed to base_link: "
            f"center=({center_x:.3f}, {center_y:.3f}, {center_z:.3f}) m, "
            f"dimensions={dim_x:.3f} x {dim_y:.3f} x {dim_z:.3f} m"
        )

    def _transform_point_to_base(self, point_stamped: PointStamped, target_frame: str = 'base_link') -> PointStamped:
        """
        Transform a point from camera frame to base_link frame.
        
        Args:
            point_stamped: PointStamped in camera frame
            target_frame: Target frame (default: 'base_link')
            
        Returns:
            PointStamped in base_link frame, or None on failure
        """
        source_frame = point_stamped.header.frame_id

        try:
            # Look up transform from source_frame -> target_frame
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),  # latest available transform
                timeout=Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(
                f"Failed to lookup transform {source_frame} -> {target_frame}: {e}"
            )
            return None

        # Apply transform to the point
        transformed = do_transform_point(point_stamped, transform)

        # Update header to reflect new frame and time
        transformed.header.stamp = self.get_clock().now().to_msg()
        transformed.header.frame_id = target_frame

        return transformed


def main(args=None):
    rclpy.init(args=args)
    node = TransformPerception()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
