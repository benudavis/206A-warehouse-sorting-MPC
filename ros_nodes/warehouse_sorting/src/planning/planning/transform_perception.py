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
    Transforms perception outputs from a camera frame to base_link.

    Subscribes:
      - /labeled_cubes   (LabeledCubeArray)  [points in camera frame]
      - /obstacles       (BoxBounds)         [bounds in camera frame; no header]

    Publishes:
      - /labeled_cubes_base (LabeledCubeArray) [points in base_link]
      - /obstacles_base     (BoxBounds)        [AABB in base_link]
    """

    def __init__(self):
        super().__init__('transform_perception')

        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('default_camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('tf_timeout_sec', 0.25)

        # symmetric inflation in BASE frame
        self.declare_parameter('obstacle_inflate_xy', 0.00)
        self.declare_parameter('obstacle_inflate_z', 0.00)

        self.target_frame = self.get_parameter('target_frame').value
        self.default_camera_frame = self.get_parameter('default_camera_frame').value
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.obstacle_inflate_xy = float(self.get_parameter('obstacle_inflate_xy').value)
        self.obstacle_inflate_z = float(self.get_parameter('obstacle_inflate_z').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_cube_source_frame = None

        self.labeled_cubes_sub = self.create_subscription(
            LabeledCubeArray, '/labeled_cubes', self.labeled_cubes_callback, 10
        )
        self.obstacles_sub = self.create_subscription(
            BoxBounds, '/obstacles', self.obstacles_callback, 10
        )

        self.labeled_cubes_base_pub = self.create_publisher(
            LabeledCubeArray, '/labeled_cubes_base', 10
        )
        self.obstacles_base_pub = self.create_publisher(
            BoxBounds, '/obstacles_base', 10
        )

        self.get_logger().info(
            f"transform_perception started. target_frame='{self.target_frame}', default_camera_frame='{self.default_camera_frame}'"
        )

    def _lookup_transform_latest(self, target_frame: str, source_frame: str):
        timeout = Duration(seconds=self.tf_timeout_sec)
        return self.tf_buffer.lookup_transform(target_frame, source_frame, Time(), timeout=timeout)

    def _transform_point_latest(self, pt: PointStamped, target_frame: str):
        if not pt.header.frame_id:
            return None

        pt2 = PointStamped()
        pt2.header.frame_id = pt.header.frame_id
        pt2.header.stamp = self.get_clock().now().to_msg()
        pt2.point = pt.point

        try:
            tf = self._lookup_transform_latest(target_frame, pt2.header.frame_id)
            out = do_transform_point(pt2, tf)
            out.header.frame_id = target_frame
            out.header.stamp = self.get_clock().now().to_msg()
            return out
        except Exception as e:
            self.get_logger().warn(f"TF failed for point {pt2.header.frame_id} -> {target_frame}: {e}")
            return None

    def _camera_frame_for_obstacles(self) -> str:
        return self.last_cube_source_frame or self.default_camera_frame

    def labeled_cubes_callback(self, msg: LabeledCubeArray):
        if not msg.cubes:
            return

        target = self.target_frame
        out_arr = LabeledCubeArray()
        out_arr.header.frame_id = target
        out_arr.header.stamp = self.get_clock().now().to_msg()
        out_arr.cubes = []

        for c in msg.cubes:
            src_frame = c.point.header.frame_id or self.default_camera_frame
            self.last_cube_source_frame = src_frame

            in_pt = c.point
            if not in_pt.header.frame_id:
                in_pt.header.frame_id = src_frame

            if in_pt.header.frame_id == target:
                out_pt = PointStamped()
                out_pt.header.frame_id = target
                out_pt.header.stamp = self.get_clock().now().to_msg()
                out_pt.point = in_pt.point
            else:
                out_pt = self._transform_point_latest(in_pt, target)
                if out_pt is None:
                    continue

            out_cube = LabeledCube()
            out_cube.color_label = c.color_label
            out_cube.point = out_pt
            out_arr.cubes.append(out_cube)

        if out_arr.cubes:
            self.labeled_cubes_base_pub.publish(out_arr)

    def obstacles_callback(self, msg: BoxBounds):
        target = self.target_frame
        source = self._camera_frame_for_obstacles()

        corners_cam = np.array([
            [msg.x_min, msg.y_min, msg.z_min],
            [msg.x_max, msg.y_min, msg.z_min],
            [msg.x_min, msg.y_max, msg.z_min],
            [msg.x_max, msg.y_max, msg.z_min],
            [msg.x_min, msg.y_min, msg.z_max],
            [msg.x_max, msg.y_min, msg.z_max],
            [msg.x_min, msg.y_max, msg.z_max],
            [msg.x_max, msg.y_max, msg.z_max],
        ], dtype=float)

        try:
            tf = self._lookup_transform_latest(target, source)
        except Exception as e:
            self.get_logger().warn(f"TF failed for obstacle {source} -> {target}: {e}")
            return

        now_msg = self.get_clock().now().to_msg()
        corners_base = []
        for (x, y, z) in corners_cam:
            p = PointStamped()
            p.header.frame_id = source
            p.header.stamp = now_msg
            p.point.x = float(x)
            p.point.y = float(y)
            p.point.z = float(z)

            try:
                q = do_transform_point(p, tf)
                corners_base.append([q.point.x, q.point.y, q.point.z])
            except Exception as e:
                self.get_logger().warn(f"Corner transform failed: {e}")
                return

        corners_base = np.array(corners_base, dtype=float)

        x_min_b = float(np.min(corners_base[:, 0])) - self.obstacle_inflate_xy
        x_max_b = float(np.max(corners_base[:, 0])) + self.obstacle_inflate_xy
        y_min_b = float(np.min(corners_base[:, 1])) - self.obstacle_inflate_xy
        y_max_b = float(np.max(corners_base[:, 1])) + self.obstacle_inflate_xy
        z_min_b = float(np.min(corners_base[:, 2])) - self.obstacle_inflate_z
        z_max_b = float(np.max(corners_base[:, 2])) + self.obstacle_inflate_z

        out = BoxBounds()
        out.x_min = x_min_b
        out.x_max = x_max_b
        out.y_min = y_min_b
        out.y_max = y_max_b
        out.z_min = z_min_b
        out.z_max = z_max_b

        self.obstacles_base_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TransformPerception()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
