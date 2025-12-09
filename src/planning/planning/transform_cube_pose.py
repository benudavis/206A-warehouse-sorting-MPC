import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PointStamped, Point
from custom_msgs.msg import CubeMsg, CubeArray
from visualization_msgs.msg import Marker, MarkerArray
import tf2_geometry_msgs

class TransformCubePose(Node):
    def __init__(self):
        super().__init__('transform_cube_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pose_sub = self.create_subscription(
            CubeArray,
            '/cube_poses',
            self.cube_pose_callback,
            10
        )

        self.cube_pose_pub = self.create_publisher(CubeArray, '/cubes_in_base_link', 3) # Please ensure this is filled

        # rclpy.spin_once(self, timeout_sec=2)
        self.cube_poses = None

        self.marker_pub = self.create_publisher(Marker, '/cube_markers', 3)

    def cube_pose_callback(self, msg: CubeArray):
        # while self.cube_poses is None:
        #     self.cube_poses = self.transform_cube_pose(msg)
        transformed_msg = self.transform_cube_pose(msg)
        if transformed_msg is not None:
            self.cube_pose_pub.publish(transformed_msg)
            point_marker = self.cubes_to_markers(transformed_msg)
            self.marker_pub.publish(point_marker)

    def transform_cube_pose(self, msg: CubeArray):
        """ 
        Transform point into base_link frame
        Args: 
            - msg: PointStamped - The message from /cube_pose, of the position of the cube in camera_depth_optical_frame
        Returns:
            Point: point in base_link_frame in form [x, y, z]
        """
        target_frame = "base_link"

        try:
            transform_lookup = self.tf_buffer.lookup_transform(
                target_frame, # target frame
                "camera_depth_optical_frame", # source frame
                rclpy.time.Time(),
                timeout = rclpy.duration.Duration(seconds = 0.1))
            
            for cube in msg.cubes:
                transformed_point_stamped = tf2_geometry_msgs.do_transform_point(cube.point, transform_lookup)
                cube.point.point.x = transformed_point_stamped.point.x
                cube.point.point.y = transformed_point_stamped.point.y
                cube.point.point.z = transformed_point_stamped.point.z
                cube.point.header = transformed_point_stamped.header
            return msg
        except Exception as e:
            rclpy.spin_once(self, timeout_sec=0.1)
            # self.get_logger().info(f"Transform failed: {e}")
            return None
        return

    # for visualization
    def cubes_to_markers(self, cube_array: CubeArray):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'cube_points'
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.id = 0

        marker.scale.x = 0.02
        marker.scale.y = 0.02

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for cube in cube_array.cubes:
            p = Point()
            p.x = cube.point.point.x
            p.y = cube.point.point.y
            p.z = cube.point.point.z
            marker.points.append(p)

        return marker

def main(args=None):
    rclpy.init(args=args)
    node = TransformCubePose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
