import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

class RealSensePCSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')

        # Plane coefficients and max distance (meters)
        self.declare_parameter('plane.a', 0.0)
        self.declare_parameter('plane.b', 0.0)
        self.declare_parameter('plane.c', 0.0)
        self.declare_parameter('plane.d', 0.0)
        self.declare_parameter('max_distance', 0.6)

        self.a = self.get_parameter('plane.a').value
        self.b = self.get_parameter('plane.b').value
        self.c = self.get_parameter('plane.c').value
        self.d = self.get_parameter('plane.d').value
        self.max_distance = self.get_parameter('max_distance').value

        # Subscribers
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # Publishers
        self.cube_pose_pub = self.create_publisher(PointStamped, '/cube_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info("Subscribed to PointCloud2 topic and marker publisher ready")

    def pointcloud_callback(self, msg: PointCloud2):
        # Convert PointCloud2 to Nx3 array
        points = []
        for p in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
            points.append([p[0], p[1], p[2]])

        points = np.array(points)
        # ------------------------
        #TODO: Add your code here!
        # ------------------------
        if points.size == 0:
            self.logger().warn("PointCloud had no valid (non-Nan) points")
            return
        
        pts = points.astype(np.float32, copy=False)
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.size == 0:
            self.get_logger().warn("All points were non-finite after conversion.")
            return 

        dists = np.linalg.norm(pts, axis=1)
        dist_mask = dists <= float(self.max_distance)

        a, b, c, d = float(self.a), float(self.b), float(self.c), float(self.d)
        signed = a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d

        plane_mask = signed < 0.0

        height_mask = (0.15 <= pts[:, 1]) & (pts[:, 1] <= 0.2)   
        keep_mask = np.logical_and(dist_mask & plane_mask, ~height_mask)        
        filtered_points = pts[keep_mask]

        print(min(filtered_points[:, 0]), min(filtered_points[:, 1]), min(filtered_points[:, 2]))
        print(max(filtered_points[:, 0]), max(filtered_points[:, 1]), max(filtered_points[:, 2]))


        if filtered_points.size == 0:
            self.get_logger().warn(
                "No points survived filering. Adjust max_distance or plane coefficients"
            )
            return
        
        centriod = filtered_points.mean(axis=0)

        cube_x, cube_y, cube_z = map(float, centriod)

        self.get_logger().info(f"Filtered points: {filtered_points.shape[0]}")

        cube_pose = PointStamped()
        cube_pose.header = Header()
        cube_pose.header.stamp = msg.header.stamp
        cube_pose.header.frame_id = msg.header.frame_id
        cube_pose.point.x = cube_x
        cube_pose.point.y = cube_y
        cube_pose.point.z = cube_z

        self.cube_pose_pub.publish(cube_pose)

        self.publish_filtered_points(filtered_points, msg.header)

    def publish_filtered_points(self, filtered_points: np.ndarray, header: Header):
        # Create PointCloud2 message from filtered Nx3 array
        filtered_msg = pc2.create_cloud_xyz32(header, filtered_points.tolist())
        self.filtered_points_pub.publish(filtered_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePCSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()