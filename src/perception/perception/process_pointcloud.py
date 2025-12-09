import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, Point
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header, ColorRGBA
from custom_msgs.msg import CubeMsg, CubeArray, BoxBounds, LabeledCube, LabeledCubeArray
import open3d as o3d
import sys
from visualization_msgs.msg import Marker

print(sys.executable)

class RealSensePCSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_pc_subscriber')

        # Plane coefficients and max distance (meters)
        self.declare_parameter('plane.a', -0.1)
        self.declare_parameter('plane.b', 30.0)
        self.declare_parameter('plane.c', 0.2)
        self.declare_parameter('plane.d', -1.0)
        self.declare_parameter('max_distance', 0.4)

        self.a = self.get_parameter('plane.a').value
        self.b = self.get_parameter('plane.b').value
        self.c = self.get_parameter('plane.c').value
        self.d = self.get_parameter('plane.d').value
        self.max_distance = self.get_parameter('max_distance').value
        # A: 0.04285410737539653, B: 0.9787593236334413, C: 0.20048369480251182, D: -0.11077546885306976
        # A: 0.05605865125563571, B: 0.9779047915923962, C: 0.20139425562818586, D: -0.10542028878128488
        #A: -0.05246574985236953, B: -0.9778029494580152, C: -0.20285151496509746, D: 0.09309383035589032
        # A: -0.04108913662654519, B: 0.83685830225683, C: 0.5458753198259663, D: -0.5252099623498152
        # A: 0.058497469242714185, B: 0.9919099390134098, C: 0.11266196775580856, D: -0.1072380530762859
        # A: 0.07057627911463674, B: 0.9918095618417286, C: 0.10645553938452945, D: -0.0902546378511314



        # Subscribers
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )

        # Publishers
        self.cube_pose_pub = self.create_publisher(CubeArray, '/cube_poses', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)
        self.obstacles_pub = self.create_publisher(BoxBounds, '/obstacles', 1)
        self.obstacle_marker_pub = self.create_publisher(Marker, '/obstacle_marker',1)
        self.labeled_cube_pub = self.create_publisher(LabeledCubeArray, '/labeled_cubes', 1)

        self.get_logger().info("Subscribed to PointCloud2 topic and marker publisher ready")

    def classify_color(self, rgb_mean: np.ndarray) -> str:
        r, g, b = rgb_mean[0], rgb_mean[1], rgb_mean[2]
        
        if r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        else:
            return "unknown"


    def pointcloud_callback(self, msg: PointCloud2):
        # Convert PointCloud2 to Nx3 array
        points = []
        colors = []
        for p in pc2.read_points(msg, field_names=('x','y','z','rgb'), skip_nans=True):
            points.append([p[0], p[1], p[2]])
            colors.append(p[3])

        points = np.array(points)
        rgb_uint32 = np.frombuffer(np.array(colors, dtype=np.float32).tobytes(), dtype=np.uint32)
        r = ((rgb_uint32 >> 16) & 0xFF)/255
        g = ((rgb_uint32 >> 8) & 0xFF)/255
        b = (rgb_uint32 & 0xFF)/255
        colors = np.stack([r, g, b], axis=1).astype(np.float64) # Nx3 flaot array from 0.0 to 1.0
        # print([r,g,b])


        # Apply max distance filter 
        distances = np.linalg.norm(points, axis=1)
        max_dist_mask = distances <= self.max_distance
        
        # Filter points below the plane
        plane = self.a*points[:,0] + self.b*points[:,1]+self.c*points[:,2] + self.d
        plane_mask = plane <= 0.0
        combined_mask = np.logical_and(max_dist_mask, plane_mask)
       
        # Apply other filtering relative to plane
        filtered_points = points[combined_mask]
        filtered_colors = colors[combined_mask]
        filtered_colors = filtered_colors.astype(np.float64)
        self.get_logger().info(f"Filtered points: {filtered_points.shape}")
        self.get_logger().info(f"Filtered colors: {filtered_colors.shape}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        labels = np.array(
            pcd.cluster_dbscan(eps=0.015,  # distance threshold
                            min_points=40,   # minimum cluster size
                            print_progress=True)
        )


        max_label = labels.max()
        # print(f"Found {max_label + 1} potential objects")

        cube_list = []
        labeled_cube_list = []
        cluster_sizes = []
        MIN_CLUSTER_POINTS = 100
        for label in range(max_label + 1):
            indices = np.where(labels == label)[0]
            # self.get_logger().info(f"size cluster: {len(indices)}")
            if len(indices) < MIN_CLUSTER_POINTS:
                continue # skipping to next label
            cluster_sizes.append(len(indices))

            cluster = pcd.select_by_index(indices)
            cluster_points = np.asarray(cluster.points)
            cluster_colors = np.asarray(cluster.colors) # if cluster.has_colors() else None
            cluster_color = np.mean(cluster_colors, axis=0)
            # Color Classification
            cube_label = self.classify_color(cluster_color)

            # Compute position of the cube via remaining points
            centroid = np.mean(cluster_points, axis=0)
            cube_x = float(centroid[0])
            cube_y = float(centroid[1])
            cube_z = float(centroid[2])

            cube = CubeMsg()
            cube.color.r = cluster_color[0]
            cube.color.g = cluster_color[1]
            cube.color.b = cluster_color[2]
            cube.color.a = 1.0
            cube.point.header = msg.header
            cube.point.point.x = cube_x
            cube.point.point.y = cube_y
            cube.point.point.z = cube_z

            

            if len(indices) > 10000:
                x_box = cluster_points[:, 0]
                y_box = cluster_points[:, 1]
                z_box = cluster_points[:, 2]
                x_box_min, x_box_max = x_box.min(), x_box.max()
                y_box_min, y_box_max = y_box.min(), y_box.max()
                z_box_min, z_box_max = z_box.min(), z_box.max()
                
                marker = Marker()
                marker.header = msg.header
                marker.ns = "bounding_boxes"
                marker.id = label
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.x = (x_box_min + x_box_max) / 2.0
                marker.pose.position.y = (y_box_min + y_box_max) / 2.0
                marker.pose.position.z = (z_box_min + z_box_max) / 2.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = float(x_box_max - x_box_min)
                marker.scale.y = float(y_box_max - y_box_min)
                marker.scale.z = float(z_box_max - z_box_min)

                # Set Color (e.g., Green with transparency)
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.3 # 30% opaque

                self.obstacle_marker_pub.publish(marker)
                self.get_logger().info(f"Published CUBE Marker for cluster {label}.")

                obstacle_msg = BoxBounds()
                obstacle_msg.x_min = float(x_box_min)
                obstacle_msg.x_max = float(x_box_max)
                obstacle_msg.y_min = float(y_box_min)
                obstacle_msg.y_max = float(y_box_max)
                obstacle_msg.z_min = float(z_box_min)
                obstacle_msg.z_max = float(z_box_max)

                self.obstacles_pub.publish(obstacle_msg)
                self.get_logger().info(f"Published Bounding Box for large cluster (size: {len(indices)}).")
                
            else:
                cube_list.append(cube)
                labeled_cube = LabeledCube()
                labeled_cube.point.header = msg.header
                labeled_cube.point.point.x = cube_x
                labeled_cube.point.point.y = cube_y
                labeled_cube.point.point.z = cube_z
                labeled_cube.color_label = cube_label
                labeled_cube_list.append(labeled_cube)

        self.publish_filtered_points(filtered_points, msg.header)

        cubes = CubeArray()
        cubes.cubes = cube_list
        self.cube_pose_pub.publish(cubes)

        labeled_cubes_array = LabeledCubeArray()
        labeled_cubes_array.header = msg.header
        labeled_cubes_array.cubes = labeled_cube_list
        self.labeled_cube_pub.publish(labeled_cubes_array)
        

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