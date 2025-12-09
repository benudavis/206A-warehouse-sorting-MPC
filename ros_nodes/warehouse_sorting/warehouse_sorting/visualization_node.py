#!/usr/bin/env python3
"""
Scene Manager Node (Upgraded Visualization)

Manages world state and publishes visualization markers for:
- Robot model (URDF)
- Boxes (with attachment logic for pick/place)
- Baskets
- Obstacles
- MPC trajectory
- Collision warnings
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3, TransformStamped
from std_msgs.msg import ColorRGBA, Header, String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters


class BoxState:
    """Represents state of a box in the scene."""
    def __init__(self, name, color, position, size=0.03):
        self.name = name
        self.color_name = color
        self.pose = Pose()
        self.pose.position.x = position[0]
        self.pose.position.y = position[1]
        self.pose.position.z = position[2]
        self.pose.orientation.w = 1.0
        self.size = size
        self.attached = False
        self.attached_to_frame = None
        # Offset when attached to gripper (computed when grasped)
        self.offset_in_gripper = Pose()
        
    def get_color_rgba(self):
        """Return RGBA color based on color name."""
        if 'red' in self.color_name:
            return [0.9, 0.1, 0.1, 1.0]
        elif 'blue' in self.color_name:
            return [0.1, 0.1, 0.9, 1.0]
        else:
            return [0.5, 0.5, 0.5, 1.0]


class SceneManagerNode(Node):
    """Manages scene state and publishes visualization markers."""

    def __init__(self):
        super().__init__('scene_manager')
        
        # Publishers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/visualization_markers',
            10
        )
        
        self.box_markers_pub = self.create_publisher(
            MarkerArray,
            '/box_markers',
            10
        )
        
        self.trajectory_marker_pub = self.create_publisher(
            Marker,
            '/mpc_trajectory_viz',
            10
        )
        
        # Republish robot_description as topic (foxglove_bridge needs topics, not params)
        self.robot_desc_pub = self.create_publisher(
            String,
            '/robot_description',
            10
        )
        
        # Subscribers
        self.traj_sub = self.create_subscription(
            JointTrajectory,
            '/mpc/joint_trajectory',
            self.trajectory_callback,
            10
        )
        
        # Listen for gripper events
        self.gripper_sub = self.create_subscription(
            String,
            '/gripper_events',
            self.handle_gripper_event,
            10
        )
        
        # TF for getting end-effector pose when box is attached
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Fetch and republish robot_description from robot_state_publisher
        self.fetch_and_republish_urdf()
        
        # Timer for periodic visualization updates
        self.timer = self.create_timer(0.1, self.publish_scene)
        
        # Box states (managed dynamically)
        self.box_states = {
            "red_1": BoxState("red_1", "red", [0.35, -0.28, 0.52], 0.030),
            "red_2": BoxState("red_2", "red", [0.40, -0.32, 0.52], 0.028),
            "red_3": BoxState("red_3", "red", [0.45, -0.28, 0.52], 0.032),
            "blue_1": BoxState("blue_1", "blue", [0.35, -0.42, 0.52], 0.030),
            "blue_2": BoxState("blue_2", "blue", [0.40, -0.38, 0.52], 0.028),
            "blue_3": BoxState("blue_3", "blue", [0.45, -0.42, 0.52], 0.032),
        }
        
        # Basket positions (static)
        self.baskets = [
            {"name": "red_basket", "pos": [-0.45, -0.10, 0.48], "color": [0.8, 0.2, 0.2, 0.5]},
            {"name": "blue_basket", "pos": [-0.45, -0.60, 0.48], "color": [0.2, 0.2, 0.8, 0.5]},
        ]
        
        # Wall obstacle (vertical wall between table and baskets)
        self.obstacles = [
            {"name": "wall", "pos": [-0.15, -0.35, 0.5], "size": [0.01, 0.6, 0.8], "color": [0.6, 0.4, 0.2, 0.7]},
        ]
        
        self.current_trajectory = None
        
        self.get_logger().info('Visualization Node ready - publishing to Foxglove')

    def trajectory_callback(self, msg):
        """Receive MPC trajectory for visualization."""
        self.current_trajectory = msg
    
    def fetch_and_republish_urdf(self):
        """
        Fetch robot_description parameter from robot_state_publisher
        and republish as topic for Foxglove.
        """
        # Create a timer to periodically republish URDF
        self.create_timer(1.0, self.republish_urdf_callback)
        self.urdf_fetched = False
        self.get_logger().info('Started URDF republisher (waiting for robot_state_publisher...)')
    
    def republish_urdf_callback(self):
        """Periodically try to fetch and republish URDF."""
        if self.urdf_fetched:
            return
        
        try:
            # Create parameter client for robot_state_publisher
            from rclpy.parameter_client import SyncParametersClient
            
            param_client = SyncParametersClient(self, '/robot_state_publisher')
            
            if param_client.wait_for_node(timeout_sec=1.0):
                # Get robot_description parameter
                params = param_client.get_parameters(['robot_description'])
                
                if len(params) > 0 and params[0].value:
                    urdf_string = params[0].value
                    
                    # Publish to topic
                    msg = String()
                    msg.data = urdf_string
                    
                    # Publish multiple times
                    for _ in range(10):
                        self.robot_desc_pub.publish(msg)
                    
                    self.get_logger().info(f'✓ Republished UR5e URDF to /robot_description topic ({len(urdf_string)} bytes)')
                    self.urdf_fetched = True
                    
        except Exception as e:
            # Silently retry (robot_state_publisher might not be ready yet)
            pass
    
    def handle_gripper_event(self, msg):
        """Handle gripper events: grasp:box_name or release:box_name:basket_name"""
        try:
            parts = msg.data.split(':')
            action = parts[0]
            box_name = parts[1]
            
            if box_name not in self.box_states:
                self.get_logger().warn(f'Unknown box: {box_name}')
                return
            
            box = self.box_states[box_name]
            
            if action == 'grasp':
                # Attach box to end-effector
                box.attached = True
                box.attached_to_frame = 'tool0'  # UR5e end-effector frame
                # Store offset (box position relative to gripper when grasped)
                box.offset_in_gripper.position.x = 0.0
                box.offset_in_gripper.position.y = 0.0
                box.offset_in_gripper.position.z = 0.05  # 5cm below gripper
                box.offset_in_gripper.orientation.w = 1.0
                self.get_logger().info(f'✓ Grasped {box_name}')
                
            elif action == 'release':
                # Detach box and place at basket
                box.attached = False
                if len(parts) > 2:
                    basket_name = parts[2]
                    # Find basket position
                    for basket in self.baskets:
                        if basket["name"] == basket_name:
                            # Place box in basket (slightly above surface)
                            box.pose.position.x = basket["pos"][0]
                            box.pose.position.y = basket["pos"][1]
                            box.pose.position.z = basket["pos"][2] + 0.05
                            self.get_logger().info(f'✓ Placed {box_name} in {basket_name}')
                            break
                else:
                    # Just release at current position
                    self.get_logger().info(f'✓ Released {box_name}')
                    
        except Exception as e:
            self.get_logger().error(f'Error handling gripper event: {e}')

    def publish_scene(self):
        """Publish all scene markers."""
        marker_array = MarkerArray()
        
        # Ground plane
        ground = self.create_box_marker(
            id=0,
            name="ground",
            pos=[0.0, 0.0, 0.0],
            size=[4.0, 4.0, 0.01],
            color=[0.3, 0.3, 0.3, 1.0]
        )
        marker_array.markers.append(ground)
        
        # Table
        table = self.create_box_marker(
            id=1,
            name="table",
            pos=[0.0, -0.4, 0.2],
            size=[0.8, 0.5, 0.2],
            color=[0.5, 0.46, 0.5, 1.0]
        )
        marker_array.markers.append(table)
        
        # Boxes (handle attachment to gripper)
        box_array = MarkerArray()
        for i, (box_name, box_state) in enumerate(self.box_states.items()):
            # Determine box position
            if box_state.attached:
                # Get end-effector pose from TF
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'base_link',  # target frame
                        box_state.attached_to_frame,  # source frame (tool0)
                        rclpy.time.Time(),  # latest
                        Duration(seconds=0.1)  # timeout
                    )
                    
                    # Transform offset to world frame
                    pose_stamped = tf2_geometry_msgs.PoseStamped()
                    pose_stamped.pose = box_state.offset_in_gripper
                    pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
                    box_pos = [
                        pose_transformed.pose.position.x,
                        pose_transformed.pose.position.y,
                        pose_transformed.pose.position.z
                    ]
                except Exception as e:
                    # If TF lookup fails, use last known position
                    box_pos = [
                        box_state.pose.position.x,
                        box_state.pose.position.y,
                        box_state.pose.position.z
                    ]
            else:
                # Use stored world position
                box_pos = [
                    box_state.pose.position.x,
                    box_state.pose.position.y,
                    box_state.pose.position.z
                ]
            
            marker = self.create_box_marker(
                id=100 + i,
                name=box_name,
                pos=box_pos,
                size=[box_state.size] * 3,
                color=box_state.get_color_rgba()
            )
            box_array.markers.append(marker)
            marker_array.markers.append(marker)
        
        # Baskets
        for i, basket in enumerate(self.baskets):
            marker = self.create_box_marker(
                id=200 + i,
                name=basket["name"],
                pos=basket["pos"],
                size=[0.15, 0.15, 0.02],
                color=basket["color"]
            )
            marker_array.markers.append(marker)
        
        # Obstacles (wall)
        for i, obstacle in enumerate(self.obstacles):
            marker = self.create_box_marker(
                id=300 + i,
                name=obstacle["name"],
                pos=obstacle["pos"],
                size=obstacle["size"],
                color=obstacle["color"]
            )
            marker_array.markers.append(marker)
        
        # Publish markers
        self.marker_pub.publish(marker_array)
        self.box_markers_pub.publish(box_array)
        
        # Publish trajectory if available
        if self.current_trajectory is not None:
            self.publish_trajectory_marker()

    def create_box_marker(self, id, name, pos, size, color):
        """Create a box marker."""
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "world"
        
        marker.ns = "scene"
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(pos[0])
        marker.pose.position.y = float(pos[1])
        marker.pose.position.z = float(pos[2])
        marker.pose.orientation.w = 1.0
        
        if isinstance(size, list):
            marker.scale.x = float(size[0])
            marker.scale.y = float(size[1])
            marker.scale.z = float(size[2])
        else:
            marker.scale.x = marker.scale.y = marker.scale.z = float(size)
        
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        
        marker.lifetime.sec = 0  # Persistent
        
        return marker

    def publish_trajectory_marker(self):
        """Publish MPC trajectory as a line strip."""
        if not self.current_trajectory or not self.current_trajectory.points:
            return
        
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "world"
        
        marker.ns = "mpc_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.01  # Line width
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        # TODO: Convert joint trajectory to Cartesian positions using FK
        # For now, just create a simple visualization
        marker.lifetime.sec = 0
        
        self.trajectory_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SceneManagerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
