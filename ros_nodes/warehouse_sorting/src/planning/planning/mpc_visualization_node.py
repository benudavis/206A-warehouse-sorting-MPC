#!/usr/bin/env python3
"""
ROS2 Node for Real-Time MPC Trajectory Visualization

This node subscribes to ROS2 topics and visualizes:
- MPC trajectory horizons (from /display_planned_path)
- End-effector positions (computed from /joint_states)
- Cubes (from /labeled_cubes_base)
- Obstacles (from /obstacles_base)
- Executed trajectory (tracked from joint states)

The visualization persists even if the main node is killed (Ctrl+C).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from custom_msgs.msg import LabeledCubeArray, LabeledCube, BoxBounds
from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import Point
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import threading
import time
from collections import deque
from planning.forward_kinematics import ur7e_forward_kinematics_from_angles

# Try to import scipy for smooth interpolation
try:
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def draw_box(ax, center, half_size, color='blue', alpha=0.5, edgecolor='black', linewidth=1):
    """Draw a 3D box (axis-aligned) on the given axes."""
    x_min, x_max = center[0] - half_size[0], center[0] + half_size[0]
    y_min, y_max = center[1] - half_size[1], center[1] + half_size[1]
    z_min, z_max = center[2] - half_size[2], center[2] + half_size[2]
    
    vertices = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
    ]
    
    box = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=edgecolor, linewidths=linewidth)
    ax.add_collection3d(box)
    return vertices


class MPCVisualizationNode(Node):
    def __init__(self):
        super().__init__('mpc_visualization_node')
        
        # Data storage
        self.joint_states_history = deque(maxlen=1000)  # Store joint states with timestamps
        self.mpc_horizons = []  # List of MPC horizon trajectories
        self.cubes = []  # Current cube positions
        self.obstacles = []  # Current obstacles
        self.executed_path = []  # Tracked executed end-effector positions
        self.lock = threading.Lock()
        
        # Visualization parameters
        self.update_interval = 0.5  # Update visualization every 0.5 seconds
        self.last_update_time = time.time()
        self.fig = None
        self.ax = None
        self.plot_initialized = False
        self.shutdown_flag = False
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.mpc_traj_sub = self.create_subscription(
            DisplayTrajectory,
            '/display_planned_path',
            self.mpc_trajectory_callback,
            10
        )
        
        self.cubes_sub = self.create_subscription(
            LabeledCubeArray,
            '/labeled_cubes_base',
            self.cubes_callback,
            10
        )
        
        self.obstacles_sub = self.create_subscription(
            BoxBounds,
            '/obstacles_base',
            self.obstacles_callback,
            10
        )
        
        # Initialize matplotlib in a separate thread
        self.viz_thread = threading.Thread(target=self._init_visualization, daemon=True)
        self.viz_thread.start()
        
        self.get_logger().info("MPC Visualization Node started. Visualization will open in a separate window.")
        self.get_logger().info("Press Ctrl+C to save the final visualization and exit.")
    
    def joint_state_callback(self, msg: JointState):
        """Store joint state and compute end-effector position."""
        with self.lock:
            # Extract joint angles (assuming UR7e joint order)
            joint_names = msg.name
            joint_positions = np.array(msg.position)
            
            # Map to standard order (shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3)
            # Try different possible joint name formats
            try:
                name_to_idx = {name: i for i, name in enumerate(joint_names)}
                
                # Try different joint name formats
                ur_joint_order_variants = [
                    ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                     'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
                    ['ur_arm_shoulder_pan_joint', 'ur_arm_shoulder_lift_joint', 'ur_arm_elbow_joint',
                     'ur_arm_wrist_1_joint', 'ur_arm_wrist_2_joint', 'ur_arm_wrist_3_joint'],
                ]
                
                q = None
                for ur_joint_order in ur_joint_order_variants:
                    try:
                        q = np.array([joint_positions[name_to_idx[name]] for name in ur_joint_order])
                        break
                    except KeyError:
                        continue
                
                # If no variant worked, try to use joint names as-is (assuming correct order)
                if q is None:
                    if len(joint_positions) == 6:
                        q = np.array(joint_positions[:6])
                    else:
                        raise ValueError(f"Unexpected number of joints: {len(joint_positions)}")
                
                # Compute end-effector position using forward kinematics
                gst = ur7e_forward_kinematics_from_angles(q)
                ee_pos = gst[0:3, 3]  # Extract position
                
                # Store with timestamp
                self.joint_states_history.append({
                    'time': time.time(),
                    'joint_state': msg,
                    'joint_angles': q,
                    'ee_position': ee_pos
                })
                
                # Track executed path (only add if significantly different from last)
                if len(self.executed_path) == 0 or np.linalg.norm(ee_pos - self.executed_path[-1]) > 0.01:
                    self.executed_path.append(ee_pos.copy())
                    
            except (KeyError, IndexError) as e:
                self.get_logger().warn(f"Could not process joint state: {e}")
    
    def mpc_trajectory_callback(self, msg: DisplayTrajectory):
        """Store MPC trajectory horizon."""
        if not msg.trajectory:
            return
        
        with self.lock:
            # Extract trajectory from DisplayTrajectory
            robot_traj = msg.trajectory[0]  # Get first trajectory
            joint_traj = robot_traj.joint_trajectory
            
            if not joint_traj.points:
                return
            
            # Extract joint angles from trajectory
            joint_names = joint_traj.joint_names
            name_to_idx = {name: i for i, name in enumerate(joint_names)}
            
            # Try different joint name formats
            ur_joint_order_variants = [
                ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
                ['ur_arm_shoulder_pan_joint', 'ur_arm_shoulder_lift_joint', 'ur_arm_elbow_joint',
                 'ur_arm_wrist_1_joint', 'ur_arm_wrist_2_joint', 'ur_arm_wrist_3_joint'],
            ]
            
            try:
                # Find which joint order works
                ur_joint_order = None
                for variant in ur_joint_order_variants:
                    try:
                        # Test if all joint names exist
                        [name_to_idx[name] for name in variant]
                        ur_joint_order = variant
                        break
                    except KeyError:
                        continue
                
                # If no variant worked, assume joint_names are in correct order
                if ur_joint_order is None:
                    if len(joint_names) == 6:
                        ur_joint_order = joint_names
                    else:
                        raise ValueError(f"Unexpected number of joints: {len(joint_names)}")
                
                # Get joint angles for each point
                q_traj = []
                for pt in joint_traj.points:
                    if ur_joint_order == joint_names:
                        # Use positions directly if order matches
                        q = np.array(pt.positions[:6])
                    else:
                        q = np.array([pt.positions[name_to_idx[name]] for name in ur_joint_order])
                    q_traj.append(q)
                
                q_traj = np.array(q_traj)
                
                # Compute end-effector positions for entire trajectory
                ee_traj = []
                for q in q_traj:
                    gst = ur7e_forward_kinematics_from_angles(q)
                    ee_pos = gst[0:3, 3]
                    ee_traj.append(ee_pos)
                
                ee_traj = np.array(ee_traj)
                
                # Store this horizon
                horizon_data = {
                    'time': time.time(),
                    'q_traj': q_traj,
                    'ee_traj': ee_traj
                }
                
                self.mpc_horizons.append(horizon_data)
                
                # Keep only last 50 horizons to avoid memory issues
                if len(self.mpc_horizons) > 50:
                    self.mpc_horizons.pop(0)
                    
            except (KeyError, IndexError) as e:
                self.get_logger().warn(f"Could not process MPC trajectory: {e}")
    
    def cubes_callback(self, msg: LabeledCubeArray):
        """Store cube positions."""
        with self.lock:
            self.cubes = []
            for cube in msg.cubes:
                pos = np.array([cube.point.x, cube.point.y, cube.point.z])
                self.cubes.append({
                    'position': pos,
                    'color': cube.label,
                    'id': cube.id
                })
    
    def obstacles_callback(self, msg: BoxBounds):
        """Store obstacle positions."""
        with self.lock:
            self.obstacles = []
            # BoxBounds contains center and size
            center = np.array([msg.center.x, msg.center.y, msg.center.z])
            half_size = np.array([msg.size.x / 2.0, msg.size.y / 2.0, msg.size.z / 2.0])
            self.obstacles.append((center, half_size))
    
    def _init_visualization(self):
        """Initialize matplotlib visualization in separate thread."""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.plot_initialized = True
        self.shutdown_flag = False
        
        # Set up periodic update
        try:
            while not self.shutdown_flag and rclpy.ok():
                time.sleep(self.update_interval)
                if self.plot_initialized:
                    self._update_visualization()
        except Exception as e:
            self.get_logger().error(f"Visualization thread error: {e}")
        
        # When node shuts down, save the final plot
        self._save_final_visualization()
    
    def _update_visualization(self):
        """Update the 3D visualization with current data."""
        if not self.plot_initialized:
            return
        
        with self.lock:
            # Clear axes
            self.ax.clear()
            
            # Get current data
            current_ee = None
            if self.joint_states_history:
                current_ee = self.joint_states_history[-1]['ee_position']
            
            # Plot MPC horizons
            colors = plt.cm.tab20(np.linspace(0, 1, min(len(self.mpc_horizons), 20)))
            if len(self.mpc_horizons) > 20:
                colors = np.tile(colors, (len(self.mpc_horizons) // 20 + 1, 1))[:len(self.mpc_horizons)]
            
            for i, horizon in enumerate(self.mpc_horizons):
                ee_traj = horizon['ee_traj']
                if len(ee_traj) > 1:
                    alpha = 0.5 + 0.4 * (i / max(len(self.mpc_horizons), 1))
                    color = colors[i] if i < len(colors) else 'blue'
                    
                    # Plot trajectory line
                    self.ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
                               '-', linewidth=2.0, color=color, alpha=alpha, 
                               label=f'Horizon {i}' if i < 10 else None, zorder=3)
                    
                    # Plot points
                    point_sizes = np.ones(len(ee_traj)) * 30
                    point_sizes[0] = 60
                    point_sizes[-1] = 60
                    
                    self.ax.scatter(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
                                   s=point_sizes, color=color, alpha=alpha*1.2, 
                                   marker='o', edgecolors='black', linewidths=1.0, zorder=4)
            
            # Plot executed path
            if len(self.executed_path) > 1:
                executed_path_array = np.array(self.executed_path)
                
                # Smooth interpolation if scipy available
                if HAS_SCIPY and len(executed_path_array) >= 2:
                    try:
                        # Create parameterization
                        distances = np.zeros(len(executed_path_array))
                        for i in range(1, len(executed_path_array)):
                            distances[i] = distances[i-1] + np.linalg.norm(executed_path_array[i] - executed_path_array[i-1])
                        
                        if distances[-1] > 0:
                            distances = distances / distances[-1]
                            t_smooth = np.linspace(0, 1, max(50, len(executed_path_array) * 10))
                            
                            kind = 'cubic' if len(executed_path_array) >= 4 else 'linear'
                            fx = interp1d(distances, executed_path_array[:, 0], kind=kind, 
                                         bounds_error=False, fill_value='extrapolate')
                            fy = interp1d(distances, executed_path_array[:, 1], kind=kind, 
                                         bounds_error=False, fill_value='extrapolate')
                            fz = interp1d(distances, executed_path_array[:, 2], kind=kind, 
                                         bounds_error=False, fill_value='extrapolate')
                            
                            path_smooth = np.array([fx(t_smooth), fy(t_smooth), fz(t_smooth)]).T
                            
                            self.ax.plot(path_smooth[:, 0], path_smooth[:, 1], path_smooth[:, 2],
                                        'k-', linewidth=4, label='Executed Path', zorder=12, alpha=0.9)
                    except:
                        # Fallback to linear
                        self.ax.plot(executed_path_array[:, 0], executed_path_array[:, 1], executed_path_array[:, 2],
                                   'k-', linewidth=4, label='Executed Path', zorder=12, alpha=0.9)
                else:
                    self.ax.plot(executed_path_array[:, 0], executed_path_array[:, 1], executed_path_array[:, 2],
                               'k-', linewidth=4, label='Executed Path', zorder=12, alpha=0.9)
                
                # Plot waypoints
                self.ax.scatter(executed_path_array[:, 0], executed_path_array[:, 1], executed_path_array[:, 2],
                               c='black', s=100, marker='D', zorder=13, edgecolors='white', linewidths=2,
                               label='Executed Waypoints')
            
            # Plot current end-effector position
            if current_ee is not None:
                self.ax.scatter([current_ee[0]], [current_ee[1]], [current_ee[2]],
                               c='green', s=200, marker='s', label='Current EE', zorder=15,
                               edgecolors='black', linewidths=2)
            
            # Plot cubes
            for cube in self.cubes:
                pos = cube['position']
                color_map = {'red': 'red', 'blue': 'blue', 'green': 'green'}
                cube_color = color_map.get(cube['color'], 'gray')
                self.ax.scatter([pos[0]], [pos[1]], [pos[2]], c=cube_color, s=200, 
                               marker='o', label=f'Cube ({cube["color"]})' if cube == self.cubes[0] else None,
                               zorder=14, edgecolors='black', linewidths=2)
            
            # Plot obstacles
            for center, half_size in self.obstacles:
                draw_box(self.ax, center, half_size, color='orange', alpha=0.5, 
                        edgecolor='darkorange', linewidth=2)
                
                # Draw safety margin (assuming 0.10m)
                safety_margin = 0.10
                inflated_half_size = half_size + safety_margin
                draw_box(self.ax, center, inflated_half_size, color='yellow', alpha=0.15,
                        edgecolor='yellow', linewidth=1)
            
            # Set labels and title
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            
            title = f'MPC Trajectory Visualization (Real-Time)\n'
            title += f'Horizons: {len(self.mpc_horizons)}, Executed Points: {len(self.executed_path)}, '
            title += f'Cubes: {len(self.cubes)}, Obstacles: {len(self.obstacles)}'
            self.ax.set_title(title, fontsize=10)
            
            # Set equal aspect ratio
            if len(self.executed_path) > 0 or len(self.mpc_horizons) > 0:
                all_points = []
                if len(self.executed_path) > 0:
                    all_points.append(np.array(self.executed_path))
                for horizon in self.mpc_horizons:
                    all_points.append(horizon['ee_traj'])
                if len(all_points) > 0:
                    all_points = np.vstack(all_points)
                    center = all_points.mean(axis=0)
                    max_range = np.abs(all_points - center).max()
                    self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
                    self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
                    self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
            # Add legend
            self.ax.legend(loc='upper left', fontsize=8)
            
            # Update plot
            plt.draw()
            plt.pause(0.01)
    
    def _save_final_visualization(self):
        """Save the final visualization when node shuts down."""
        if not self.plot_initialized or self.fig is None:
            return
        
        try:
            with self.lock:
                # Final update
                self._update_visualization()
                
                # Save figure
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mpc_visualization_{timestamp}.png"
                
                try:
                    self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"\n{'='*70}")
                    print(f"✓ Saved final visualization to: {os.path.abspath(filename)}")
                    print(f"{'='*70}\n")
                except Exception as e:
                    print(f"\n⚠️  Failed to save visualization: {e}\n")
        except Exception as e:
            print(f"Error in _save_final_visualization: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MPCVisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Received Ctrl+C. Saving visualization and shutting down...")
        node.shutdown_flag = True
        # Give visualization thread time to save
        time.sleep(1.0)
    except Exception as e:
        node.get_logger().error(f"Error in main: {e}")
        node.shutdown_flag = True
        time.sleep(1.0)
    finally:
        node.shutdown_flag = True
        node.destroy_node()
        rclpy.shutdown()
        
        # Keep plot open even after ROS2 shutdown
        if node.plot_initialized and node.fig is not None:
            node.get_logger().info("Visualization window will remain open. Close it manually or it will auto-close.")
            try:
                plt.ioff()
                plt.show(block=True)
            except:
                pass


if __name__ == '__main__':
    main()
