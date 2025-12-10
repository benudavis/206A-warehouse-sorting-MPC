from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
from rclpy.duration import Duration
from custom_msgs.msg import LabeledCubeArray, LabeledCube, BoxBounds
import numpy as np
from planning.ik import IKPlanner
from planning.mpc_controller import MPCController

class UR7e_CubeGrasp(Node):

    def __init__(self):

        super().__init__('cube_grasp')

        # TF buffer & listener for transforming cube poses from camera to base_link
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to labeled cubes from cube_detector
        self.labeled_cubes_sub = self.create_subscription(
            LabeledCubeArray,
            '/labeled_cubes',
            self.labeled_cubes_callback,
            10
        )

        # Subscribe to obstacles from cube_detector
        self.obstacles_sub = self.create_subscription(
            BoxBounds,
            '/obstacles',
            self.obstacles_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            1
        )

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        # Hardcoded drop locations for color-based sorting
        # Red cubes drop location (in base_link frame)
        self.red_drop_location = [0.5, 0.2, 0.15]  # [x, y, z] in meters
        # Blue cubes drop location (in base_link frame)
        self.blue_drop_location = [0.5, -0.2, 0.15]  # [x, y, z] in meters

        # State management
        self.joint_state = None
        self.cube_queue = []  # Queue of (cube_pose, color, drop_location) tuples
        self.processing_cube = False  # Flag to prevent processing multiple cubes simultaneously
        self.processed_cube_ids = set()  # Track processed cubes to avoid duplicates
        self.current_obstacles = []  # List of (center, half_size) tuples in base_link frame
        self.camera_frame_id = None  # Track camera frame from labeled_cubes messages

        # IK planner node (uses MoveIt services)
        self.ik_planner = IKPlanner()

        # MPC controller for trajectory planning with obstacle avoidance
        self.mpc = MPCController(n_joints=6, horizon=10, dt=0.1)  # Increased dt for faster execution

        # Publisher for MPC trajectory visualization in RViz
        self.mpc_traj_pub = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            1
        )

        # Entries should be either:
        #   - (JointState, use_mpc: bool) for joint movements
        #   - 'toggle_grip' for gripper actions
        self.job_queue = []

    def joint_state_callback(self, msg: JointState):

        self.joint_state = msg

    def labeled_cubes_callback(self, msg: LabeledCubeArray):
        """
        Process LabeledCubeArray from cube_detector.
        Filters by color (red/blue), transforms to base_link, and queues for processing.
        """
        # Store camera frame_id for obstacle transformation
        if msg.cubes:
            self.camera_frame_id = msg.cubes[0].point.header.frame_id
        
        if self.joint_state is None:
            self.get_logger().debug("No joint state yet, skipping cube processing")
            return

        if self.processing_cube:
            self.get_logger().debug("Already processing a cube, skipping new detections")
            return

        # Process each cube in the array
        for i, labeled_cube in enumerate(msg.cubes):
            # Only process red and blue cubes
            if labeled_cube.color_label not in ['red', 'blue']:
                continue

            # Create a unique ID for this cube (based on position and timestamp)
            cube_id = f"{labeled_cube.point.point.x:.4f}_{labeled_cube.point.point.y:.4f}_{labeled_cube.point.point.z:.4f}"
            
            # Skip if already processed
            if cube_id in self.processed_cube_ids:
                continue

            # Transform cube position from camera frame to base_link
            cube_pose_base = self._transform_cube_to_base(labeled_cube.point)
            if cube_pose_base is None:
                self.get_logger().warn(f"Failed to transform cube {i} to base_link")
                continue

            # Determine drop location based on color
            if labeled_cube.color_label == 'red':
                drop_location = self.red_drop_location
            elif labeled_cube.color_label == 'blue':
                drop_location = self.blue_drop_location
            else:
                continue  # Should not reach here due to filter above

            # Add to queue
            self.cube_queue.append((cube_pose_base, labeled_cube.color_label, drop_location))
            self.processed_cube_ids.add(cube_id)
            
            self.get_logger().info(
                f"Queued {labeled_cube.color_label} cube at "
                f"({cube_pose_base.point.x:.3f}, {cube_pose_base.point.y:.3f}, {cube_pose_base.point.z:.3f}) "
                f"for drop at ({drop_location[0]:.3f}, {drop_location[1]:.3f}, {drop_location[2]:.3f})"
            )

        # Start processing if queue has items and not currently processing
        if self.cube_queue and not self.processing_cube:
            self._process_next_cube()

    def _transform_cube_to_base(self, point_stamped: PointStamped) -> PointStamped:
        """
        Transform cube position from camera frame to base_link frame.
        
        Args:
            point_stamped: PointStamped in camera frame
            
        Returns:
            PointStamped in base_link frame, or None on failure
        """
        target_frame = 'base_link'
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

    def obstacles_callback(self, msg: BoxBounds):
        """
        Process obstacle bounds from cube_detector.
        Transforms bounding box from camera frame to base_link and updates MPC obstacles.
        
        Args:
            msg: BoxBounds message with x_min, x_max, y_min, y_max, z_min, z_max in camera frame
        """
        # Use the camera frame_id from the most recent labeled_cubes message
        # If not available, try common camera frame names
        if self.camera_frame_id is None:
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
        # Use extrapolated z values
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
            self.get_logger().warn(f"Could not find transform {source_frame} -> {target_frame} for obstacles: {e}")
            return
        
        # Transform each corner to base_link
        corners_base = []
        
        # Transform all corners
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
        
        # Convert to (center, half_size) format for MPC
        center = np.array([
            (x_min_base + x_max_base) / 2.0,
            (y_min_base + y_max_base) / 2.0,
            (z_min_base + z_max_base) / 2.0
        ])
        half_size = np.array([
            (x_max_base - x_min_base) / 2.0,
            (y_max_base - y_min_base) / 2.0,
            (z_max_base - z_min_base) / 2.0
        ])
        
        # Update current obstacles (replace with latest obstacle)
        # Note: cube_detector publishes one obstacle at a time, so we replace
        # If multiple obstacles are needed, we'd need to track them differently
        self.current_obstacles = [(center, half_size)]
        
        self.get_logger().info(
            f"Updated obstacle in base_link: center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), "
            f"half_size=({half_size[0]:.3f}, {half_size[1]:.3f}, {half_size[2]:.3f}), "
            f"z_extrapolated in camera frame: [{msg.z_min:.3f}, {msg.z_max:.3f}] -> [{z_min_extended:.3f}, {z_max_extended:.3f}]"
        )

    def _process_next_cube(self):
        """
        Process the next cube from the queue.
        """
        if not self.cube_queue or self.processing_cube:
            return

        if self.joint_state is None:
            self.get_logger().warn("No joint state available, cannot process cube")
            return

        # Get next cube from queue
        cube_pose, color, drop_location = self.cube_queue.pop(0)
        self.processing_cube = True

        self.get_logger().info(
            f"Processing {color} cube at "
            f"({cube_pose.point.x:.3f}, {cube_pose.point.y:.3f}, {cube_pose.point.z:.3f})"
        )

        # Build job queue for this cube
        self._build_cube_job_queue(cube_pose, drop_location)

        # Start executing
        self.execute_jobs()

    def _build_cube_job_queue(self, cube_pose: PointStamped, drop_location: list):
        """
        Build the job queue for picking up a cube and placing it at drop_location.
        
        Args:
            cube_pose: PointStamped in base_link frame with cube position
            drop_location: [x, y, z] target drop location in base_link frame
        """
        # Clear any existing job queue
        self.job_queue = []

        # Base cube position (in base_link frame)
        cx = cube_pose.point.x
        cy = cube_pose.point.y
        cz = cube_pose.point.z

        # 1) Move to Pre-Grasp Position (gripper above the cube)
        # Offsets:
        #   x offset: 0.0
        #   y offset: -0.035
        #   z offset: +0.185
        pre_x = cx + 0.0
        pre_y = cy - 0.035
        pre_z = cz + 0.185
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error("IK failed for pre-grasp pose.")
            self.processing_cube = False
            return
        # Use MoveIt for pre-grasp (picking sequence)
        self.job_queue.append((pre_grasp_js, False))

        # 2) Move to Grasp Position (lower the gripper to the cube)
        # DO NOT CHANGE z offset lower than +0.16
        grasp_x = cx + 0.0
        grasp_y = cy - 0.035
        grasp_z = cz + 0.16
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error("IK failed for grasp pose.")
            self.processing_cube = False
            return
        # Use MoveIt for grasp (picking sequence)
        self.job_queue.append((grasp_js, False))

        # 3) Close the gripper
        self.job_queue.append('toggle_grip')

        # 4) Move back to Pre-Grasp Position (lift the block)
        # Use MoveIt for lift (picking sequence)
        self.job_queue.append((pre_grasp_js, False))

        # 5) Move to release Position (use drop_location instead of hardcoded offset)
        rel_x = drop_location[0]
        rel_y = drop_location[1]
        rel_z = drop_location[2]
        release_js = self.ik_planner.compute_ik(self.joint_state, rel_x, rel_y, rel_z)
        if release_js is None:
            self.get_logger().error("IK failed for release pose.")
            self.processing_cube = False
            return
        # Use MPC for release movement (with obstacle avoidance)
        self.job_queue.append((release_js, True))

        # 6) Release the gripper
        self.job_queue.append('toggle_grip')

    def execute_jobs(self):

        if not self.job_queue:
            self.get_logger().info("All jobs completed for current cube.")
            # Mark that we're done processing this cube
            self.processing_cube = False
            # Process next cube if available
            if self.cube_queue:
                self._process_next_cube()
            else:
                self.get_logger().info("No more cubes in queue. Waiting for new detections...")
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, tuple) and len(next_job) == 2:
            # Job is (JointState, use_mpc: bool)
            target_js, use_mpc = next_job
            
            if self.joint_state is None:
                self.get_logger().error("No current joint state; cannot plan trajectory.")
                return
            
            if use_mpc:
                # Use MPC for release movement (with obstacle avoidance)
                traj = self._plan_with_mpc(self.joint_state, target_js)
                if traj is None:
                    self.get_logger().error("MPC failed to plan trajectory")
                    return
                self.get_logger().info("MPC planned trajectory")
                self._execute_joint_trajectory(traj)
            else:
                # Use MoveIt plan_to_joints for picking sequence (pre-grasp, grasp, lift)
                traj = self.ik_planner.plan_to_joints(target_js)
                if traj is None:
                    self.get_logger().error("Failed to plan to position using MoveIt")
                    return
                self.get_logger().info("MoveIt planned trajectory")
                self._execute_joint_trajectory(traj.joint_trajectory)
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next jobplan_to_joints

    def _plan_with_mpc(self, current_js: JointState, target_js: JointState) -> JointTrajectory:
        """
        Plan a trajectory from current joint state to target using MPC.
        
        Args:
            current_js: Current joint state
            target_js: Target joint state
            
        Returns:
            JointTrajectory message ready to execute
        """
        # Extract joint positions, ensuring consistent ordering
        name_to_index = {name: i for i, name in enumerate(current_js.name)}
        
        try:
            q_current = np.array(
                [current_js.position[name_to_index[name]] for name in target_js.name],
                dtype=float,
            )
        except KeyError as e:
            self.get_logger().error(f"Joint name mismatch: {e}")
            return None
        
        q_target = np.array(target_js.position, dtype=float)
        
        # Extract actual joint velocities from current state
        try:
            dq_current = np.array(
                [current_js.velocity[name_to_index[name]] for name in target_js.name],
                dtype=float,
            )
        except (KeyError, IndexError):
            # Fallback to zero if velocities not available
            self.get_logger().warn("Velocities not available in joint state, using zero")
            dq_current = np.zeros_like(q_current)
        
        # Build current state [q, dq]
        current_state = np.concatenate([q_current, dq_current])
        
        # Update MPC obstacles with current obstacles (transformed to base_link)
        self.mpc.clear_obstacles()
        for center, half_size in self.current_obstacles:
            self.mpc.add_obstacle(center, half_size)
        
        # Solve MPC
        q_next, q_traj = self.mpc.compute_control(current_state, q_target)
        
        # Convert MPC trajectory to JointTrajectory format
        jt = JointTrajectory()
        jt.joint_names = list(target_js.name)
        jt.header.stamp = self.get_clock().now().to_msg()
        jt.header.frame_id = "base_link"
        
        # Add trajectory points with timing and velocities
        from builtin_interfaces.msg import Duration
        for k in range(q_traj.shape[0]):
            pt = JointTrajectoryPoint()
            pt.positions = q_traj[k].tolist()
            
            # Compute velocities from position differences
            if k < q_traj.shape[0] - 1:
                velocities = (q_traj[k + 1] - q_traj[k]) / self.mpc.dt
                pt.velocities = velocities.tolist()
            else:
                # Last point: zero velocity (or use previous velocity)
                if k > 0:
                    velocities = (q_traj[k] - q_traj[k - 1]) / self.mpc.dt
                    pt.velocities = velocities.tolist()
                else:
                    pt.velocities = [0.0] * len(q_traj[k])
            
            # Time from start based on MPC dt
            t = float(k) * self.mpc.dt
            secs = int(t)
            nsecs = int((t - secs) * 1e9)
            pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
            
            jt.points.append(pt)
        
        # Publish trajectory for visualization in RViz
        self._publish_mpc_trajectory(current_js, jt)
        
        return jt

    def _publish_mpc_trajectory(self, current_js: JointState, joint_traj: JointTrajectory):
        """
        Publish the MPC trajectory as DisplayTrajectory for visualization in RViz.
        
        RViz setup:
          - Add "MotionPlanning" display
          - In "Planned Path" tab, set topic to /display_planned_path
        """
        # RobotState: use current joint state as the starting state
        robot_state = RobotState()
        robot_state.joint_state = current_js

        # RobotTrajectory: wrap the JointTrajectory
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj

        # DisplayTrajectory message
        display_msg = DisplayTrajectory()
        display_msg.model_id = "ur"  # UR robot model name for MoveIt
        display_msg.trajectory_start = robot_state
        display_msg.trajectory.append(robot_traj)

        self.mpc_traj_pub.publish(display_msg)
        self.get_logger().debug("Published MPC trajectory to /display_planned_path for RViz visualization")

    def _toggle_gripper(self):

        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

    def _execute_joint_trajectory(self, joint_traj):

        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):

        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            # Reset processing flag on error so we can try again
            self.processing_cube = False

def main(args=None):

    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
