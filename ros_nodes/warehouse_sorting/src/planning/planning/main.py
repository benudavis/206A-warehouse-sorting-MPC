from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
# TF imports removed - transformations now handled by transform_perception node
from custom_msgs.msg import LabeledCubeArray, LabeledCube, BoxBounds
import numpy as np
from planning.ik import IKPlanner
from planning.mpc_controller import MPCController
from planning.forward_kinematics import ur7e_forward_kinematics_from_angles

class UR7e_CubeGrasp(Node):

    def __init__(self):

        super().__init__('cube_grasp')

        # Subscribe to transformed perception topics (already in base_link frame)
        # These are published by transform_perception node
        self.labeled_cubes_sub = self.create_subscription(
            LabeledCubeArray,
            '/labeled_cubes_base',  # Transformed to base_link by transform_perception
            self.labeled_cubes_callback,
            10
        )

        # Subscribe to transformed obstacles (already in base_link frame)
        self.obstacles_sub = self.create_subscription(
            BoxBounds,
            '/obstacles_base',  # Transformed to base_link by transform_perception
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
        self.gripper_closed = False  # Track gripper state
        self.pick_height = None  # Store z-height when cube is picked (after moving up)
        self.last_job_was_move_up = False  # Track if last job was the "move up after grip" step

        # IK planner node (uses MoveIt services)
        self.ik_planner = IKPlanner()

        # MPC controller for trajectory planning with obstacle avoidance
        # Reduced horizon and increased dt for faster computation
        self.mpc = MPCController(n_joints=6, horizon=6, dt=0.15)  # Faster: smaller horizon, larger dt
        
        # Configure MPC with safety margin for obstacle avoidance
        safety_margin = 0.10
        self.mpc.set_safety_margin(safety_margin)
        self.get_logger().info(f"MPC configured with safety_margin={safety_margin:.3f}m (hard constraint)")

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
        Process LabeledCubeArray already transformed to base_link frame.
        Filters by color (red/blue) and queues for processing.
        """
        if self.joint_state is None:
            self.get_logger().debug("No joint state yet, skipping cube processing")
            return

        if self.processing_cube:
            self.get_logger().debug("Already processing a cube, skipping new detections")
            return

        # Process each cube in the array (already in base_link frame)
        for i, labeled_cube in enumerate(msg.cubes):
            # Only process red and blue cubes
            if labeled_cube.color_label not in ['red', 'blue']:
                continue

            # Create a unique ID for this cube (based on position)
            cube_id = f"{labeled_cube.point.point.x:.4f}_{labeled_cube.point.point.y:.4f}_{labeled_cube.point.point.z:.4f}"
            
            # Skip if already processed
            if cube_id in self.processed_cube_ids:
                continue

            # Cube is already in base_link frame (transformed by transform_perception node)
            cube_pose_base = labeled_cube.point

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

    def obstacles_callback(self, msg: BoxBounds):
        """
        Process obstacle bounds already transformed to base_link frame.
        Converts to (center, half_size) format for MPC.
        
        Args:
            msg: BoxBounds message with x_min, x_max, y_min, y_max, z_min, z_max in base_link frame
        """
        # Obstacle is already in base_link frame (transformed by transform_perception node)
        # Convert to (center, half_size) format for MPC
        center = np.array([
            (msg.x_min + msg.x_max) / 2.0,
            (msg.y_min + msg.y_max) / 2.0,
            (msg.z_min + msg.z_max) / 2.0
        ])
        half_size = np.array([
            (msg.x_max - msg.x_min) / 2.0,
            (msg.y_max - msg.y_min) / 2.0,
            (msg.z_max - msg.z_min) / 2.0
        ])
        
        # Update current obstacles (replace with latest obstacle)
        # Note: cube_detector publishes one obstacle at a time, so we replace
        # If multiple obstacles are needed, we'd need to track them differently
        self.current_obstacles = [(center, half_size)]
        
        # Calculate actual dimensions (full size, not half_size)
        dim_x = (msg.x_max - msg.x_min)
        dim_y = (msg.y_max - msg.y_min)
        dim_z = (msg.z_max - msg.z_min)
        
        self.get_logger().info(
            f"Obstacle in base_link:"
        )
        self.get_logger().info(
            f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m"
        )
        self.get_logger().info(
            f"  Dimensions: {dim_x:.3f} x {dim_y:.3f} x {dim_z:.3f} m (width x depth x height)"
        )
        self.get_logger().info(
            f"  Bounds: x=[{msg.x_min:.3f}, {msg.x_max:.3f}], "
            f"y=[{msg.y_min:.3f}, {msg.y_max:.3f}], "
            f"z=[{msg.z_min:.3f}, {msg.z_max:.3f}] m"
        )
        self.get_logger().info(
            f"  Half-size: ({half_size[0]:.3f}, {half_size[1]:.3f}, {half_size[2]:.3f}) m"
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
        # Reset state for new cube
        self.gripper_closed = False
        self.pick_height = None
        self.last_job_was_move_up = False

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
        # Mark this job so we can detect when it completes and set minimum_z
        self.job_queue.append((pre_grasp_js, False, 'move_up_after_grip'))

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

        if isinstance(next_job, tuple):
            # Handle both (target_js, use_mpc) and (target_js, use_mpc, marker) formats
            if len(next_job) == 2:
                target_js, use_mpc = next_job
                job_marker = None
            elif len(next_job) == 3:
                target_js, use_mpc, job_marker = next_job
            else:
                self.get_logger().error(f"Invalid job tuple format: {next_job}")
                self.execute_jobs()
                return
            
            # Store job marker to detect when "move up after grip" completes
            self.last_job_was_move_up = (job_marker == 'move_up_after_grip')
            
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
        """Toggle gripper and manage minimum_z constraint."""
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.gripper_closed = not self.gripper_closed
        
        if self.gripper_closed:
            self.get_logger().info("Gripper closed. Will set minimum_z constraint after moving up.")
        else:
            # Clear minimum_z when opening gripper
            if hasattr(self.mpc, 'clear_minimum_z'):
                self.mpc.clear_minimum_z()
            self.pick_height = None
            self.get_logger().info("Gripper opened. Cleared minimum_z constraint")

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job
    
    def _compute_end_effector_z(self, joint_state: JointState) -> float:
        """
        Compute end-effector z-coordinate from joint state using forward kinematics.
        
        Args:
            joint_state: Current joint state
            
        Returns:
            Z-coordinate [m] of end-effector in base_link frame, or None if computation fails
        """
        try:
            joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            
            name_to_index = {name: i for i, name in enumerate(joint_state.name)}
            joint_angles = np.array([
                joint_state.position[name_to_index[name]] for name in joint_names
            ], dtype=float)
            
            gst = ur7e_forward_kinematics_from_angles(joint_angles)
            ee_z = gst[2, 3]
            
            return float(ee_z)
        except (KeyError, IndexError, Exception) as e:
            self.get_logger().error(f"Failed to compute end-effector z: {e}")
            return None
    
    def _set_minimum_z_after_move_up(self):
        """
        Set minimum_z constraint after the arm has moved up after gripping.
        This ensures we capture the correct height (after moving up, not at grasp height).
        The constraint prevents the arm from going below this height during MPC-controlled movements.
        """
        if not self.gripper_closed:
            return  # Gripper not closed, don't set constraint
        
        if self.joint_state is None:
            self.get_logger().warn("No joint state available, cannot set minimum_z")
            return
        
        # Compute current end-effector z position (after moving up)
        current_z = self._compute_end_effector_z(self.joint_state)
        if current_z is not None:
            self.pick_height = current_z
            minimum_z = current_z - 0.01  # 1cm margin below current height
            if hasattr(self.mpc, 'set_minimum_z'):
                self.mpc.set_minimum_z(minimum_z)
                self.get_logger().info(
                    f"Set minimum_z constraint: {minimum_z:.3f}m "
                    f"(current EE z={current_z:.3f}m - 1cm margin). "
                    f"Arm will not go below this height during MPC movements."
                )
            else:
                self.get_logger().warn("MPC controller does not support set_minimum_z method")
        else:
            self.get_logger().warn("Could not compute end-effector z, minimum_z not set")

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
            
            # Check if we just completed the "move up after grip" step
            # If so, set minimum_z constraint now (after arm has moved up)
            if self.last_job_was_move_up and self.gripper_closed:
                self._set_minimum_z_after_move_up()
                self.last_job_was_move_up = False  # Reset flag
            
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