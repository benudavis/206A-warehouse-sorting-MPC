from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from custom_msgs.msg import LabeledCubeArray, LabeledCube, BoxBounds
from planning.ik import IKPlanner
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
import numpy as np
from planning.mpc_controller import MPCController
from planning.forward_kinematics import ur7e_forward_kinematics_from_angles


class UR7e_CubeGrasp(Node):

    def __init__(self):

        super().__init__('cube_grasp')

        # Subscriptions
        self.labeled_cubes_sub = self.create_subscription(
            LabeledCubeArray,
            '/labeled_cubes_base',
            self.labeled_cubes_callback,
            10
        )

        self.obstacles_sub = self.create_subscription(
            BoxBounds,
            '/obstacles_base',
            self.obstacles_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            1
        )

        # Action client for arm trajectories
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Service client for gripper (toggle)
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Fixed drop locations in base_link frame
        self.red_drop_location = PointStamped()
        self.red_drop_location.header.frame_id = "base_link"
        self.red_drop_location.point.x = -0.35
        self.red_drop_location.point.y = 0.50
        self.red_drop_location.point.z = 0.0

        self.blue_drop_location = PointStamped()
        self.blue_drop_location.header.frame_id = "base_link"
        self.blue_drop_location.point.x = -0.35
        self.blue_drop_location.point.y = 0.74
        self.blue_drop_location.point.z = 0.0

        # State
        self.joint_state = None
        self.initial_joint_state = None
        self.cube_queue = []         # list of (PointStamped, color, drop_location)
        self.processing_cube = False
        self.completed_cubes = []    # list of (x, y, z) that have been successfully processed
        self.current_cube_pose = None
        self.current_cube_color = None
        self.obstacle_top_z = None
        self.clearance_height = 0.35
        self.clearance_margin = 0.15

        self.ik_planner = IKPlanner()
        self.job_queue = []          # list of JointState | 'toggle_grip' | ('return_home', JointState)
        self.smooth_waypoints_count = 3

        # MPC
        self.mpc = MPCController(n_joints=6, horizon=40, dt=0.05)  # Faster: smaller horizon, larger dt
        # Configure MPC with safety margin for obstacle avoidance
        safety_margin = 0.01
        self.mpc.set_safety_margin(safety_margin)

        # Publisher for MPC trajectory visualization in RViz
        self.mpc_traj_pub = self.create_publisher(
            Marker,
            '/mpc_trajectory',
            1
        )

    # --------------------------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------------------------

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

        if self.initial_joint_state is None:
            self.initial_joint_state = JointState()
            self.initial_joint_state.header = msg.header
            self.initial_joint_state.name = list(msg.name)
            self.initial_joint_state.position = list(msg.position)
            self.initial_joint_state.velocity = list(msg.velocity) if msg.velocity else []
            self.initial_joint_state.effort = list(msg.effort) if msg.effort else []
            self.get_logger().info("Stored initial/home joint state")
            self.get_logger().info("Node is ready. Waiting for cube detections...")

    def labeled_cubes_callback(self, msg: LabeledCubeArray):
        """Route cubes to drop locations based on color (red/blue)."""
        if self.joint_state is None:
            self.get_logger().debug("No joint state yet, skipping cube processing")
            return

        # Skip processing if we're currently handling a cube to prevent position updates
        if self.processing_cube:
            self.get_logger().debug(
                "Currently processing a cube, skipping new detections to prevent position updates"
            )
            return

        for labeled_cube in msg.cubes:
            if labeled_cube.color_label not in ['red', 'black']:
                continue

            cube_pose_stamped = labeled_cube.point

            # Ensure we have some frame; if empty, assume base_link
            if not cube_pose_stamped.header.frame_id:
                cube_pose_stamped.header.frame_id = "base_link"

            cube_pose_stamped.header.stamp = self.get_clock().now().to_msg()

            # Transform to base_link if needed
            if cube_pose_stamped.header.frame_id != "base_link":
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "base_link",
                        cube_pose_stamped.header.frame_id,
                        Time()
                    )
                    cube_pose_base = do_transform_point(cube_pose_stamped, transform)
                except Exception as e:
                    self.get_logger().error(
                        f"Failed to transform cube from {cube_pose_stamped.header.frame_id} "
                        f"to base_link: {e}. Skipping cube."
                    )
                    continue
            else:
                cube_pose_base = cube_pose_stamped

            cube_x = cube_pose_base.point.x
            cube_y = cube_pose_base.point.y
            cube_z = cube_pose_base.point.z

            # Skip cubes that are very close to ones we've already successfully processed
            skip_due_to_completed = False
            for px, py, pz in self.completed_cubes:
                dist = np.sqrt((cube_x - px) ** 2 + (cube_y - py) ** 2 + (cube_z - pz) ** 2)
                if dist < 0.05:  # within 5 cm of a completed cube
                    skip_due_to_completed = True
                    break
            if skip_due_to_completed:
                continue

            # Skip cubes that are already in the queue (camera updates)
            already_queued = False
            for queued_cube_pose, _, _ in self.cube_queue:
                qx = queued_cube_pose.point.x
                qy = queued_cube_pose.point.y
                qz = queued_cube_pose.point.z
                dist = np.sqrt((cube_x - qx) ** 2 + (cube_y - qy) ** 2 + (cube_z - qz) ** 2)
                if dist < 0.05:  # within 5cm, consider same cube
                    already_queued = True
                    break
            if already_queued:
                continue

            # Choose drop location by color
            if labeled_cube.color_label == 'red':
                drop_location = self.red_drop_location
            elif labeled_cube.color_label == 'black':
                drop_location = self.blue_drop_location
            else:
                continue

            self.cube_queue.append((cube_pose_base, labeled_cube.color_label, drop_location))

            self.get_logger().info(
                f"Queued {labeled_cube.color_label} cube at "
                f"({cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}) "
                f"→ drop at ({drop_location.point.x:.3f}, "
                f"{drop_location.point.y:.3f}, {drop_location.point.z:.3f})"
            )

        # Start processing if not currently busy
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

    # --------------------------------------------------------------------------
    # Cube processing pipeline
    # --------------------------------------------------------------------------

    def _process_next_cube(self):
        """Process the next cube from the queue."""
        if not self.cube_queue or self.processing_cube:
            return

        if self.joint_state is None:
            self.get_logger().warn("No joint state available, cannot process cube")
            return

        cube_pose, color, drop_location = self.cube_queue.pop(0)
        self.processing_cube = True
        self.current_cube_pose = cube_pose
        self.current_cube_color = color

        self.get_logger().info(
            f"Processing {color} cube at "
            f"({cube_pose.point.x:.3f}, {cube_pose.point.y:.3f}, {cube_pose.point.z:.3f})"
        )

        if not self._build_cube_job_queue(cube_pose, drop_location):
            # Failed to build job queue; reset state and move on to next cube if any
            self.get_logger().warn(
                "Failed to build job queue for cube; skipping to next cube if available."
            )
            self.processing_cube = False
            self.current_cube_pose = None
            self.current_cube_color = None
            if self.cube_queue:
                self._process_next_cube()
            return

        self.execute_jobs()

    def _build_cube_job_queue(self, cube_pose: PointStamped,
                              drop_location: PointStamped) -> bool:
        """
        Build job queue:
        pre-grasp → grasp → grip → smooth arc to drop → final drop → release → home.
        """
        if cube_pose.header.frame_id != "base_link":
            self.get_logger().error(
                f"Cube pose is not in base_link frame: {cube_pose.header.frame_id}. Aborting."
            )
            self.job_queue = []
            return False

        if drop_location.header.frame_id != "base_link":
            self.get_logger().error(
                f"Drop location is not in base_link frame: {drop_location.header.frame_id}. Aborting."
            )
            self.job_queue = []
            return False

        self.job_queue = []

        cx = cube_pose.point.x
        cy = cube_pose.point.y
        cz = cube_pose.point.z

        # Pre-grasp pose
        pre_x = cx + 0.005
        pre_y = cy - 0.030
        # pre_z = cz + 0.183
        pre_z = cz + 0.3
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error("IK failed for pre-grasp pose.")
            return False
        self.job_queue.append(pre_grasp_js)
        self.get_logger().info(f"going to position {ur7e_forward_kinematics_from_angles(pre_grasp_js.position)[:3,3]}")

        # Grasp pose
        grasp_x = cx + 0.005
        grasp_y = cy - 0.03
        # grasp_z = cz + 0.14
        grasp_z = cz + 0.25
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error("IK failed for grasp pose.")
            self.job_queue = []
            return False
        self.job_queue.append(grasp_js)
        self.get_logger().info(f"going to position {ur7e_forward_kinematics_from_angles(grasp_js.position)[:3,3]}")

        # Close gripper
        self.job_queue.append('toggle_grip')
        self.job_queue.append(pre_grasp_js)

        # MPC to drop location
        drop_x = drop_location.point.x
        drop_y = drop_location.point.y
        drop_z = drop_location.point.z
        drop_js = self.ik_planner.compute_ik(self.joint_state, drop_x, drop_y, drop_z)
        self.job_queue.append(drop_js)
        self.get_logger().info(f"going to position {ur7e_forward_kinematics_from_angles(drop_js.position)[:3,3]}")

        move_up_z = max(self.clearance_height, pre_z + 0.15)

        self.get_logger().info(
            "Added goal pose to queue"
        )

        # Release the gripper at drop location (open)
        self.job_queue.append('toggle_grip')
        self.get_logger().info("Added gripper release command")

        # move gripper up
        drop_z_up = drop_z + 0.1
        drop_js_up = self.ik_planner.compute_ik(self.joint_state, drop_x, drop_y, drop_z_up)
        self.job_queue.append(drop_js_up)
        self.get_logger().info(f"going to position {ur7e_forward_kinematics_from_angles(drop_js_up.position)[:3,3]}")

        # Return to home if available
        if self.initial_joint_state is not None:
            self.job_queue.append(('return_home', self.initial_joint_state))
            self.get_logger().info("Added return to home position after cube processing")
        else:
            self.get_logger().warn(
                "Initial joint state not available, skipping return to home"
            )

        return True

    # --------------------------------------------------------------------------
    # Job execution
    # --------------------------------------------------------------------------

    def execute_jobs(self):
        """Execute jobs from the queue sequentially."""
        if not self.job_queue:
            self.get_logger().info("All jobs completed for current cube.")
            # Mark current cube as completed so we don't process it again
            if self.current_cube_pose is not None:
                px = self.current_cube_pose.point.x
                py = self.current_cube_pose.point.y
                pz = self.current_cube_pose.point.z
                self.completed_cubes.append((px, py, pz))
                self.get_logger().info(
                    f"Marked {self.current_cube_color} cube at "
                    f"({px:.3f}, {py:.3f}, {pz:.3f}) as completed."
                )
                self.current_cube_pose = None
                self.current_cube_color = None

            self.processing_cube = False

            if self.cube_queue:
                self.get_logger().info(
                    f"{len(self.cube_queue)} cube(s) remaining in queue. "
                    f"Processing next cube..."
                )
                self._process_next_cube()
            else:
                self.get_logger().info(
                    "No more cubes in queue. Waiting for new detections..."
                )
            return

        self.get_logger().info(
            f"Executing job queue, {len(self.job_queue)} jobs remaining."
        )
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, tuple) and len(next_job) == 2 and next_job[0] == 'return_home':
            home_joint_state = next_job[1]
            if self.joint_state is None:
                self.get_logger().error("No current joint state; cannot plan trajectory.")
                self.processing_cube = False
                self.job_queue = []
                return

            traj = self.ik_planner.plan_to_joints(home_joint_state)
            if traj is None:
                self.get_logger().error(
                    "Failed to plan to home position using MoveIt"
                )
                self.processing_cube = False
                self.job_queue = []
                return
            self.get_logger().info("MoveIt planned trajectory to home position")
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif isinstance(next_job, JointState):
            # use MPC to move
            if self.joint_state is None:
                self.get_logger().error("No current joint state; cannot plan trajectory.")
                self.processing_cube = False
                self.job_queue = []
                return

            traj = self._plan_with_mpc(self.joint_state, next_job)
            j_points = traj.points 
            mpc_destination_j = j_points[-1].positions
            mpc_destination = ur7e_forward_kinematics_from_angles(mpc_destination_j)[:3,3]
            final_destination = ur7e_forward_kinematics_from_angles(next_job.position)[:3,3]

            self.get_logger().info(f"MPC destination: {mpc_destination}, Final destination: {final_destination}")

            if traj is None:
                self.get_logger().error("Failed to plan to position using MoveIt")
                self.processing_cube = False
                self.job_queue = []
                return
            self.get_logger().info("MPC planned trajectory")
            self._execute_joint_trajectory(traj)

        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()

        else:
            self.get_logger().error(f"Unknown job type: {type(next_job)}")
            self.processing_cube = False
            self.job_queue = []

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
        if self.current_obstacles is not None:
            for center, half_size in self.current_obstacles:
                self.mpc.add_obstacle(center, half_size)
        
        # Solve MPC
        q_next, q_traj = self.mpc.compute_control(current_state, q_target)

        CONTROLLER_JOINT_ORDER = [
            'shoulder_lift_joint', 
            'elbow_joint', 
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint', 
            'shoulder_pan_joint'
        ]
        MPC_INTERNAL_NAMES = list(target_js.name)

        # Create a map from controller name -> its index in the MPC's array
        mpc_index_map = [MPC_INTERNAL_NAMES.index(name) for name in CONTROLLER_JOINT_ORDER]
        
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

            # pt.velocities = [0.0] * len(q_traj[k])
            
            # Time from start based on MPC dt
            t = float(k) * self.mpc.dt
            secs = int(t)
            nsecs = int((t - secs) * 1e9)
            pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
            
            jt.points.append(pt)
        
        # Publish trajectory for visualization in RViz
        self._publish_mpc_trajectory(current_js, jt)

        # reorder joints
        exec_jt = JointTrajectory()
        exec_jt.joint_names = CONTROLLER_JOINT_ORDER
        exec_jt.header = jt.header
        
        for vis_pt in jt.points:
            reordered_pt = JointTrajectoryPoint()
            reordered_ps = [vis_pt.positions[i] for i in mpc_index_map]
            reordered_vs = [vis_pt.velocities[i] for i in mpc_index_map]

            reordered_pt.positions = reordered_ps
            reordered_pt.velocities = reordered_vs

            reordered_pt.time_from_start = vis_pt.time_from_start
            exec_jt.points.append(reordered_pt)
        
        return exec_jt

    def _publish_mpc_trajectory(self, current_js: JointState, joint_traj: JointTrajectory):

        trajectory = Marker()
        trajectory.header.frame_id = 'base_link'

        for point in joint_traj.points:

            try:
                positions = point.positions
                ee_xyz = ur7e_forward_kinematics_from_angles(positions)[:3,3]
                trajectory.ns = "red_points"
                trajectory.type = Marker.POINTS
                trajectory.scale.x = 0.01
                trajectory.scale.y = 0.01
                trajectory.color.r = 0.0
                trajectory.color.g = 1.0
                trajectory.color.b = 1.0
                trajectory.color.a = 0.8
                point = Point(x=ee_xyz[0], y=ee_xyz[1], z=ee_xyz[2])
                trajectory.points.append(point)

            except Exception as e:
                self.get_logger().debug("failed to add point")

        self.mpc_traj_pub.publish(trajectory)
        self.get_logger().debug("Published MPC trajectory to /mpc_trajectory for RViz visualization")

    # --------------------------------------------------------------------------
    # Gripper control (async, no nested spin)
    # --------------------------------------------------------------------------

    def _toggle_gripper(self):
        """Toggle gripper open/closed asynchronously."""
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            self.processing_cube = False
            self.job_queue = []
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        future.add_done_callback(self._on_gripper_done)

    def _on_gripper_done(self, future):
        """Callback once the gripper toggle service completes."""
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error(
                    f'Gripper service failed: {response.message}'
                )
                self.processing_cube = False
                self.job_queue = []
                return
            self.get_logger().info('Gripper toggled.')
        except Exception as e:
            self.get_logger().error(f'Gripper service call failed: {e}')
            self.processing_cube = False
            self.job_queue = []
            return

        # Proceed to next job after gripper is done
        self.execute_jobs()

    # --------------------------------------------------------------------------
    # Trajectory execution via FollowJointTrajectory action
    # --------------------------------------------------------------------------

    def _execute_joint_trajectory(self, joint_traj: JointTrajectory):
        """Execute a joint trajectory."""
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        """Handle goal sent callback."""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Trajectory goal was not accepted by controller')
                self.processing_cube = False
                self.job_queue = []
                return

            self.get_logger().info('Executing trajectory...')
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_exec_done)
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory goal: {e}')
            self.processing_cube = False
            self.job_queue = []

    def _on_exec_done(self, future):
        """Handle execution done callback."""
        try:
            _ = future.result().result
            self.get_logger().info('Execution complete.')
            # Proceed to next job
            self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            self.processing_cube = False
            self.job_queue = []


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()