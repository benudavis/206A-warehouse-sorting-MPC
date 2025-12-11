from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory, DisplayTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from custom_msgs.msg import LabeledCubeArray, LabeledCube, BoxBounds
import numpy as np
from planning.ik import IKPlanner
from planning.mpc_controller import MPCController
from planning.forward_kinematics import ur7e_forward_kinematics_from_angles

class UR7e_CubeGrasp(Node):

    def __init__(self):

        super().__init__('cube_grasp')

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

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.red_drop_location = [0.5, 0.2, 0.15]
        self.blue_drop_location = [0.5, -0.2, 0.15]

        self.joint_state = None
        self.initial_joint_state = None  # Store initial/home position
        self.cube_queue = []
        self.processing_cube = False
        self.processed_cube_ids = set()
        self.current_obstacles = []
        self.pick_height = None
        self.gripper_closed = False

        self.ik_planner = IKPlanner()
        self.mpc = MPCController(n_joints=6, horizon=6, dt=0.15)
        
        safety_margin = 0.10
        self.mpc.set_safety_margin(safety_margin)
        self.get_logger().info(f"MPC configured with safety_margin={safety_margin:.3f}m (hard constraint)")

        self.mpc_traj_pub = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            1
        )

        self.job_queue = []

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg
        
        # Store initial joint state on first callback (home position)
        if self.initial_joint_state is None:
            # Create a copy of the joint state to store as home position
            self.initial_joint_state = JointState()
            self.initial_joint_state.header = msg.header
            self.initial_joint_state.name = list(msg.name)
            self.initial_joint_state.position = list(msg.position)
            self.initial_joint_state.velocity = list(msg.velocity) if msg.velocity else []
            self.initial_joint_state.effort = list(msg.effort) if msg.effort else []
            self.get_logger().info("Stored initial/home joint state")

    def labeled_cubes_callback(self, msg: LabeledCubeArray):
        """Process detected cubes and queue them for picking."""
        if self.joint_state is None:
            self.get_logger().debug("No joint state yet, skipping cube processing")
            return

        if self.processing_cube:
            self.get_logger().debug("Already processing a cube, skipping new detections")
            return

        for i, labeled_cube in enumerate(msg.cubes):
            if labeled_cube.color_label not in ['red', 'blue']:
                continue

            cube_id = f"{labeled_cube.point.point.x:.4f}_{labeled_cube.point.point.y:.4f}_{labeled_cube.point.point.z:.4f}"
            
            if cube_id in self.processed_cube_ids:
                continue

            cube_pose_base = labeled_cube.point

            if labeled_cube.color_label == 'red':
                drop_location = self.red_drop_location
            elif labeled_cube.color_label == 'blue':
                drop_location = self.blue_drop_location
            else:
                continue

            self.cube_queue.append((cube_pose_base, labeled_cube.color_label, drop_location))
            self.processed_cube_ids.add(cube_id)
            
            self.get_logger().info(
                f"Queued {labeled_cube.color_label} cube at "
                f"({cube_pose_base.point.x:.3f}, {cube_pose_base.point.y:.3f}, {cube_pose_base.point.z:.3f}) "
                f"for drop at ({drop_location[0]:.3f}, {drop_location[1]:.3f}, {drop_location[2]:.3f})"
            )

        if self.cube_queue and not self.processing_cube:
            self._process_next_cube()

    def obstacles_callback(self, msg: BoxBounds):
        """Process obstacle bounds and convert to (center, half_size) format for MPC."""
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
        
        self.current_obstacles = [(center, half_size)]
        
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
        """Process the next cube from the queue."""
        if not self.cube_queue or self.processing_cube:
            return

        if self.joint_state is None:
            self.get_logger().warn("No joint state available, cannot process cube")
            return

        cube_pose, color, drop_location = self.cube_queue.pop(0)
        self.processing_cube = True
        self.gripper_closed = False
        self.pick_height = None

        self.get_logger().info(
            f"Processing {color} cube at "
            f"({cube_pose.point.x:.3f}, {cube_pose.point.y:.3f}, {cube_pose.point.z:.3f})"
        )

        self._build_cube_job_queue(cube_pose, drop_location)
        self.execute_jobs()

    def _build_cube_job_queue(self, cube_pose: PointStamped, drop_location: list):
        """
        Build the job queue for picking up a cube and placing it at drop_location.
        
        Args:
            cube_pose: PointStamped in base_link frame with cube position
            drop_location: [x, y, z] target drop location in base_link frame
        """
        self.job_queue = []

        cx = cube_pose.point.x
        cy = cube_pose.point.y
        cz = cube_pose.point.z

        pre_x = cx + 0.0
        pre_y = cy - 0.035
        pre_z = cz + 0.185
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error("IK failed for pre-grasp pose. Skipping this cube.")
            self.processing_cube = False
            # Try to process next cube
            if self.cube_queue:
                self._process_next_cube()
            return
        self.job_queue.append((pre_grasp_js, False))

        grasp_x = cx + 0.0
        grasp_y = cy - 0.035
        grasp_z = cz + 0.16
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error("IK failed for grasp pose. Skipping this cube.")
            self.processing_cube = False
            # Try to process next cube
            if self.cube_queue:
                self._process_next_cube()
            return
        self.job_queue.append((grasp_js, False))

        self.job_queue.append('toggle_grip')
        self.job_queue.append((pre_grasp_js, False))

        rel_x = drop_location[0]
        rel_y = drop_location[1]
        rel_z = drop_location[2]
        release_js = self.ik_planner.compute_ik(self.joint_state, rel_x, rel_y, rel_z)
        if release_js is None:
            self.get_logger().error("IK failed for release pose. Skipping this cube.")
            self.processing_cube = False
            # Try to process next cube
            if self.cube_queue:
                self._process_next_cube()
            return
        self.job_queue.append((release_js, True))

        self.job_queue.append('toggle_grip')
        
        # After releasing, return to initial/home position before processing next cube
        if self.initial_joint_state is not None:
            self.job_queue.append((self.initial_joint_state, False))
            self.get_logger().info("Added return to home position after release")
        else:
            self.get_logger().warn("Initial joint state not available, skipping return to home")

    def execute_jobs(self):

        if not self.job_queue:
            self.get_logger().info("All jobs completed for current cube.")
            self.processing_cube = False
            if self.cube_queue:
                self._process_next_cube()
            else:
                self.get_logger().info("No more cubes in queue. Waiting for new detections...")
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, tuple) and len(next_job) == 2:
            target_js, use_mpc = next_job
            
            if self.joint_state is None:
                self.get_logger().error("No current joint state; cannot plan trajectory.")
                return
            
            if use_mpc:
                traj = self._plan_with_mpc(self.joint_state, target_js)
                if traj is None:
                    self.get_logger().error("MPC failed to plan trajectory. Attempting recovery...")
                    # Error recovery: try to continue with next job or return to home
                    self._handle_mpc_failure()
                    return
                self.get_logger().info("MPC planned trajectory")
                self._execute_joint_trajectory(traj)
            else:
                traj = self.ik_planner.plan_to_joints(target_js)
                if traj is None:
                    self.get_logger().error("Failed to plan to position using MoveIt. Attempting recovery...")
                    # Error recovery: try to continue with next job or return to home
                    self._handle_moveit_failure()
                    return
                self.get_logger().info("MoveIt planned trajectory")
                self._execute_joint_trajectory(traj.joint_trajectory)
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()

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

    def _plan_with_mpc(self, current_js: JointState, target_js: JointState) -> JointTrajectory:
        """
        Plan a trajectory from current joint state to target using MPC.
        
        Args:
            current_js: Current joint state
            target_js: Target joint state
            
        Returns:
            JointTrajectory message ready to execute
        """
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
        
        try:
            dq_current = np.array(
                [current_js.velocity[name_to_index[name]] for name in target_js.name],
                dtype=float,
            )
        except (KeyError, IndexError):
            self.get_logger().warn("Velocities not available in joint state, using zero")
            dq_current = np.zeros_like(q_current)
        
        current_state = np.concatenate([q_current, dq_current])
        
        self.mpc.clear_obstacles()
        for center, half_size in self.current_obstacles:
            self.mpc.add_obstacle(center, half_size)
        
        q_next, q_traj = self.mpc.compute_control(current_state, q_target)
        
        jt = JointTrajectory()
        jt.joint_names = list(target_js.name)
        jt.header.stamp = self.get_clock().now().to_msg()
        jt.header.frame_id = "base_link"
        
        from builtin_interfaces.msg import Duration
        for k in range(q_traj.shape[0]):
            pt = JointTrajectoryPoint()
            pt.positions = q_traj[k].tolist()
            
            if k < q_traj.shape[0] - 1:
                velocities = (q_traj[k + 1] - q_traj[k]) / self.mpc.dt
                pt.velocities = velocities.tolist()
            else:
                if k > 0:
                    velocities = (q_traj[k] - q_traj[k - 1]) / self.mpc.dt
                    pt.velocities = velocities.tolist()
                else:
                    pt.velocities = [0.0] * len(q_traj[k])
            
            t = float(k) * self.mpc.dt
            secs = int(t)
            nsecs = int((t - secs) * 1e9)
            pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
            
            jt.points.append(pt)
        
        self._publish_mpc_trajectory(current_js, jt)
        
        return jt

    def _publish_mpc_trajectory(self, current_js: JointState, joint_traj: JointTrajectory):
        """
        Publish the MPC trajectory as DisplayTrajectory for visualization in RViz.
        
        RViz setup:
          - Add "MotionPlanning" display
          - In "Planned Path" tab, set topic to /display_planned_path
        """
        robot_state = RobotState()
        robot_state.joint_state = current_js

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj

        display_msg = DisplayTrajectory()
        display_msg.model_id = "ur"
        display_msg.trajectory_start = robot_state
        display_msg.trajectory.append(robot_traj)

        self.mpc_traj_pub.publish(display_msg)
        self.get_logger().debug("Published MPC trajectory to /display_planned_path for RViz visualization")

    def _toggle_gripper(self):
        """
        Toggle gripper and manage minimum_z constraint.
        When closing: set minimum_z = pick_height - 0.01m (1cm margin below pick height)
        When opening: clear minimum_z constraint
        """
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.gripper_closed = not self.gripper_closed
        
        if self.gripper_closed:
            if self.joint_state is not None:
                pick_height = self._compute_end_effector_z(self.joint_state)
                if pick_height is not None:
                    self.pick_height = pick_height
                    minimum_z = pick_height - 0.01
                    self.mpc.set_minimum_z(minimum_z)
                    self.get_logger().info(
                        f"Gripper closed. Set minimum_z={minimum_z:.3f}m "
                        f"(pick_height={pick_height:.3f}m - 1cm margin)"
                    )
                else:
                    self.get_logger().warn("Could not compute pick height, minimum_z not set")
            else:
                self.get_logger().warn("No joint state available, minimum_z not set")
        else:
            self.mpc.clear_minimum_z()
            self.pick_height = None
            self.get_logger().info("Gripper opened. Cleared minimum_z constraint")

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()
    
    def _handle_mpc_failure(self):
        """Handle MPC planning failure with recovery strategy."""
        self.get_logger().warn("MPC planning failed. Clearing job queue and attempting recovery.")
        
        # Clear remaining jobs for this cube
        self.job_queue = []
        
        # Try to return to home position if available
        if self.initial_joint_state is not None:
            self.get_logger().info("Attempting to return to home position for recovery")
            home_traj = self.ik_planner.plan_to_joints(self.initial_joint_state)
            if home_traj is not None:
                self._execute_joint_trajectory(home_traj.joint_trajectory)
            else:
                self.get_logger().error("Failed to plan return to home. Manual intervention may be required.")
        else:
            self.get_logger().error("Initial joint state not available for recovery")
        
        # Mark cube processing as failed
        self.processing_cube = False
        
        # Try to process next cube if available
        if self.cube_queue:
            self.get_logger().info("Attempting to process next cube after recovery")
            self._process_next_cube()
        else:
            self.get_logger().info("No more cubes in queue. Waiting for new detections...")
    
    def _handle_moveit_failure(self):
        """Handle MoveIt planning failure with recovery strategy."""
        self.get_logger().warn("MoveIt planning failed. Attempting recovery.")
        
        # For MoveIt failures, we can try to continue or return to home
        # Clear the failed job
        if self.job_queue:
            self.get_logger().info("Skipping failed job, continuing with next job")
            # Continue with next job
            self.execute_jobs()
        else:
            # No more jobs, try to return to home
            if self.initial_joint_state is not None:
                self.get_logger().info("No more jobs, attempting to return to home position")
                home_traj = self.ik_planner.plan_to_joints(self.initial_joint_state)
                if home_traj is not None:
                    self._execute_joint_trajectory(home_traj.joint_trajectory)
                else:
                    self.get_logger().error("Failed to plan return to home")
            
            self.processing_cube = False
            if self.cube_queue:
                self._process_next_cube()

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
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Trajectory goal was not accepted by controller')
                self._handle_execution_failure()
                return

            self.get_logger().info('Executing...')
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_exec_done)
        except Exception as e:
            self.get_logger().error(f'Error sending trajectory goal: {e}')
            self._handle_execution_failure()

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            error_code = result.error_code if hasattr(result, 'error_code') else None
            
            if error_code is not None and error_code != 0:
                self.get_logger().error(f'Trajectory execution failed with error code: {error_code}')
                self._handle_execution_failure()
            else:
                self.get_logger().info('Execution complete.')
                self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            self._handle_execution_failure()
    
    def _handle_execution_failure(self):
        """Handle trajectory execution failure with recovery strategy."""
        self.get_logger().warn("Trajectory execution failed. Attempting recovery.")
        
        # Clear remaining jobs for this cube
        self.job_queue = []
        
        # Try to return to home position if available
        if self.initial_joint_state is not None:
            self.get_logger().info("Attempting to return to home position after execution failure")
            home_traj = self.ik_planner.plan_to_joints(self.initial_joint_state)
            if home_traj is not None:
                self._execute_joint_trajectory(home_traj.joint_trajectory)
            else:
                self.get_logger().error("Failed to plan return to home. Manual intervention may be required.")
        else:
            self.get_logger().error("Initial joint state not available for recovery")
        
        # Mark cube processing as failed
        self.processing_cube = False
        
        # Try to process next cube if available
        if self.cube_queue:
            self.get_logger().info("Attempting to process next cube after execution failure recovery")
            self._process_next_cube()
        else:
            self.get_logger().info("No more cubes in queue. Waiting for new detections...")

def main(args=None):

    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
