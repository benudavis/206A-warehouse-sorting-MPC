from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from custom_msgs.msg import LabeledCubeArray, LabeledCube, BoxBounds
from planning.ik import IKPlanner
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
import numpy as np


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
        """Update obstacle clearance height for trajectory planning."""
        self.obstacle_top_z = msg.z_max
        self.clearance_height = self.obstacle_top_z + self.clearance_margin

        self.get_logger().info(
            f"Obstacle detected: top_z={self.obstacle_top_z:.3f}m, "
            f"clearance_height={self.clearance_height:.3f}m"
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
        pre_z = cz + 0.183
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error("IK failed for pre-grasp pose.")
            return False
        self.job_queue.append(pre_grasp_js)

        # Grasp pose
        grasp_x = cx + 0.005
        grasp_y = cy - 0.03
        grasp_z = cz + 0.14
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error("IK failed for grasp pose.")
            self.job_queue = []
            return False
        self.job_queue.append(grasp_js)

        # Close gripper
        self.job_queue.append('toggle_grip')

        # Smooth arc waypoints to drop location
        drop_x = drop_location.point.x
        drop_y = drop_location.point.y
        drop_z = drop_location.point.z
        move_up_z = max(self.clearance_height, pre_z + 0.15)

        smooth_waypoints = self._generate_smooth_waypoints(
            start_pos=[pre_x, pre_y, pre_z],
            end_pos=[drop_x, drop_y, drop_z],
            clearance_z=move_up_z,
            num_waypoints=self.smooth_waypoints_count
        )

        for i, (wx, wy, wz) in enumerate(smooth_waypoints):
            waypoint_js = self.ik_planner.compute_ik(self.joint_state, wx, wy, wz)
            if waypoint_js is None:
                self.get_logger().error(
                    f"IK failed for smooth waypoint {i + 1}/{len(smooth_waypoints)} "
                    f"at ({wx:.3f}, {wy:.3f}, {wz:.3f})"
                )
                self.job_queue = []
                return False
            self.job_queue.append(waypoint_js)

        self.get_logger().info(
            f"Generated {len(smooth_waypoints)} smooth waypoints from "
            f"({pre_x:.3f}, {pre_y:.3f}, {pre_z:.3f}) to "
            f"({drop_x:.3f}, {drop_y:.3f}, {drop_z:.3f}) via clearance z={move_up_z:.3f}m"
        )

        # Ensure we're at the exact drop location before releasing
        drop_js = self.ik_planner.compute_ik(self.joint_state, drop_x, drop_y, drop_z)
        if drop_js is None:
            self.get_logger().error("IK failed for final drop pose.")
            self.job_queue = []
            return False
        self.job_queue.append(drop_js)
        self.get_logger().info(
            f"Added final drop position at ({drop_x:.3f}, {drop_y:.3f}, {drop_z:.3f})"
        )

        # Release the gripper at drop location (open)
        self.job_queue.append('toggle_grip')
        self.get_logger().info("Added gripper release command")

        # Return to home if available
        if self.initial_joint_state is not None:
            self.job_queue.append(('return_home', self.initial_joint_state))
            self.get_logger().info("Added return to home position after cube processing")
        else:
            self.get_logger().warn(
                "Initial joint state not available, skipping return to home"
            )

        return True

    def _generate_smooth_waypoints(self, start_pos, end_pos,
                                   clearance_z, num_waypoints=8):
        """
        Generate smooth waypoints along a curved arc:
        up to clearance, then down to end.
        """
        start = np.array(start_pos)
        end = np.array(end_pos)
        waypoints = []
        t_values = np.linspace(0.0, 1.0, num_waypoints + 2)

        for t in t_values:
            t_smooth = t * t * (3.0 - 2.0 * t)
            xy = (1.0 - t_smooth) * start[:2] + t_smooth * end[:2]

            if t_smooth < 0.5:
                z_alpha = 2.0 * t_smooth
                z = start[2] + (clearance_z - start[2]) * (z_alpha * z_alpha)
            else:
                z_alpha = 2.0 * (1.0 - t_smooth)
                z = end[2] + (clearance_z - end[2]) * (z_alpha * z_alpha)

            waypoints.append([xy[0], xy[1], z])

        return waypoints

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
            if self.joint_state is None:
                self.get_logger().error("No current joint state; cannot plan trajectory.")
                self.processing_cube = False
                self.job_queue = []
                return

            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("Failed to plan to position using MoveIt")
                self.processing_cube = False
                self.job_queue = []
                return
            self.get_logger().info("MoveIt planned trajectory")
            self._execute_joint_trajectory(traj.joint_trajectory)

        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()

        else:
            self.get_logger().error(f"Unknown job type: {type(next_job)}")
            self.processing_cube = False
            self.job_queue = []

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