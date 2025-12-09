#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_srvs.srv import Trigger
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

import numpy as np

from moveit_msgs.msg import (
    DisplayTrajectory,
    RobotState,
    RobotTrajectory,
    PlanningScene,
    CollisionObject,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

from planning.ik import IKPlanner
from planning.mpc_controller import MPCController


class UR7e_MPC_CubeGrasp(Node):
    """
    Pick-and-place controller using IK + MPC-generated joint trajectories
    with EE-level obstacle avoidance, and visualization in MoveIt/RViz.

    Pipeline:
        /cube_pose_in_base -> cube_callback ->
            build pre-grasp, grasp, lift, release joint targets via IK ->
            for each target:
                - MPC plan in joint space (with obstacle boxes)
                - publish DisplayTrajectory to /display_planned_path (for RViz)
                - send JointTrajectory to /scaled_joint_trajectory_controller.
    """

    def __init__(self):
        super().__init__('mpc_cube_grasp')

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter('obstacle_centers', [0.30, 0.00, 0.15])  # flattened list of xyz
        self.declare_parameter('obstacle_sizes', [0.05, 0.05, 0.15])    # flattened list of half-extents
        self.declare_parameter('mpc_horizon', 30)
        self.declare_parameter('mpc_dt', 0.08)
        self.declare_parameter('mpc_safety_margin', 0.12)
        self.declare_parameter('mpc_Q', 500.0)
        self.declare_parameter('mpc_QT', 1000.0)
        self.declare_parameter('mpc_R', 0.1)
        self.declare_parameter('basket_centers', [0.55, -0.20, 0.10, 0.55, 0.20, 0.10])
        self.declare_parameter('basket_sizes', [0.08, 0.08, 0.05, 0.08, 0.08, 0.05])

        # --------------------------------------------------
        # Subscriptions
        # --------------------------------------------------
        self.cube_sub = self.create_subscription(
            PointStamped,
            '/cube_pose_in_base',
            self.cube_callback,
            1
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            1
        )

        # --------------------------------------------------
        # Action client for UR controller
        # --------------------------------------------------
        self.exec_ac = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # --------------------------------------------------
        # Gripper service client
        # --------------------------------------------------
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        # --------------------------------------------------
        # MoveIt IK planner (services /compute_ik, /plan_kinematic_path)
        # --------------------------------------------------
        self.ik_planner = IKPlanner()

        # --------------------------------------------------
        # MPC controller
        # --------------------------------------------------
        self.mpc = MPCController(
            n_joints=6,
            horizon=int(self.get_parameter('mpc_horizon').value),
            dt=float(self.get_parameter('mpc_dt').value),
            enable_fk=True,
        )
        self.mpc.safety_margin = float(self.get_parameter('mpc_safety_margin').value)
        self.mpc.set_cost_weights(
            float(self.get_parameter('mpc_Q').value),
            float(self.get_parameter('mpc_QT').value),
            float(self.get_parameter('mpc_R').value),
        )

        # --------------------------------------------------
        # Obstacle configuration (in base_link frame)
        # --------------------------------------------------
        self._setup_default_obstacles_from_params()

        # --------------------------------------------------
        # DisplayTrajectory publisher for MoveIt/RViz
        # --------------------------------------------------
        # RViz MotionPlanning plugin listens on /display_planned_path by default.
        self.mpc_traj_pub = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10
        )
        # Extra publisher for MoveIt default display topic (some RViz configs expect this)
        self.mpc_traj_pub_move_group = self.create_publisher(
            DisplayTrajectory,
            '/move_group/display_planned_path',
            10
        )
        # Planning scene publisher so obstacles/baskets are visible in MoveIt/RViz
        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            5
        )

        self.cube_pose = None
        self.joint_state = None

        # Queue: entries are JointState or 'toggle_grip'
        self.job_queue = []

        # Publish static scene objects once
        self._publish_scene_objects()

        self.get_logger().info("UR7e_MPC_CubeGrasp node initialised.")

    # ------------------------------------------------------------------
    # Environment / obstacle configuration
    # ------------------------------------------------------------------
    def _setup_default_obstacles_from_params(self):
        """Configure obstacle boxes in base_link frame using ROS parameters."""
        self.mpc.clear_obstacles()

        centers = self.get_parameter('obstacle_centers').value
        sizes = self.get_parameter('obstacle_sizes').value

        if len(centers) % 3 != 0 or len(sizes) % 3 != 0 or len(centers) != len(sizes):
            self.get_logger().warn(
                "Obstacle parameter lengths must match and be multiples of 3; "
                "skipping obstacle loading."
            )
            return

        n_obs = len(centers) // 3
        for i in range(n_obs):
            cx, cy, cz = centers[3 * i: 3 * i + 3]
            sx, sy, sz = sizes[3 * i: 3 * i + 3]
            self.mpc.add_obstacle([cx, cy, cz], [sx, sy, sz])

        self.get_logger().info(f"Loaded {n_obs} obstacles into MPC.")

    def _publish_scene_objects(self):
        """Publish obstacles and baskets into MoveIt planning scene for visualization."""
        if self.scene_pub is None:
            return

        ps = PlanningScene()
        ps.is_diff = True

        def build_collision_objects(prefix, centers, sizes):
            objs = []
            n = len(centers) // 3
            for i in range(n):
                cx, cy, cz = centers[3 * i: 3 * i + 3]
                sx, sy, sz = sizes[3 * i: 3 * i + 3]

                co = CollisionObject()
                co.id = f"{prefix}_{i}"
                co.header.frame_id = 'base_link'
                box = SolidPrimitive()
                box.type = SolidPrimitive.BOX
                box.dimensions = [2 * float(sx), 2 * float(sy), 2 * float(sz)]  # full lengths
                co.primitives.append(box)

                pose = Pose()
                pose.position.x = float(cx)
                pose.position.y = float(cy)
                pose.position.z = float(cz)
                pose.orientation.w = 1.0
                co.primitive_poses.append(pose)
                co.operation = CollisionObject.ADD
                objs.append(co)
            return objs

        centers = self.get_parameter('obstacle_centers').value
        sizes = self.get_parameter('obstacle_sizes').value
        baskets_c = self.get_parameter('basket_centers').value
        baskets_s = self.get_parameter('basket_sizes').value

        if len(centers) == len(sizes) and len(centers) % 3 == 0:
            ps.world.collision_objects.extend(
                build_collision_objects("obstacle", centers, sizes)
            )
        else:
            self.get_logger().warn("Obstacle params invalid; skipping planning scene obstacles.")

        if len(baskets_c) == len(baskets_s) and len(baskets_c) % 3 == 0:
            ps.world.collision_objects.extend(
                build_collision_objects("basket", baskets_c, baskets_s)
            )
        else:
            self.get_logger().warn("Basket params invalid; skipping basket visualization.")

        for _ in range(3):  # small burst to ensure RViz/MoveIt latch
            self.scene_pub.publish(ps)

    # ------------------------------------------------------------------
    # ROS Callbacks
    # ------------------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def cube_callback(self, cube_pose: PointStamped):
        # Only process the first detected cube
        if self.cube_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed.")
            return

        self.cube_pose = cube_pose

        self.get_logger().info(
            f"Received cube pose in base_link: "
            f"({cube_pose.point.x:.3f}, "
            f"{cube_pose.point.y:.3f}, "
            f"{cube_pose.point.z:.3f})"
        )

        # -----------------------------
        # Build IK targets (same offsets as your original code)
        # -----------------------------
        cx = cube_pose.point.x
        cy = cube_pose.point.y
        cz = cube_pose.point.z

        ik_link = 'tool0'

        self.get_logger().info(
            f"Building IK targets (ik_link={ik_link}) for cube at "
            f"({cx:.3f}, {cy:.3f}, {cz:.3f})."
        )

        # 1) Pre-grasp
        pre_x = cx + 0.0
        pre_y = cy - 0.035
        pre_z = cz + 0.185
        self.get_logger().info(
            f"IK target [pre-grasp]: ({pre_x:.3f}, {pre_y:.3f}, {pre_z:.3f}), link={ik_link}"
        )
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error(
                f"IK failed for pre-grasp pose at ({pre_x:.3f}, {pre_y:.3f}, {pre_z:.3f}), "
                f"link={ik_link}, error_code={self.ik_planner.last_ik_error_code}"
            )
            return
        self.job_queue.append(pre_grasp_js)

        # 2) Grasp (lower z, not below +0.16)
        grasp_x = cx + 0.0
        grasp_y = cy - 0.035
        grasp_z = cz + 0.16
        self.get_logger().info(
            f"IK target [grasp]: ({grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f}), link={ik_link}"
        )
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error(
                f"IK failed for grasp pose at ({grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f}), "
                f"link={ik_link}, error_code={self.ik_planner.last_ik_error_code}"
            )
            return
        self.job_queue.append(grasp_js)

        # 3) Close gripper
        self.job_queue.append('toggle_grip')

        # 4) Lift back to pre-grasp
        self.job_queue.append(pre_grasp_js)

        # 5) Release pose: choose nearest basket in (x, y), hover above by +0.15 m
        rel_x, rel_y, rel_z, basket_idx = self._choose_basket_release(cx, cy, cz)
        self.get_logger().info(
            f"IK target [release->basket {basket_idx if basket_idx is not None else 'fallback'}]: "
            f"({rel_x:.3f}, {rel_y:.3f}, {rel_z:.3f}), link={ik_link}"
        )

        release_js = self.ik_planner.compute_ik(self.joint_state, rel_x, rel_y, rel_z)
        if release_js is None:
            self.get_logger().error(
                f"IK failed for release pose at ({rel_x:.3f}, {rel_y:.3f}, {rel_z:.3f}), "
                f"link={ik_link}, error_code={self.ik_planner.last_ik_error_code}"
            )
            return
        self.job_queue.append(release_js)

        # 6) Open gripper
        self.job_queue.append('toggle_grip')

        # Example for cube_pose_in_base (0.1, 0.6, 0.05):
        #   pre-grasp -> (0.100, 0.565, 0.235)
        #   grasp     -> (0.100, 0.565, 0.210)
        #   release   -> nearest basket center + 0.15 z offset
        self.get_logger().info(
            f"Job queue built with {len(self.job_queue)} entries "
            f"(pre=({pre_x:.3f}, {pre_y:.3f}, {pre_z:.3f}), "
            f"grasp=({grasp_x:.3f}, {grasp_y:.3f}, {grasp_z:.3f}), "
            f"release=({rel_x:.3f}, {rel_y:.3f}, {rel_z:.3f}))."
        )
        self.execute_jobs()

    def _choose_basket_release(self, cx: float, cy: float, cz: float):
        """
        Pick the basket whose (x, y) is closest to the cube and return
        (x, y, z_above, basket_index). Falls back to the old offset if params are invalid.
        """
        centers = self.get_parameter('basket_centers').value

        if centers is None or len(centers) == 0 or len(centers) % 3 != 0:
            self.get_logger().warn("Basket parameters invalid; using offset release pose.")
            return cx + 0.4, cy - 0.035, cz + 0.185, None

        best_idx = None
        best_dist = float('inf')
        best_center = (cx + 0.4, cy - 0.035, cz)

        for i in range(len(centers) // 3):
            bx, by, bz = centers[3 * i: 3 * i + 3]
            dist = (bx - cx) ** 2 + (by - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_center = (float(bx), float(by), float(bz))

        rel_x, rel_y, base_rel_z = best_center
        rel_z = base_rel_z + 0.15  # hover above the basket
        self.get_logger().info(
            f"Chose basket {best_idx} at ({rel_x:.3f}, {rel_y:.3f}, {base_rel_z:.3f}); "
            f"release pose is ({rel_x:.3f}, {rel_y:.3f}, {rel_z:.3f})."
        )

        return rel_x, rel_y, rel_z, best_idx

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------
    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, JointState):
            if self.joint_state is None:
                self.get_logger().error("No current joint state; cannot run MPC.")
                return

            traj = self._plan_with_mpc(self.joint_state, next_job)
            if traj is None or len(traj.points) == 0:
                self.get_logger().error("MPC failed to produce a trajectory.")
                return

            # Visualise this MPC horizon in MoveIt/RViz
            self._publish_mpc_display_trajectory(self.joint_state, traj)

            # Execute on the UR controller
            self._execute_joint_trajectory(traj)

        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper.")
            self._toggle_gripper()

        else:
            self.get_logger().error("Unknown job type, skipping.")
            self.execute_jobs()

    # ------------------------------------------------------------------
    # MPC planning: current JointState -> target JointState -> JointTrajectory
    # ------------------------------------------------------------------
    def _plan_with_mpc(self, current_js: JointState, target_js: JointState) -> JointTrajectory:
        """
        Build a JointTrajectory using MPC between current_js and target_js.

        This uses joint-space MPC with analytic FK-based EE obstacle constraints.
        """
        # Ensure ordering consistency: map from current_js ordering to target_js ordering
        name_to_index = {name: i for i, name in enumerate(current_js.name)}

        try:
            q_current = np.array(
                [current_js.position[name_to_index[name]] for name in target_js.name],
                dtype=float,
            )
        except KeyError as e:
            self.get_logger().error(f"Joint name mismatch between current and target: {e}")
            return None

        # We don't use velocities explicitly in the cost; set dq=0
        dq_current = np.zeros_like(q_current)
        current_state = np.concatenate([q_current, dq_current])

        q_target = np.array(target_js.position, dtype=float)

        # Solve MPC – we use the full predicted horizon
        q_next, q_traj = self.mpc.compute_control(current_state, q_target)

        # --- DEBUG: see what MPC is actually planning ---
        self.get_logger().info(f"MPC q_current: {q_current}")
        self.get_logger().info(f"MPC q_target:  {q_target}")
        self.get_logger().info(f"MPC first point: {q_traj[0]}")
        self.get_logger().info(f"MPC last  point: {q_traj[-1]}")
        self.get_logger().info(
            f"MPC ||last - current|| = {np.linalg.norm(q_traj[-1] - q_current):.4f}, "
            f"||last - target|| = {np.linalg.norm(q_traj[-1] - q_target):.4f}"
        )

        # Convert to JointTrajectory
        jt = JointTrajectory()
        jt.joint_names = list(target_js.name)

        now = self.get_clock().now().to_msg()
        jt.header.stamp = now
        jt.header.frame_id = "base_link"

        H_plus_1 = q_traj.shape[0]
        for k in range(H_plus_1):
            pt = JointTrajectoryPoint()
            pt.positions = q_traj[k].tolist()

            t = float(k) * self.mpc.dt
            secs = int(t)
            nsecs = int((t - secs) * 1e9)
            pt.time_from_start = Duration(sec=secs, nanosec=nsecs)

            jt.points.append(pt)

        # --- DEBUG: show a few trajectory samples being sent to the controller ---
        if H_plus_1 >= 3:
            self.get_logger().info(
                f"MPC JT sample[0]: {jt.points[0].positions}, "
                f"t={jt.points[0].time_from_start.sec + jt.points[0].time_from_start.nanosec * 1e-9:.3f}s"
            )
            mid = H_plus_1 // 2
            self.get_logger().info(
                f"MPC JT sample[{mid}]: {jt.points[mid].positions}, "
                f"t={jt.points[mid].time_from_start.sec + jt.points[mid].time_from_start.nanosec * 1e-9:.3f}s"
            )
            self.get_logger().info(
                f"MPC JT sample[-1]: {jt.points[-1].positions}, "
                f"t={jt.points[-1].time_from_start.sec + jt.points[-1].time_from_start.nanosec * 1e-9:.3f}s"
            )

        return jt

    # ------------------------------------------------------------------
    # MoveIt DisplayTrajectory publisher
    # ------------------------------------------------------------------
    def _publish_mpc_display_trajectory(self, current_js: JointState, joint_traj: JointTrajectory):
        """
        Publish the MPC horizon as a moveit_msgs/DisplayTrajectory so that
        it appears in the MoveIt MotionPlanning RViz plugin.

        RViz side:
          - Add "MotionPlanning" display
          - In its "Planned Path" tab, set topic to /display_planned_path
          - Topic /move_group/display_planned_path is also published for setups using MoveGroup
        """
        if self.mpc_traj_pub is None:
            return

        # RobotState: use current_js as the starting state
        robot_state = RobotState()
        robot_state.joint_state = current_js

        # RobotTrajectory: wrap the JointTrajectory we just built
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj

        # DisplayTrajectory message
        display_msg = DisplayTrajectory()
        # MoveIt SRDF robot name is "ur" (not group name); use it so RViz accepts the trajectory
        display_msg.model_id = "ur"
        display_msg.trajectory_start = robot_state
        display_msg.trajectory.append(robot_traj)

        # Both topics are commonly used by RViz MotionPlanning displays:
        #   /display_planned_path (default "Planned Path" input)
        #   /move_group/display_planned_path (used by MoveGroup display)
        self.mpc_traj_pub.publish(display_msg)
        self.mpc_traj_pub_move_group.publish(display_msg)
        self.get_logger().info("Published MPC trajectory to /display_planned_path for RViz.")

    # ------------------------------------------------------------------
    # Gripper + execution
    # ------------------------------------------------------------------
    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available.')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()

    def _execute_joint_trajectory(self, joint_traj: JointTrajectory):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending MPC trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected by controller.')
            rclpy.shutdown()
            return

        self.get_logger().info('Trajectory accepted, executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            _result = future.result().result
            self.get_logger().info('Trajectory execution complete.')
            self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f'Trajectory execution failed: {e}')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_MPC_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
