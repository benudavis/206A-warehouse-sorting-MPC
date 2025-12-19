#!/usr/bin/env python3
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from custom_msgs.msg import LabeledCubeArray, BoxBounds

from planning.ik import IKPlanner
from planning.mpc_controller import MPCController

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time

import numpy as np
import casadi as ca


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_to_nearest(q_ref: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Return q_target adjusted by +/-2pi so each joint is closest to q_ref.
    This prevents controllers from taking the long way around.
    """
    q_ref = np.asarray(q_ref, dtype=float).reshape(-1)
    q_target = np.asarray(q_target, dtype=float).reshape(-1)
    d = wrap_to_pi(q_target - q_ref)
    return q_ref + d


class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.labeled_cubes_sub = self.create_subscription(
            LabeledCubeArray, '/labeled_cubes_base', self.labeled_cubes_callback, 10
        )
        self.obstacles_sub = self.create_subscription(
            BoxBounds, '/obstacles_base', self.obstacles_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 1
        )

        # -----------------------------
        # Action client for arm trajectories
        # -----------------------------
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Service client for gripper (toggle)
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        # TF (fallback only)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -----------------------------
        # Drop locations
        # -----------------------------
        self.red_drop_location = PointStamped()
        self.red_drop_location.header.frame_id = "base_link"
        self.red_drop_location.point.x = -0.37
        self.red_drop_location.point.y = 0.47
        self.red_drop_location.point.z = -0.03

        self.blue_drop_location = PointStamped()
        self.blue_drop_location.header.frame_id = "base_link"
        self.blue_drop_location.point.x = -0.37
        self.blue_drop_location.point.y = 0.68
        self.blue_drop_location.point.z = -0.03

        # -----------------------------
        # State
        # -----------------------------
        self.joint_state = None
        self.initial_joint_state = None
        self.cube_queue = []         # list of (PointStamped, color, drop_location)
        self.processing_cube = False
        self.completed_cubes = []    # list of (x, y, z)
        self.current_cube_pose = None
        self.current_cube_color = None

        self.obstacle_aabb = None  # np.array([xmin,xmax,ymin,ymax,zmin,zmax])

        # Planning
        self.ik_planner = IKPlanner()
        self.job_queue = []

        # MPC
        # NOTE:
        #  - obstacle avoidance is HARD in the controller now
        #  - facing-down enforced ONLY at terminal (k=N)
        self.mpc = MPCController(N=30, dt=0.1, cube_side=0.05, safety_margin=0.01)
        self.mpc_R_goal = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ], dtype=float)

        # Closed-loop MPC parameters
        self.mpc_max_iters = 40
        self.mpc_goal_tol_m = 0.03

        # RECOMMENDED: shorten step, reduces “AABB changed during execution” risk
        self.mpc_step_duration = 0.15

        self._mpc_active = False
        self._mpc_iter = 0
        self._mpc_goal = None          # np.array(3,)
        self._mpc_pickup_h = None      # float
        self._mpc_drop_location = None # PointStamped

        # Joint mapping (JS order -> MPC order)
        self._joint_map_ready = False
        self._idx_mpc_from_js = None
        self._idx_js_from_mpc = None

        self.get_logger().info(
            f"[debug:init] MPCController: N={getattr(self.mpc,'N',None)} dt={getattr(self.mpc,'dt',None)} "
            f"has_fk_p={hasattr(self.mpc,'fk_p')} has_fk_R={hasattr(self.mpc,'fk_R')} "
            f"has_solve={hasattr(self.mpc,'solve')}"
        )
        self.get_logger().info("main_waypoints node started")

    # --------------------------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

        if not self._joint_map_ready and msg.name:
            self._build_joint_mapping(msg.name)

        if self.initial_joint_state is None:
            self.initial_joint_state = JointState()
            self.initial_joint_state.header = msg.header
            self.initial_joint_state.name = list(msg.name)
            self.initial_joint_state.position = list(msg.position)
            self.initial_joint_state.velocity = list(msg.velocity) if msg.velocity else []
            self.initial_joint_state.effort = list(msg.effort) if msg.effort else []
            self.get_logger().info("Stored initial/home joint state")
            self.get_logger().info("Node is ready. Waiting for cube detections...")

    def obstacles_callback(self, msg: BoxBounds):
        self.obstacle_aabb = np.array(
            [msg.x_min, msg.x_max, msg.y_min, msg.y_max, msg.z_min, msg.z_max],
            dtype=float
        )
        self.get_logger().info(f"[debug:obstacle] aabb={self.obstacle_aabb.tolist()}")

    def labeled_cubes_callback(self, msg: LabeledCubeArray):
        if self.joint_state is None:
            return
        if self.processing_cube:
            return

        for labeled_cube in msg.cubes:
            if labeled_cube.color_label not in ['red', 'black']:
                continue

            cube_pose_stamped = labeled_cube.point
            if not cube_pose_stamped.header.frame_id:
                cube_pose_stamped.header.frame_id = "base_link"
            cube_pose_stamped.header.stamp = self.get_clock().now().to_msg()

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
                        f"[debug:cubes] TF failed from {cube_pose_stamped.header.frame_id} to base_link: {e}"
                    )
                    continue
            else:
                cube_pose_base = cube_pose_stamped

            cx, cy, cz = cube_pose_base.point.x, cube_pose_base.point.y, cube_pose_base.point.z

            if self._is_near_completed(cx, cy, cz):
                continue
            if self._is_already_queued(cx, cy, cz):
                continue

            drop_location = self.red_drop_location if labeled_cube.color_label == 'red' else self.blue_drop_location
            self.cube_queue.append((cube_pose_base, labeled_cube.color_label, drop_location))

            self.get_logger().info(
                f"Queued {labeled_cube.color_label} cube at "
                f"({cx:.3f}, {cy:.3f}, {cz:.3f}) "
                f"→ drop at ({drop_location.point.x:.3f}, {drop_location.point.y:.3f}, {drop_location.point.z:.3f})"
            )

        if self.cube_queue and not self.processing_cube:
            self._process_next_cube()

    # --------------------------------------------------------------------------
    # Joint order mapping
    # --------------------------------------------------------------------------
    def _build_joint_mapping(self, joint_names):
        expected_mpc = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        try:
            idx_mpc_from_js = [name_to_idx[n] for n in expected_mpc]
        except KeyError as e:
            self.get_logger().error(f"[joint_map] missing joint in /joint_states: {e}")
            return

        # quick round-trip check
        if self.joint_state and self.joint_state.position and len(self.joint_state.position) >= 6:
            q_js = np.array(self.joint_state.position, dtype=float)
            q_mpc = q_js[idx_mpc_from_js]
            q_js_rt = q_js.copy()
            q_js_rt[idx_mpc_from_js] = q_mpc
            err = float(np.linalg.norm(q_js_rt - q_js))

            self.get_logger().info(f"[joint_map] joint_states order: {', '.join(joint_names)}")
            self.get_logger().info(f"[joint_map] MPC FK expects:      {', '.join(expected_mpc)}")
            self.get_logger().info(f"[joint_map] indices (MPC<-JS): {idx_mpc_from_js} (i.e., q_mpc = q_js[these])")
            self.get_logger().info(f"[joint_map] round-trip ||q_js_rt - q_js|| = {err:.6e}")

        self._idx_mpc_from_js = idx_mpc_from_js
        self._joint_map_ready = True

    def _q_js_to_q_mpc(self, q_js):
        q_js = np.asarray(q_js, dtype=float).reshape(-1)
        return q_js[self._idx_mpc_from_js].copy()

    def _q_mpc_to_q_js(self, q_mpc, q_js_current):
        q_mpc = np.asarray(q_mpc, dtype=float).reshape(6,)
        q_js_current = np.asarray(q_js_current, dtype=float).reshape(-1)

        q_js_target = q_js_current.copy()
        for mpc_i, js_i in enumerate(self._idx_mpc_from_js):
            q_js_target[js_i] = q_mpc[mpc_i]

        q_js_unwrapped = q_js_target.copy()
        for js_i in self._idx_mpc_from_js:
            q_js_unwrapped[js_i] = float(unwrap_to_nearest(q_js_current[js_i], q_js_target[js_i]))

        return q_js_unwrapped

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _is_near_completed(self, x, y, z, thresh=0.05) -> bool:
        for px, py, pz in self.completed_cubes:
            if np.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) < thresh:
                return True
        return False

    def _is_already_queued(self, x, y, z, thresh=0.05) -> bool:
        for queued_cube_pose, _, _ in self.cube_queue:
            qx = queued_cube_pose.point.x
            qy = queued_cube_pose.point.y
            qz = queued_cube_pose.point.z
            if np.sqrt((x - qx) ** 2 + (y - qy) ** 2 + (z - qz) ** 2) < thresh:
                return True
        return False

    def _compute_mpc_approach_z(self, pickup_height: float, drop_z: float) -> float:
        max_r = float(max(self.mpc.sphere_radii))
        max_off_z = float(max([o[2] for o in self.mpc.sphere_offsets_ee]))
        required_ee_z = float(pickup_height) + float(self.mpc.table_margin) + max_r + max_off_z
        required_ee_z += 0.02
        return max(float(drop_z) + 0.20, required_ee_z)

    def _fk_ee(self, q_mpc: np.ndarray):
        p = np.array(self.mpc.fk_p(ca.DM(q_mpc))).astype(float).reshape(3,)
        R = np.array(self.mpc.fk_R(ca.DM(q_mpc))).astype(float).reshape(3, 3)
        return p, R

    def _sphere_centers(self, q_mpc: np.ndarray):
        p, R = self._fk_ee(q_mpc)
        centers = []
        for off in self.mpc.sphere_offsets_ee:
            centers.append(p + R @ np.asarray(off, dtype=float).reshape(3,))
        return np.array(centers, dtype=float)

    def _clearances(self, q_mpc: np.ndarray, aabb: np.ndarray, pickup_height: float):
        aabb = np.asarray(aabb, dtype=float).reshape(6,)
        centers = self._sphere_centers(q_mpc)
        M = centers.shape[0]
        obs_clear = np.zeros((M,), dtype=float)
        tab_clear = np.zeros((M,), dtype=float)

        for i in range(M):
            d = float(self.mpc.dist_point_aabb(ca.DM(centers[i]), ca.DM(aabb)))
            obs_clear[i] = d - float(self.mpc.sphere_radii[i])

            thresh = float(pickup_height) + float(self.mpc.table_margin) + float(self.mpc.sphere_radii[i])
            tab_clear[i] = float(centers[i, 2]) - thresh

        return obs_clear, tab_clear

    # --------------------------------------------------------------------------
    # Cube pipeline
    # --------------------------------------------------------------------------
    def _process_next_cube(self):
        if not self.cube_queue or self.processing_cube:
            return
        if self.joint_state is None:
            return
        if not self._joint_map_ready:
            self.get_logger().warn("[debug:pipeline] joint mapping not ready yet")
            return

        cube_pose, color, drop_location = self.cube_queue.pop(0)
        self.processing_cube = True
        self.current_cube_pose = cube_pose
        self.current_cube_color = color

        self.get_logger().info(
            f"Processing {color} cube at "
            f"({cube_pose.point.x:.3f}, {cube_pose.point.y:.3f}, {cube_pose.point.z:.3f})"
        )

        ok = self._build_cube_job_queue(cube_pose, drop_location)
        self.get_logger().info(f"[debug:pipeline] built_job_queue ok={ok} jobs={len(self.job_queue)}")
        if not ok:
            self._abort_cube()
            if self.cube_queue:
                self._process_next_cube()
            return

        self.execute_jobs()

    def _build_cube_job_queue(self, cube_pose: PointStamped, drop_location: PointStamped) -> bool:
        if cube_pose.header.frame_id != "base_link":
            self.get_logger().error(f"[debug:queue] Cube pose not base_link: {cube_pose.header.frame_id}")
            self.job_queue = []
            return False
        if drop_location.header.frame_id != "base_link":
            self.get_logger().error(f"[debug:queue] Drop not base_link: {drop_location.header.frame_id}")
            self.job_queue = []
            return False

        self.job_queue = []
        cx, cy, cz = cube_pose.point.x, cube_pose.point.y, cube_pose.point.z

        # Pre-grasp
        pre_x, pre_y, pre_z = cx + 0.005, cy - 0.030, cz + 0.183
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error("[debug:queue] IK failed for pre-grasp")
            return False
        self.job_queue.append(pre_grasp_js)

        # Grasp
        grasp_x, grasp_y, grasp_z = cx + 0.005, cy - 0.03, cz + 0.14
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error("[debug:queue] IK failed for grasp")
            self.job_queue = []
            return False
        self.job_queue.append(grasp_js)

        self.job_queue.append('toggle_grip')

        pickup_height = float(cz)
        approach_z = self._compute_mpc_approach_z(pickup_height=pickup_height, drop_z=float(drop_location.point.z))
        self.job_queue.append(('mpc_to_drop', drop_location, pickup_height, approach_z))

        drop_x, drop_y, drop_z = drop_location.point.x, drop_location.point.y, drop_location.point.z
        drop_js = self.ik_planner.compute_ik(self.joint_state, drop_x, drop_y, drop_z)
        if drop_js is None:
            self.get_logger().error("[debug:queue] IK failed for final drop")
            self.job_queue = []
            return False
        self.job_queue.append(drop_js)

        self.job_queue.append('toggle_grip')

        if self.initial_joint_state is not None:
            self.job_queue.append(('return_home', self.initial_joint_state))

        pretty = []
        for j in self.job_queue:
            if isinstance(j, JointState):
                pretty.append("JointState")
            elif isinstance(j, tuple) and len(j) > 0:
                pretty.append(f"tuple:{j[0]}")
            else:
                pretty.append(str(j))
        self.get_logger().info(f"[debug:queue] job_types={pretty}")
        return True

    # --------------------------------------------------------------------------
    # Job execution
    # --------------------------------------------------------------------------
    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("[debug:jobs] done with cube job queue")

            if self.current_cube_pose is not None:
                px, py, pz = self.current_cube_pose.point.x, self.current_cube_pose.point.y, self.current_cube_pose.point.z
                self.completed_cubes.append((px, py, pz))
                self.get_logger().info(f"[debug:jobs] marked completed ({px:.3f},{py:.3f},{pz:.3f})")

            self.current_cube_pose = None
            self.current_cube_color = None
            self.processing_cube = False

            if self.cube_queue:
                self._process_next_cube()
            return

        next_job = self.job_queue.pop(0)
        tag = (
            'toggle_grip' if next_job == 'toggle_grip' else
            (next_job[0] if isinstance(next_job, tuple) else
             'JointState')
        )
        self.get_logger().info(f"[debug:jobs] pop job={type(next_job)} value={tag} remaining={len(self.job_queue)}")

        # Return home
        if isinstance(next_job, tuple) and len(next_job) == 2 and next_job[0] == 'return_home':
            home_joint_state = next_job[1]
            traj = self.ik_planner.plan_to_joints(home_joint_state)
            if traj is None:
                self.get_logger().error("[debug:home] MoveIt plan failed")
                self._abort_cube()
                return
            self._execute_joint_trajectory(traj.joint_trajectory, segment_tag="moveit_home")
            return

        # MoveIt JointState
        if isinstance(next_job, JointState):
            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("[debug:moveit] MoveIt plan failed")
                self._abort_cube()
                return
            self._execute_joint_trajectory(traj.joint_trajectory, segment_tag="moveit_segment")
            return

        # MPC closed-loop
        if isinstance(next_job, tuple) and len(next_job) == 4 and next_job[0] == 'mpc_to_drop':
            _, drop_location, pickup_height, approach_z = next_job

            if self.obstacle_aabb is None:
                self.get_logger().error("[debug:mpc] obstacle_aabb is None (MPC will not run)")
                self._abort_cube()
                return

            self._mpc_active = True
            self._mpc_iter = 0
            self._mpc_drop_location = drop_location

            self._mpc_goal = np.array([drop_location.point.x, drop_location.point.y, float(approach_z)], dtype=float)

            aabb = np.asarray(self.obstacle_aabb, dtype=float).reshape(6,)
            pickup_height_eff = float(min(float(pickup_height), float(aabb[4])))
            self._mpc_pickup_h = pickup_height_eff

            self.get_logger().info(
                f"[debug:mpc] ENTER CLOSED-LOOP MPC mode p_goal=({self._mpc_goal[0]:.3f},{self._mpc_goal[1]:.3f},{self._mpc_goal[2]:.3f}) "
                f"pickup_height={pickup_height_eff:.3f}"
            )

            self._mpc_step_replan_and_execute()
            return

        # Toggle grip
        if next_job == 'toggle_grip':
            self.get_logger().info("[debug:grip] toggling gripper")
            self._toggle_gripper()
            return

        self.get_logger().error(f"[debug] Unknown job type: {type(next_job)}")
        self._abort_cube()

    def _abort_cube(self):
        self.get_logger().error("[debug] aborting cube: clearing queue and resetting flags")
        self.processing_cube = False
        self.job_queue = []
        self.current_cube_pose = None
        self.current_cube_color = None
        self._mpc_active = False
        self._mpc_iter = 0
        self._mpc_goal = None
        self._mpc_pickup_h = None
        self._mpc_drop_location = None

    # --------------------------------------------------------------------------
    # Closed-loop MPC step
    # --------------------------------------------------------------------------
    def _mpc_step_replan_and_execute(self):
        if not self._mpc_active:
            return
        if self.joint_state is None or not self._joint_map_ready:
            self.get_logger().warn("[mpc] missing joint_state or joint_map")
            self._abort_cube()
            return
        if self.obstacle_aabb is None:
            self.get_logger().warn("[mpc] missing obstacle_aabb")
            self._abort_cube()
            return

        self._mpc_iter += 1
        if self._mpc_iter > self.mpc_max_iters:
            self.get_logger().error("[mpc] exceeded max iters")
            self._abort_cube()
            return

        q_js = np.asarray(self.joint_state.position, dtype=float)
        q0_mpc = self._q_js_to_q_mpc(q_js)

        p_goal = np.asarray(self._mpc_goal, dtype=float).reshape(3,)
        aabb = np.asarray(self.obstacle_aabb, dtype=float).reshape(6,)
        pickup_h = float(self._mpc_pickup_h)

        ee_p0, ee_R0 = self._fk_ee(q0_mpc)
        delta = p_goal - ee_p0
        dist0 = float(np.linalg.norm(delta))

        self.get_logger().info(
            f"[mpc] iter={self._mpc_iter} ee=({ee_p0[0]:.3f},{ee_p0[1]:.3f},{ee_p0[2]:.3f}) "
            f"p_goal=({p_goal[0]:.3f},{p_goal[1]:.3f},{p_goal[2]:.3f}) "
            f"delta=({delta[0]:.3f},{delta[1]:.3f},{delta[2]:.3f}) dist={dist0:.4f}"
        )
        self.get_logger().info(f"[mpc] aabb={aabb.tolist()} pickup_height={pickup_h:.3f}")

        if dist0 < self.mpc_goal_tol_m:
            self.get_logger().info(f"[mpc] goal reached within tol {self.mpc_goal_tol_m:.3f}m -> exit MPC")
            self._mpc_active = False
            self.execute_jobs()
            return

        out = self.mpc.solve(
            q0=q0_mpc,
            p_goal=p_goal,
            R_goal=self.mpc_R_goal,  # terminal-only facing uses this only at k=N
            aabb_bounds=aabb,
            pickup_height=pickup_h
        )

        self.get_logger().info(f"[mpc] solve() success={out.get('success', False)} status={out.get('solver_status', 'unknown')}")

        if not out.get("success", False):
            self.get_logger().error(f"[mpc] MPC failed. diagnostics={['ipopt_status']}")
            self._abort_cube()
            return

        q_traj = np.asarray(out["q_traj"], dtype=float)
        dq_traj = np.asarray(out["dq_traj"], dtype=float)

        # First predicted step (k=1)
        q1_mpc = q_traj[1].copy() if q_traj.shape[0] >= 2 else (q0_mpc + dq_traj[0])

        # Terminal facing slack (scalar)
        s_face_term = float(out.get("s_face", 0.0))
        s_tab = out.get("s_tab", None)
        if s_tab is not None:
            s_tab = np.asarray(s_tab, dtype=float)
            tab_max = float(np.max(s_tab)) if s_tab.size else 0.0
            tab_mean = float(np.mean(s_tab)) if s_tab.size else 0.0
        else:
            tab_max, tab_mean = 0.0, 0.0

        self.get_logger().info(
            f"[mpc][slacks] terminal_face={s_face_term:.4e} tab_max={tab_max:.4e} tab_mean={tab_mean:.4e}"
        )

        # First-step distance + clearance diagnostics
        ee_p1, ee_R1 = self._fk_ee(q1_mpc)
        dist1 = float(np.linalg.norm(ee_p1 - p_goal))

        obs0, tab0 = self._clearances(q0_mpc, aabb, pickup_h)
        obs1, tab1 = self._clearances(q1_mpc, aabb, pickup_h)

        worst0 = int(np.argmin(obs0))
        worst1 = int(np.argmin(obs1))
        obsClear0 = float(obs0[worst0])
        obsClear1 = float(obs1[worst1])

        wtab0 = int(np.argmin(tab0))
        wtab1 = int(np.argmin(tab1))
        tabClear0 = float(tab0[wtab0])
        tabClear1 = float(tab1[wtab1])

        self.get_logger().info(
            f"[mpc][first] dist0={dist0:.4f} dist1={dist1:.4f} dDist={dist1 - dist0:+.4f} | "
            f"obsClear0={obsClear0:+.4f} obsClear1={obsClear1:+.4f} dObsClear={obsClear1-obsClear0:+.4f} | "
            f"tabClear0={tabClear0:+.4f} tabClear1={tabClear1:+.4f} dTabClear={tabClear1-tabClear0:+.4f}"
        )

        # HARD constraint sanity warning: predicted first step should NOT violate obstacle clearance
        if obsClear1 < -1e-6:
            self.get_logger().error(
                f"[mpc][HARD_VIOLATION?] predicted obsClear1={obsClear1:+.6f} < 0 "
                f"(FK/AABB mismatch or controller not enforcing hard constraint)"
            )

        # Facing debug: show it but remember it's ONLY required at terminal now
        z0 = ee_R0[:, 2]
        z1 = ee_R1[:, 2]
        dot0 = float(-z0[2])
        dot1 = float(-z1[2])
        lat0 = float(np.linalg.norm(z0[0:2]))
        lat1 = float(np.linalg.norm(z1[0:2]))
        self.get_logger().info(
            f"[mpc][face_now] dot0={dot0:+.4f} dot1={dot1:+.4f} lat0={lat0:+.4f} lat1={lat1:+.4f} "
            f"(NOTE: facing-down required only at terminal; this is just telemetry)"
        )

        dq0 = np.asarray(out["dq0"], dtype=float).reshape(6,)
        self.get_logger().info(f"[mpc] step ||dq_ctrl||={float(np.linalg.norm(dq0)):.6f}")

        q1_js_unwrapped = self._q_mpc_to_q_js(q1_mpc, q_js)

        jt = self._one_step_joint_trajectory(q_js, q1_js_unwrapped, duration=self.mpc_step_duration)
        self._execute_joint_trajectory(jt, segment_tag="mpc_step", on_done_cb=self._on_mpc_step_done)

    def _on_mpc_step_done(self, success: bool, error_string: str):
        if not success:
            self.get_logger().error(f"[mpc_step] execution failed: {error_string}")
            self._abort_cube()
            return
        self._mpc_step_replan_and_execute()

    # --------------------------------------------------------------------------
    # Trajectory helpers
    # --------------------------------------------------------------------------
    def _one_step_joint_trajectory(self, q0_js: np.ndarray, q1_js: np.ndarray, duration: float) -> JointTrajectory:
        jt = JointTrajectory()
        jt.joint_names = list(self.joint_state.name)

        p0 = JointTrajectoryPoint()
        p0.positions = [float(v) for v in q0_js.tolist()]
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = 0

        p1 = JointTrajectoryPoint()
        p1.positions = [float(v) for v in q1_js.tolist()]
        sec = int(np.floor(duration))
        nsec = int(np.round((duration - sec) * 1e9))
        if nsec >= 1_000_000_000:
            sec += 1
            nsec -= 1_000_000_000
        p1.time_from_start.sec = sec
        p1.time_from_start.nanosec = nsec

        jt.points = [p0, p1]
        return jt

    # --------------------------------------------------------------------------
    # Gripper
    # --------------------------------------------------------------------------
    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('[debug:grip] Gripper service not available')
            self._abort_cube()
            return

        req = Trigger.Request()
        fut = self.gripper_cli.call_async(req)
        fut.add_done_callback(self._on_gripper_done)

    def _on_gripper_done(self, future):
        try:
            resp = future.result()
            self.get_logger().info(f"[debug:grip] response success={resp.success} msg='{resp.message}'")
            if not resp.success:
                self._abort_cube()
                return
        except Exception as e:
            self.get_logger().error(f"[debug:grip] service call failed: {e}")
            self._abort_cube()
            return

        self.execute_jobs()

    # --------------------------------------------------------------------------
    # Trajectory execution
    # --------------------------------------------------------------------------
    def _execute_joint_trajectory(self, joint_traj: JointTrajectory, segment_tag: str = "traj", on_done_cb=None):
        try:
            npts = len(joint_traj.points)
            t0 = joint_traj.points[0].time_from_start.sec + 1e-9 * joint_traj.points[0].time_from_start.nanosec
            tN = joint_traj.points[-1].time_from_start.sec + 1e-9 * joint_traj.points[-1].time_from_start.nanosec
            qN = joint_traj.points[-1].positions
            self.get_logger().info(
                f"[debug:{segment_tag}] sending traj points={npts} T={tN - t0:.3f}s final_q={np.round(qN,3).tolist()}"
            )
        except Exception as e:
            self.get_logger().warn(f"[debug:{segment_tag}] traj stats failed: {e}")

        self.exec_ac.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info(f"[debug:{segment_tag}] send_goal_async()")
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(lambda fut: self._on_goal_sent(fut, segment_tag, on_done_cb))

    def _on_goal_sent(self, future, segment_tag: str, on_done_cb):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(f"[debug:{segment_tag}] goal NOT accepted")
                if on_done_cb:
                    on_done_cb(False, "goal_not_accepted")
                else:
                    self._abort_cube()
                return

            self.get_logger().info(f"[debug:{segment_tag}] goal accepted -> waiting result")
            res_future = goal_handle.get_result_async()
            res_future.add_done_callback(lambda fut: self._on_exec_done(fut, segment_tag, on_done_cb))
        except Exception as e:
            self.get_logger().error(f"[debug:{segment_tag}] send goal exception: {e}")
            if on_done_cb:
                on_done_cb(False, f"send_goal_exception:{e}")
            else:
                self._abort_cube()

    def _on_exec_done(self, future, segment_tag: str, on_done_cb):
        try:
            res = future.result().result
            ec = getattr(res, "error_code", None)
            es = getattr(res, "error_string", "")
            self.get_logger().info(f"[debug:{segment_tag}] execution complete error_code={ec} error_string='{es}'")
            ok = (ec == 0)

            if on_done_cb:
                on_done_cb(ok, es)
            else:
                self.execute_jobs()
        except Exception as e:
            self.get_logger().error(f"[debug:{segment_tag}] execution result exception: {e}")
            if on_done_cb:
                on_done_cb(False, f"result_exception:{e}")
            else:
                self._abort_cube()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
