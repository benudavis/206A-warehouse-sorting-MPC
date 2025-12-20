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

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTolerance
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

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

        # RViz visualization publisher
        self.mpc_viz_pub = self.create_publisher(
            MarkerArray,
            "/mpc_viz",
            10
        )

        self.action_cb_group = ReentrantCallbackGroup()
        # -----------------------------
        # Visualization options
        # -----------------------------
        self.viz_show_ee_line = True
        self.viz_show_ee_points = True
        self.viz_show_sphere_lines = True
        self.viz_show_sphere_points = False 
        self.viz_stride = 1
        self.viz_lifetime_s = 0.35

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
        # Action client & Gripper
        # -----------------------------
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.action_cb_group
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        # TF (fallback only)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -----------------------------
        # Drop locations
        # -----------------------------
        self.red_drop_location = PointStamped()
        self.red_drop_location.header.frame_id = "base_link"
        self.red_drop_location.point.x = -0.47
        self.red_drop_location.point.y = 0.43
        self.red_drop_location.point.z = -0.03

        self.blue_drop_location = PointStamped()
        self.blue_drop_location.header.frame_id = "base_link"
        self.blue_drop_location.point.x = -0.47
        self.blue_drop_location.point.y = 0.55
        self.blue_drop_location.point.z = -0.03

        # -----------------------------
        # State
        # -----------------------------
        self.joint_state = None
        self.initial_joint_state = None
        self.cube_queue = []         
        self.processing_cube = False
        self.completed_cubes = []    
        self.current_cube_pose = None
        self.current_cube_color = None
        self.obstacle_aabb = None 

        # Planning
        self.ik_planner = IKPlanner()
        self.job_queue = []

        # MPC
        self.mpc = MPCController(N=30, dt=0.08, cube_side=0.05, safety_margin=0.01)
        self.mpc_R_goal = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ], dtype=float)

        self.mpc_max_iters = 40
        self.mpc_goal_tol_m = 0.03
        self.mpc_step_duration = 0.15

        self._mpc_active = False
        self._mpc_iter = 0
        self._mpc_goal = None          
        self._mpc_pickup_h = None      
        self._mpc_job_type = None      

        self._joint_map_ready = False
        self._idx_mpc_from_js = None
        self._idx_js_from_mpc = None
        
        # -----------------------------
        # Constraints
        # -----------------------------
        self.table_height_constraint = -0.45 
        

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

    def obstacles_callback(self, msg: BoxBounds):
        self.obstacle_aabb = np.array(
            [msg.x_min, msg.x_max, msg.y_min, msg.y_max, msg.z_min, msg.z_max],
            dtype=float
        )

    def labeled_cubes_callback(self, msg: LabeledCubeArray):
        if self.joint_state is None or self.processing_cube:
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
                except Exception:
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
            self.get_logger().info(f"Queued {labeled_cube.color_label} cube")

        if self.cube_queue and not self.processing_cube:
            self._process_next_cube()

    # --------------------------------------------------------------------------
    # Joint order mapping
    # --------------------------------------------------------------------------
    def _build_joint_mapping(self, joint_names):
        expected_mpc = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        try:
            self._idx_mpc_from_js = [name_to_idx[n] for n in expected_mpc]
            self._joint_map_ready = True
        except KeyError:
            return

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
    def _duration_msg(self, seconds: float) -> Duration:
        sec = int(np.floor(seconds))
        nanosec = int(np.round((seconds - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return Duration(sec=sec, nanosec=nanosec)

    def _viz_color_for_index(self, i: int, n: int, alpha: float = 1.0) -> ColorRGBA:
        if n <= 1:
            return ColorRGBA(r=0.0, g=1.0, b=1.0, a=float(alpha))
        t = float(i) / float(max(1, n - 1))
        r = float(0.15 + 0.85 * (1.0 - t))
        g = float(0.20 + 0.80 * (t))
        b = float(0.90 - 0.70 * abs(t - 0.5) * 2.0)
        return ColorRGBA(r=r, g=g, b=b, a=float(alpha))

    def _publish_mpc_viz(self, q_traj, p_goal, aabb, pickup_h=None):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        q_traj = np.asarray(q_traj, dtype=float)
        p_goal = np.asarray(p_goal, dtype=float).reshape(3,)
        aabb = np.asarray(aabb, dtype=float).reshape(6,)

        stride = max(1, int(self.viz_stride))
        idxs = list(range(0, q_traj.shape[0], stride))
        if idxs[-1] != (q_traj.shape[0] - 1):
            idxs.append(q_traj.shape[0] - 1)

        lifetime = self._duration_msg(self.viz_lifetime_s)

        if self.viz_show_ee_line:
            path = Marker()
            path.header.frame_id = "base_link"
            path.header.stamp = now
            path.ns = "mpc"
            path.id = 0
            path.type = Marker.LINE_STRIP
            path.action = Marker.ADD
            path.pose.orientation.w = 1.0
            path.scale.x = 0.006
            path.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            path.lifetime = lifetime
            for k in idxs:
                p, _ = self._fk_ee(q_traj[k])
                path.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            ma.markers.append(path)

        if self.viz_show_ee_points:
            pts = Marker()
            pts.header.frame_id = "base_link"
            pts.header.stamp = now
            pts.ns = "mpc"
            pts.id = 3
            pts.type = Marker.SPHERE_LIST
            pts.action = Marker.ADD
            pts.pose.orientation.w = 1.0
            pts.scale.x = pts.scale.y = pts.scale.z = 0.05
            pts.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.9)
            pts.lifetime = lifetime
            for k in idxs:
                p, _ = self._fk_ee(q_traj[k])
                pts.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            ma.markers.append(pts)

        goal = Marker()
        goal.header.frame_id = "base_link"
        goal.header.stamp = now
        goal.ns = "mpc"
        goal.id = 1
        goal.type = Marker.SPHERE
        goal.action = Marker.ADD
        goal.pose.position.x = float(p_goal[0])
        goal.pose.position.y = float(p_goal[1])
        goal.pose.position.z = float(p_goal[2])
        goal.pose.orientation.w = 1.0
        goal.scale.x = goal.scale.y = goal.scale.z = 0.03
        goal.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
        goal.lifetime = lifetime
        ma.markers.append(goal)

        xmin, xmax, ymin, ymax, zmin, zmax = aabb.tolist()
        box = Marker()
        box.header.frame_id = "base_link"
        box.header.stamp = now
        box.ns = "mpc"
        box.id = 2
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x = 0.5 * (xmin + xmax)
        box.pose.position.y = 0.5 * (ymin + ymax)
        box.pose.position.z = 0.5 * (zmin + zmax)
        box.pose.orientation.w = 1.0
        box.scale.x = max(1e-6, (xmax - xmin))
        box.scale.y = max(1e-6, (ymax - ymin))
        box.scale.z = max(1e-6, (zmax - zmin))
        box.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.30)
        box.lifetime = lifetime
        ma.markers.append(box)

        if self.viz_show_sphere_lines or self.viz_show_sphere_points:
            centers_list = []
            for k in idxs:
                centers_list.append(self._sphere_centers(q_traj[k]))
            centers_over_h = np.stack(centers_list, axis=0) if centers_list else None

            if centers_over_h is not None and centers_over_h.ndim == 3:
                K, M, _ = centers_over_h.shape
                if self.viz_show_sphere_lines:
                    for j in range(M):
                        m = Marker()
                        m.header.frame_id = "base_link"
                        m.header.stamp = now
                        m.ns = "mpc_spheres"
                        m.id = 10 + j
                        m.type = Marker.LINE_STRIP
                        m.action = Marker.ADD
                        m.pose.orientation.w = 1.0
                        m.scale.x = 0.004
                        m.color = self._viz_color_for_index(j, M, alpha=0.95)
                        m.lifetime = lifetime
                        for kk in range(K):
                            c = centers_over_h[kk, j, :]
                            m.points.append(Point(x=float(c[0]), y=float(c[1]), z=float(c[2])))
                        ma.markers.append(m)

        self.mpc_viz_pub.publish(ma)

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
        if cube_pose.header.frame_id != "base_link" or drop_location.header.frame_id != "base_link":
            return False

        self.job_queue = []
        cx, cy, cz = cube_pose.point.x, cube_pose.point.y, cube_pose.point.z

        # 1. Pre-Grasp (IK)
        pre_x, pre_y, pre_z = cx + 0.005, cy - 0.030, cz + 0.185
        pre_grasp_js = self.ik_planner.compute_ik(self.joint_state, pre_x, pre_y, pre_z)
        if pre_grasp_js is None:
            self.get_logger().error("[debug:queue] IK failed for pre-grasp")
            return False
        self.job_queue.append(pre_grasp_js)

        # 2. Grasp (IK)
        grasp_x, grasp_y, grasp_z = cx + 0.005, cy - 0.03, cz + 0.14
        grasp_js = self.ik_planner.compute_ik(self.joint_state, grasp_x, grasp_y, grasp_z)
        if grasp_js is None:
            self.get_logger().error("[debug:queue] IK failed for grasp")
            self.job_queue = []
            return False
        self.job_queue.append(grasp_js)

        # 3. Grip
        self.job_queue.append('toggle_grip')

        # 4. MPC Sequence: Approach -> Descent -> Lift -> Return
        pickup_height = float(cz)
        drop_z = float(drop_location.point.z)
        approach_z = max(drop_z + 0.20, pickup_height + 0.20)

        # A: MPC Approach (Hover above drop)
        self.job_queue.append(('mpc_move', drop_location, pickup_height, approach_z, 'approach'))

        # B: MPC Descent (Go down to drop) - Using MPC instead of IK to avoid failures
        self.job_queue.append(('mpc_move', drop_location, pickup_height, drop_z, 'descent'))

        # C: Release
        self.job_queue.append('toggle_grip')

        # D: MPC Lift (Back to hover)
        self.job_queue.append(('mpc_move', drop_location, pickup_height, approach_z, 'lift'))

        # E: MPC Return Home
        if self.initial_joint_state is not None:
            q_home_js = np.array(self.initial_joint_state.position, dtype=float)
            q_home_mpc = self._q_js_to_q_mpc(q_home_js)
            p_home, _ = self._fk_ee(q_home_mpc)
            
            home_pt = PointStamped()
            home_pt.header.frame_id = "base_link"
            home_pt.point.x = float(p_home[0])
            home_pt.point.y = float(p_home[1])
            home_pt.point.z = float(p_home[2])

            self.job_queue.append(('mpc_move', home_pt, pickup_height, None, 'return'))

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
            self.current_cube_pose = None
            self.current_cube_color = None
            self.processing_cube = False
            if self.cube_queue:
                self._process_next_cube()
            return

        next_job = self.job_queue.pop(0)
        self.get_logger().info(f"[debug:jobs] Executing job: {type(next_job)}")

        # IK Move (Pre-grasp / Grasp)
        if isinstance(next_job, JointState):
            traj = self.ik_planner.plan_to_joints(next_job)
            if traj is None:
                self.get_logger().error("[debug:moveit] MoveIt plan failed")
                self._abort_cube()
                return
            self._execute_joint_trajectory(traj.joint_trajectory, segment_tag="moveit_segment")
            return

        # MPC Move (Approach, Descent, Lift, Return)
        # tuple format: ('mpc_move', target_pt, pickup_h, target_z, phase)
        if isinstance(next_job, tuple) and len(next_job) == 5 and next_job[0] == 'mpc_move':
            _, target_pt, pickup_h, target_z, phase = next_job

            if self.obstacle_aabb is None:
                self.get_logger().error("[debug:mpc] obstacle_aabb is None")
                self._abort_cube()
                return

            self._mpc_active = True
            self._mpc_iter = 0
            self._mpc_job_type = phase

            # Goal configuration
            tx, ty = target_pt.point.x, target_pt.point.y
            tz = target_z if target_z is not None else target_pt.point.z
            self._mpc_goal = np.array([tx, ty, float(tz)], dtype=float)

            # Floor constraint configuration based on Phase
            aabb = np.asarray(self.obstacle_aabb, dtype=float).reshape(6,)
            
            if phase == 'return':
                # Force flying OVER the obstacle
                self._mpc_pickup_h = float(aabb[5])
            elif phase == 'descent':
                # Going down: floor is the strict table height constraint (or even lower if needed)
                self._mpc_pickup_h = self.table_height_constraint
            else:
                # Normal Transport: floor is the strict table height constraint
                self._mpc_pickup_h = self.table_height_constraint

            self.get_logger().info(f"[MPC] Phase={phase} Goal={self._mpc_goal} Floor={self._mpc_pickup_h}")
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
            self._abort_cube()
            return
        if self.obstacle_aabb is None:
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

        if dist0 < self.mpc_goal_tol_m:
            self.get_logger().info(f"[mpc] goal reached within tol {self.mpc_goal_tol_m:.3f}m -> exit MPC")
            self._mpc_active = False
            self.execute_jobs()
            return

        out = self.mpc.solve(
            q0=q0_mpc,
            p_goal=p_goal,
            R_goal=self.mpc_R_goal,
            aabb_bounds=aabb,
            pickup_height=pickup_h
        )

        if not out.get("success", False):
            self.get_logger().error(f"[mpc] MPC failed.")
            self._abort_cube()
            return

        q_traj = np.asarray(out["q_traj"], dtype=float)
        dq_traj = np.asarray(out["dq_traj"], dtype=float)

        self._publish_mpc_viz(q_traj=q_traj, p_goal=p_goal, aabb=aabb, pickup_h=pickup_h)

        # First predicted step
        q1_mpc = q_traj[1].copy() if q_traj.shape[0] >= 2 else (q0_mpc + dq_traj[0])
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
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        
        # Tolerances to prevent hanging
        goal.goal_time_tolerance = Duration(sec=0, nanosec=500_000_000) 
        for name in joint_traj.joint_names:
            tol = JointTolerance()
            tol.name = name
            tol.position = 0.01  
            tol.velocity = 0.01
            goal.goal_tolerance.append(tol)

        if not self.exec_ac.wait_for_server(timeout_sec=2.0):
             self.get_logger().error(f"[debug:{segment_tag}] Action server not available!")
             self._abort_cube()
             return

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
            ok = (ec == 0)

            if on_done_cb:
                on_done_cb(ok, es)
            else:
                if ok:
                    self.execute_jobs()
                else:
                    self.get_logger().error(f"Execution failed with code {ec}")
                    self._abort_cube()
        except Exception as e:
            self.get_logger().error(f"[debug:{segment_tag}] execution result exception: {e}")
            if on_done_cb:
                on_done_cb(False, f"result_exception:{e}")
            else:
                self._abort_cube()


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()