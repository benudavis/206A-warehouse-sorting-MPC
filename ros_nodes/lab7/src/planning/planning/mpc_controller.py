#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import casadi as ca

# Make MuJoCo optional – safe for ROS-only use
try:
    import mujoco
except ImportError:
    mujoco = None


# ----------------------------------------------------------------------
# Analytic FK: q (6,) -> EE position (3,) in UR base_link frame
# ----------------------------------------------------------------------
def build_ur5e_fk_function():
    """
    Build a CasADi function fk(q) -> p_ee (3,) in UR base_link frame.

    Conventions:
        - q is a 6-element vector [q1..q6] in radians.
        - EE position is at arm_hand_pinch site (pinch point).
        - Output is in UR base frame (for ROS: base_link).
    """
    # Joint vector
    q = ca.SX.sym("q", 6)

    # Basis vectors
    ex = ca.SX([1.0, 0.0, 0.0])
    ey = ca.SX([0.0, 1.0, 0.0])
    ez = ca.SX([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # Kinematic parameters (H, P, R_6T) from optimized UR5e model
    # ------------------------------------------------------------------
    H = ca.SX(3, 6)
    H[:, 0] = ez
    H[:, 1] = -ey
    H[:, 2] = -ey
    H[:, 3] = -ey
    H[:, 4] = -ez
    H[:, 5] = -ey

    P = ca.SX(3, 7)
    P[:, 0] = ca.SX([-0.0002000011246430994, 0.12381999433286241, 0.13800000017695882])
    P[:, 1] = ca.SX([-2.5310463076446646e-09, -0.00017500068716176992, -0.024499999824680645])
    P[:, 2] = ca.SX([-0.4250000085106623, -0.00017500068667997845, -9.379979883136704e-09])
    P[:, 3] = ca.SX([-0.39200000521049083, -0.00017500068973860236, -2.1210003962969342e-09])
    P[:, 4] = ca.SX([-2.5087870245593798e-09, -0.13347500069013965, -0.09985000022399897])
    P[:, 5] = ca.SX([-2.0967273265274825e-09, -0.06155999960694523, -0.00015000022379548536])
    P[:, 6] = ca.SX([-3.7045055566769567e-09, -0.16115999966160519, 0.04929999208756993])

    # Tool rotation R_6T = Rot_x(+pi/2)
    th_tool = ca.pi / 2
    c = ca.cos(th_tool)
    s = ca.sin(th_tool)
    R_6T = ca.SX(3, 3)
    R_6T[0, 0] = 1.0
    R_6T[0, 1] = 0.0
    R_6T[0, 2] = 0.0
    R_6T[1, 0] = 0.0
    R_6T[1, 1] = c
    R_6T[1, 2] = -s
    R_6T[2, 0] = 0.0
    R_6T[2, 1] = s
    R_6T[2, 2] = c

    # Rodrigues rotation
    def rot_axis(w: ca.SX, theta: ca.SX) -> ca.SX:
        wx, wy, wz = w[0], w[1], w[2]
        zero = ca.SX(0)
        row1 = ca.hcat([zero, -wz, wy])
        row2 = ca.hcat([wz, zero, -wx])
        row3 = ca.hcat([-wy, wx, zero])
        W = ca.vertcat(row1, row2, row3)
        I3 = ca.SX_eye(3)
        return I3 + ca.sin(theta) * W + (1.0 - ca.cos(theta)) * (W @ W)

    # Forward kinematics base -> tool0
    R = ca.SX_eye(3)
    p = P[:, 0]  # p01

    for i in range(6):
        R = R @ rot_axis(H[:, i], q[i])
        if i < 5:
            p = p + R @ P[:, i + 1]

    R_06 = R
    p_06 = p

    # tool0 frame
    p_0T = p_06 + R_06 @ P[:, 6]
    R_0T = R_06 @ R_6T

    # Offset from tool0 to pinch site (base frame)
    TOOL0_TO_PINCH = np.array([0.0, -0.0493, 0.03308], dtype=float)
    offset_pinch = ca.SX(TOOL0_TO_PINCH)

    # Pinch position in UR base frame (for ROS: base_link)
    p_pinch_base = p_0T + R_0T @ offset_pinch  # (3,)

    fk_fun = ca.Function("fk_ur5e_pinch_base", [q], [p_pinch_base])
    return fk_fun


# ----------------------------------------------------------------------
# MPC controller with EE obstacle avoidance using analytic FK
# ----------------------------------------------------------------------
class MPCController:
    """MPC controller for a position-controlled arm with analytic FK obstacle avoidance."""

    def __init__(
        self,
        n_joints: int = 6,
        horizon: int = 30,
        dt: float = 0.05,
        enable_fk: bool = True,
    ):
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt
        self.enable_fk = enable_fk

        # Cost weights (joint-space tracking + smoothness)
        self.Q = np.eye(n_joints) * 500.0
        self.R = np.eye(n_joints) * 0.1
        self.Q_terminal = np.eye(n_joints) * 1000.0

        # Obstacle avoidance (soft margin)
        self.obstacle_weight = 1e4

        # Joint & velocity constraints
        self.joint_limits = (
            np.array([-2 * np.pi] * n_joints),
            np.array([+2 * np.pi] * n_joints),
        )
        self.max_velocity = 3.0  # [rad/s]

        # Obstacles: list of (center[3], half_size[3]) in base_link frame
        self.obstacles = []
        self.n_max_obstacles = 10
        self.safety_margin = 0.12  # [m]

        # Optional MuJoCo link body IDs
        self.link_body_ids = []

        # Analytic FK
        self.fk_fun = None
        if self.enable_fk:
            try:
                self.fk_fun = build_ur5e_fk_function()
                print("✓ Using analytic UR5e FK (base_link frame) inside MPC.")
            except Exception as e:
                print(f"⚠ Failed to build analytic UR FK: {e}")
                print("  MPC obstacle avoidance will have no analytic EE constraints.")
        else:
            print("Analytic FK disabled in MPC (enable_fk=False).")

        self.solver = None
        self.prev_solution = None
        self._setup_optimization()

    # ------------------------------------------------------------------
    # Public configuration helpers
    # ------------------------------------------------------------------
    def initialize_link_bodies(self, model):
        """Initialize bodies used for MuJoCo collision checks (optional)."""
        if mujoco is None:
            print("MuJoCo not available; skipping link body initialization.")
            return

        if self.link_body_ids:
            return

        candidate_names = [
            "arm_base_link",
            "arm_shoulder_link",
            "arm_upper_arm_link",
            "arm_forearm_link",
            "arm_wrist_1_link",
            "arm_wrist_2_link",
            "arm_wrist_3_link",
            "arm_tool0",
            "arm_hand_base",
        ]

        for name in candidate_names:
            try:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    self.link_body_ids.append(bid)
            except Exception:
                continue

        if not self.link_body_ids:
            for bid in range(model.nbody):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if nm is None:
                    continue
                if any(tok in nm for tok in ["arm", "wrist", "shoulder", "forearm"]):
                    self.link_body_ids.append(bid)

        if self.link_body_ids:
            print(f"✓ MPC collision checks will use {len(self.link_body_ids)} arm bodies.")
        else:
            print("⚠ No arm link bodies found for MuJoCo collision checks.")

    def set_cost_weights(self, Q_scalar, Q_terminal_scalar, R_scalar):
        self.Q = np.eye(self.n_joints) * Q_scalar
        self.Q_terminal = np.eye(self.n_joints) * Q_terminal_scalar
        self.R = np.eye(self.n_joints) * R_scalar
        self.prev_solution = None
        self._setup_optimization()

    def set_joint_limits(self, lower, upper):
        self.joint_limits = (np.array(lower), np.array(upper))
        self.prev_solution = None
        self._setup_optimization()

    def set_velocity_limit(self, max_vel):
        self.max_velocity = float(max_vel)
        self.prev_solution = None
        self._setup_optimization()

    def add_obstacle(self, position, size):
        """
        Add an obstacle for avoidance.

        Args:
            position: (3,) center [x, y, z] in base_link.
            size: (3,) half-extents [sx, sy, sz] (axis-aligned box).
        """
        self.obstacles.append((np.array(position, dtype=float),
                               np.array(size, dtype=float)))
        print(f"Added obstacle at {position} with size {size}")

    def clear_obstacles(self):
        self.obstacles = []
        print("Cleared all obstacles from MPC")

    # ------------------------------------------------------------------
    # Optimization problem
    # ------------------------------------------------------------------
    def _setup_optimization(self):
        n = self.n_joints
        H = self.horizon

        q = ca.SX.sym("q", n, H + 1)
        q_current = ca.SX.sym("q_current", n)
        q_target = ca.SX.sym("q_target", n)

        obs_pos = ca.SX.sym("obs_pos", 3, self.n_max_obstacles)
        obs_size = ca.SX.sym("obs_size", 3, self.n_max_obstacles)
        n_active_obs = ca.SX.sym("n_active_obs", 1)

        cost = 0

        # Running cost
        for k in range(H):
            q_error = q[:, k] - q_target
            cost += ca.mtimes([q_error.T, self.Q, q_error])

            if k > 0:
                q_change = q[:, k] - q[:, k - 1]
                cost += ca.mtimes([q_change.T, self.R, q_change])

        # Terminal cost
        q_error_final = q[:, H] - q_target
        cost += ca.mtimes([q_error_final.T, self.Q_terminal, q_error_final])

        # Constraints
        constraints = []
        lbg = []
        ubg = []

        # Initial condition
        constraints.append(q[:, 0] - q_current)
        lbg.extend([0.0] * n)
        ubg.extend([0.0] * n)

        # Velocity constraints
        for k in range(H):
            velocity = (q[:, k + 1] - q[:, k]) / self.dt
            for j in range(n):
                constraints.append(velocity[j])
                lbg.append(-self.max_velocity)
                ubg.append(+self.max_velocity)

        # Obstacle constraints
        if self.fk_fun is not None and self.enable_fk:
            print("  Using analytic FK for EE obstacle constraints + margin cost")
            for k in range(H + 1):
                ee_pos_k = self.fk_fun(q[:, k])  # (3,) in base_link

                for i_obs in range(self.n_max_obstacles):
                    center = obs_pos[:, i_obs]
                    half_size = obs_size[:, i_obs]
                    diff = ee_pos_k - center

                    inside_clearances = half_size - ca.fabs(diff)
                    min_inside = inside_clearances[0]
                    min_inside = ca.fmin(min_inside, inside_clearances[1])
                    min_inside = ca.fmin(min_inside, inside_clearances[2])

                    active_mask = ca.if_else(i_obs < n_active_obs, 1.0, 0.0)
                    g_obs = active_mask * min_inside
                    constraints.append(g_obs)
                    lbg.append(-1e3)
                    ubg.append(0.0)  # must not be strictly inside

                    # Soft inflated margin
                    inflated = half_size + self.safety_margin
                    u_margin = ca.fabs(diff) - inflated
                    pen_vec = ca.fmin(0, u_margin)
                    pen_mag = ca.sqrt(ca.sumsqr(pen_vec))
                    cost += active_mask * self.obstacle_weight * pen_mag ** 2
        else:
            print("  Analytic FK NOT used in MPC; no explicit EE obstacle constraints.")

        # Variable bounds (joint limits)
        lbx = []
        ubx = []
        for _ in range(H + 1):
            lbx.extend(self.joint_limits[0])
            ubx.extend(self.joint_limits[1])

        opt_variables = ca.reshape(q, -1, 1)
        opt_params = ca.vertcat(
            q_current,
            q_target,
            ca.reshape(obs_pos, -1, 1),
            ca.reshape(obs_size, -1, 1),
            n_active_obs,
        )

        nlp = {
            "x": opt_variables,
            "p": opt_params,
            "f": cost,
            "g": ca.vertcat(*constraints) if constraints else ca.SX.zeros(0, 1),
        }

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "print_time": 0,
            "ipopt.tol": 5e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.acceptable_iter": 3,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)
        self.prev_solution = None

        print(
            f"✓ MPC initialized: {self.n_joints} joints, horizon={self.horizon}, dt={self.dt}"
        )

    # ------------------------------------------------------------------
    # MPC solve
    # ------------------------------------------------------------------
    def compute_control(self, current_state, target_state,
                        model=None, data_scratch=None, site_id=None):
        """
        Compute optimal next joint command + predicted trajectory.

        Args:
            current_state: [q, dq] array of shape (2*n_joints,).
            target_state: target joint positions (n_joints,).

            model, data_scratch, site_id: optional MuJoCo objects.

        Returns:
            q_next: (n_joints,) next-step command.
            q_traj: (H+1, n_joints) full predicted trajectory.
        """
        q_current = np.asarray(current_state[: self.n_joints], dtype=float)
        q_target = np.asarray(target_state, dtype=float)

        # Warm start
        if self.prev_solution is None:
            x0 = np.zeros(self.n_joints * (self.horizon + 1), dtype=float)
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                x0[k * self.n_joints : (k + 1) * self.n_joints] = (
                    (1.0 - alpha) * q_current + alpha * q_target
                )
        else:
            x0 = np.zeros(self.n_joints * (self.horizon + 1), dtype=float)
            x0[: self.n_joints * self.horizon] = self.prev_solution[self.n_joints :]
            x0[self.n_joints * self.horizon :] = self.prev_solution[-self.n_joints :]

        # Obstacles params
        obs_pos_flat = np.zeros(3 * self.n_max_obstacles, dtype=float)
        obs_size_flat = np.zeros(3 * self.n_max_obstacles, dtype=float)
        n_active = min(len(self.obstacles), self.n_max_obstacles)

        for i, (pos, size) in enumerate(self.obstacles[: self.n_max_obstacles]):
            obs_pos_flat[i * 3 : (i + 1) * 3] = pos
            obs_size_flat[i * 3 : (i + 1) * 3] = size

        params = np.concatenate(
            [q_current, q_target, obs_pos_flat, obs_size_flat, np.array([n_active], dtype=float)]
        )

        try:
            # Optional MuJoCo heuristic
            if (
                mujoco is not None
                and model is not None
                and data_scratch is not None
                and (site_id is not None or self.link_body_ids)
                and self.obstacles
            ):
                traj_guess = x0.reshape(self.horizon + 1, self.n_joints)
                if self._trajectory_collides(traj_guess, model, data_scratch, site_id):
                    x0 = self._generate_collision_free_guess(
                        q_current, q_target, model, data_scratch, site_id
                    )

            sol = self.solver(
                x0=x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params,
            )

            x_opt = np.array(sol["x"].full().flatten(), dtype=float)
            self.prev_solution = x_opt.copy()
            q_opt = x_opt.reshape(self.horizon + 1, self.n_joints)

            q_next = q_opt[1].copy()
            return q_next, q_opt

        except Exception as e:
            print(f"MPC solve failed: {e}, using fallback joint-space step.")
            alpha = 0.05
            q_next = (1.0 - alpha) * q_current + alpha * q_target
            q_traj = np.tile(q_next, (self.horizon + 1, 1))
            return q_next, q_traj

    # ------------------------------------------------------------------
    # MuJoCo collision helpers (optional)
    # ------------------------------------------------------------------
    def _trajectory_collides(self, trajectory, model, data_scratch, site_id):
        if mujoco is None or not self.obstacles:
            return False

        H_plus_1 = trajectory.shape[0]
        step_stride = max(1, H_plus_1 // 5)

        for k in range(0, H_plus_1, step_stride):
            data_scratch.qpos[: self.n_joints] = trajectory[k]
            mujoco.mj_kinematics(model, data_scratch)

            points = []

            if site_id is not None:
                points.append(data_scratch.site_xpos[site_id].copy())

            for bid in self.link_body_ids:
                points.append(data_scratch.xpos[bid].copy())

            if not points:
                continue

            for p in points:
                for obs_pos, obs_size in self.obstacles:
                    dist = self._point_box_distance(p, obs_pos, obs_size)
                    if dist < self.safety_margin:
                        return True

        return False

    @staticmethod
    def _point_box_distance(point, box_center, box_half_size):
        diff = point - box_center
        clamped = np.clip(diff, -box_half_size, box_half_size)

        if np.allclose(diff, clamped):
            distances = box_half_size - np.abs(diff)
            return -float(np.min(distances))
        else:
            return float(np.linalg.norm(diff - clamped))

    def _generate_collision_free_guess(self, q_start, q_goal, model, data_scratch, site_id):
        H = self.horizon
        n = self.n_joints
        x0 = np.zeros(n * (H + 1), dtype=float)

        if self.obstacles:
            waypoint1 = q_start.copy()
            if n >= 3:
                waypoint1[1] -= 0.8
                waypoint1[2] -= 0.5

            waypoint2 = q_goal.copy()
            if n >= 3:
                waypoint2[1] -= 0.4

            for k in range(H + 1):
                t = k / H
                if t < 1.0 / 3.0:
                    alpha = t / (1.0 / 3.0)
                    q_k = (1.0 - alpha) * q_start + alpha * waypoint1
                elif t < 2.0 / 3.0:
                    alpha = (t - 1.0 / 3.0) / (1.0 / 3.0)
                    q_k = (1.0 - alpha) * waypoint1 + alpha * waypoint2
                else:
                    alpha = (t - 2.0 / 3.0) / (1.0 / 3.0)
                    q_k = (1.0 - alpha) * waypoint2 + alpha * q_goal

                x0[k * n : (k + 1) * n] = q_k
        else:
            for k in range(H + 1):
                alpha = k / H
                q_k = (1.0 - alpha) * q_start + alpha * q_goal
                x0[k * n : (k + 1) * n] = q_k

        return x0
