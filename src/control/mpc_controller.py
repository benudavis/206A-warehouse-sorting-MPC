"""
Model Predictive Control (MPC) Controller
Position-space MPC for position-controlled robots, with analytic FK
and hard obstacle constraints at the end-effector.
"""

import numpy as np
import casadi as ca
import mujoco
from pathlib import Path


class MPCController:
    """MPC controller for a position-controlled robotic arm with analytic FK obstacle avoidance."""

    def __init__(
        self,
        n_joints: int = 6,
        horizon: int = 30,
        dt: float = 0.05,
        enable_fk: bool = True,
    ):
        """
        Initialize MPC controller.

        Args:
            n_joints: number of robot joints.
            horizon: prediction horizon (number of steps).
            dt: time step [s]; should match outer control loop period.
            enable_fk: whether to use the analytic FK inside the MPC.
        """
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt
        self.enable_fk = enable_fk

        # Cost weights (joint-space tracking + smoothness)
        self.Q = np.eye(n_joints) * 500.0
        self.R = np.eye(n_joints) * 0.1
        self.Q_terminal = np.eye(n_joints) * 1000.0

        # Obstacle avoidance (soft margin)
        self.obstacle_weight = 1e4  # weight on margin violation

        # Joint & velocity constraints
        self.joint_limits = (
            np.array([-2 * np.pi] * n_joints),
            np.array([+2 * np.pi] * n_joints),
        )
        self.max_velocity = 3.0  # [rad/s]

        # Obstacles: list of (center[3], half_size[3]) in world frame
        self.obstacles = []
        self.n_max_obstacles = 10
        # "Safety margin" radius: distance from obstacle faces where extra cost kicks in.
        # NOTE: this is ONLY used in the cost, not in the hard constraint.
        self.safety_margin = 0.12  # [m]

        # Bodies along the arm for MuJoCo-based geometric checks
        self.link_body_ids = []

        # Analytic FK (UR5e)
        self.fk_fun = None
        if self.enable_fk:
            try:
                from src.control.forward_kinematics import build_ur5e_fk_function
                self.fk_fun = build_ur5e_fk_function()
                print("✓ Using analytic UR5e FK (CasADi) inside MPC.")
            except Exception as e:
                print(f"⚠ Failed to build analytic UR5e FK: {e}")
                print("  MPC obstacle avoidance will fall back to MuJoCo-only heuristics.")
        else:
            print("Analytic FK disabled in MPC (enable_fk=False).")

        # CasADi solver & cached bounds
        self.solver = None
        self.prev_solution = None  # warm start
        self._setup_optimization()

    # ------------------------------------------------------------------
    # Public configuration helpers
    # ------------------------------------------------------------------

    def initialize_link_bodies(self, model):
        """
        Initialize which robot bodies (links) are used for collision checks.

        We try a set of common UR5e link names with the "arm_" prefix.
        If no matches are found, we fall back to auto-detection or EE-only checks.
        """
        if self.link_body_ids:
            return  # already initialized

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

        # Fallback: try to auto-detect any bodies containing "arm" or "wrist"
        if not self.link_body_ids:
            for bid in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if name is None:
                    continue
                if any(tok in name for tok in ["arm", "wrist", "shoulder", "forearm"]):
                    self.link_body_ids.append(bid)

        if self.link_body_ids:
            print(
                f"✓ MPC collision checks will use {len(self.link_body_ids)} arm link bodies."
            )
        else:
            print(
                "⚠ MPC collision checks: no arm link bodies found; "
                "falling back to EE-only checks."
            )

    def set_cost_weights(self, Q_scalar, Q_terminal_scalar, R_scalar):
        """Set scalar cost weights (diagonal) and rebuild solver."""
        self.Q = np.eye(self.n_joints) * Q_scalar
        self.Q_terminal = np.eye(self.n_joints) * Q_terminal_scalar
        self.R = np.eye(self.n_joints) * R_scalar
        self.prev_solution = None
        self._setup_optimization()

    def set_joint_limits(self, lower, upper):
        """Set joint position limits and rebuild solver."""
        self.joint_limits = (np.array(lower), np.array(upper))
        self.prev_solution = None
        self._setup_optimization()

    def set_velocity_limit(self, max_vel):
        """Set maximum joint velocity and rebuild solver."""
        self.max_velocity = float(max_vel)
        self.prev_solution = None
        self._setup_optimization()

    def add_obstacle(self, position, size):
        """
        Add an obstacle for avoidance.

        Args:
            position: 3D center [x, y, z].
            size: 3D half-extents [sx, sy, sz] (box).
        """
        self.obstacles.append((np.array(position, dtype=float), np.array(size, dtype=float)))
        print(f"Added obstacle at {position} with size {size}")

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = []
        print("Cleared all obstacles from MPC")

    # ------------------------------------------------------------------
    # Optimization problem
    # ------------------------------------------------------------------

    def _setup_optimization(self):
        """Set up CasADi NLP for MPC."""
        n = self.n_joints
        H = self.horizon

        # Decision variables: joint positions q_k for k=0..H
        q = ca.SX.sym("q", n, H + 1)

        # Parameters: current/target joint configurations
        q_current = ca.SX.sym("q_current", n)  # q_0 must equal this
        q_target = ca.SX.sym("q_target", n)

        # Obstacle parameters (centers + half-extents)
        obs_pos = ca.SX.sym("obs_pos", 3, self.n_max_obstacles)   # shape (3, M)
        obs_size = ca.SX.sym("obs_size", 3, self.n_max_obstacles) # shape (3, M)
        n_active_obs = ca.SX.sym("n_active_obs", 1)               # scalar

        # -----------------------------
        # Cost function
        # -----------------------------
        cost = 0

        # Running cost: tracking + smoothness
        for k in range(H):
            # Joint tracking to target
            q_error = q[:, k] - q_target
            cost += ca.mtimes([q_error.T, self.Q, q_error])

            # Smoothness (penalize changes in joint positions)
            if k > 0:
                q_change = q[:, k] - q[:, k - 1]
                cost += ca.mtimes([q_change.T, self.R, q_change])

        # Terminal cost
        q_error_final = q[:, H] - q_target
        cost += ca.mtimes([q_error_final.T, self.Q_terminal, q_error_final])

        # -----------------------------
        # Constraints
        # -----------------------------
        constraints = []
        lbg = []
        ubg = []

        # Initial condition: q_0 = q_current
        constraints.append(q[:, 0] - q_current)
        lbg.extend([0.0] * n)
        ubg.extend([0.0] * n)

        # Velocity constraints: |(q_{k+1} - q_k)/dt| <= max_velocity
        for k in range(H):
            velocity = (q[:, k + 1] - q[:, k]) / self.dt
            for j in range(n):
                constraints.append(velocity[j])
                lbg.append(-self.max_velocity)
                ubg.append(+self.max_velocity)

        # -----------------------------
        # Obstacle constraints + margin cost
        # -----------------------------
        if self.fk_fun is not None and self.enable_fk:
            print("  Using analytic FK for EE obstacle constraints + margin cost")
            for k in range(H + 1):
                # EE position from FK (3x1)
                ee_pos_k = self.fk_fun(q[:, k])

                for i_obs in range(self.n_max_obstacles):
                    center = obs_pos[:, i_obs]    # (3,)
                    half_size = obs_size[:, i_obs]  # (3,)

                    # diff: signed coord differences
                    diff = ee_pos_k - center  # (3,)

                    # -------- Hard constraint: stay OUT of actual obstacle box --------
                    # inside_clearances = half_size - |diff|
                    # > 0 => inside; ==0 => on face; < 0 => outside
                    inside_clearances = half_size - ca.fabs(diff)
                    min_inside = inside_clearances[0]
                    min_inside = ca.fmin(min_inside, inside_clearances[1])
                    min_inside = ca.fmin(min_inside, inside_clearances[2])

                    # If obstacle index i_obs >= n_active_obs, deactivate via mask
                    active_mask = ca.if_else(i_obs < n_active_obs, 1.0, 0.0)

                    # Constraint: min_inside <= 0 (cannot be strictly inside)
                    g_obs = active_mask * min_inside
                    constraints.append(g_obs)
                    lbg.append(-1e3)  # effectively no lower bound
                    ubg.append(0.0)   # upper bound 0 → not interior

                    # -------- Soft margin cost: distance to inflated box --------
                    # Safety margin region: box with half-size + safety_margin
                    inflated = half_size + self.safety_margin
                    # u_margin = |diff| - inflated:
                    #   negative inside inflated box, positive outside
                    u_margin = ca.fabs(diff) - inflated
                    pen_vec = ca.fmin(0, u_margin)  # keep only negative (inside inflated zone)
                    pen_mag = ca.sqrt(ca.sumsqr(pen_vec))

                    cost += active_mask * self.obstacle_weight * pen_mag ** 2
        else:
            print("  Analytic FK NOT used in MPC; no analytic EE constraints (MuJoCo heuristics only).")

        # -----------------------------
        # Variable bounds (joint limits)
        # -----------------------------
        lbx = []
        ubx = []
        for _ in range(H + 1):
            lbx.extend(self.joint_limits[0])
            ubx.extend(self.joint_limits[1])

        # Pack decision variables and parameters
        opt_variables = ca.reshape(q, -1, 1)
        opt_params = ca.vertcat(
            q_current,
            q_target,
            ca.reshape(obs_pos, -1, 1),
            ca.reshape(obs_size, -1, 1),
            n_active_obs,
        )

        # NLP definition
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
        if self.fk_fun is not None and self.enable_fk:
            print("  Obstacle handling:")
            print("    • Hard constraint: EE outside obstacle box")
            print("    • Soft cost: EE outside box + safety margin")
            print(f"    • Max obstacles: {self.n_max_obstacles}")
            print(f"    • Safety margin (cost only): {self.safety_margin} m")
        else:
            print("  No explicit EE obstacle constraints in the NLP (heuristics only).")

    # ------------------------------------------------------------------
    # MPC solve
    # ------------------------------------------------------------------

    def compute_control(self, current_state, target_state, model=None, data_scratch=None, site_id=None):
        """
        Compute optimal next joint command using MPC (one receding-horizon step).

        Args:
            current_state: array of shape (2*n_joints,) containing [q, dq].
            target_state: array of shape (n_joints,) containing target joint positions.
            model: MuJoCo model (optional, for heuristic collision checking).
            data_scratch: MuJoCo data (optional) used as a scratch buffer.
            site_id: end-effector site ID in the MuJoCo model (optional).

        Returns:
            q_next: optimal next joint position command (n_joints,).
            q_traj: predicted trajectory over horizon (H+1, n_joints).
        """
        q_current = np.asarray(current_state[: self.n_joints], dtype=float)
        q_target = np.asarray(target_state, dtype=float)

        # Initial guess x0 (warm-starting with previous solution when available)
        if self.prev_solution is None:
            x0 = np.zeros(self.n_joints * (self.horizon + 1), dtype=float)
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                x0[k * self.n_joints : (k + 1) * self.n_joints] = (
                    (1.0 - alpha) * q_current + alpha * q_target
                )
        else:
            x0 = np.zeros(self.n_joints * (self.horizon + 1), dtype=float)
            # Shift previous solution one step forward
            x0[: self.n_joints * self.horizon] = self.prev_solution[self.n_joints :]
            x0[self.n_joints * self.horizon :] = self.prev_solution[-self.n_joints :]

        # Prepare obstacle parameters
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
            # Optional: MuJoCo-based check to adjust initial guess if obviously colliding
            if (
                model is not None
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

            # Receding horizon: only the next step is applied
            q_next = q_opt[1].copy()
            return q_next, q_opt

        except Exception as e:
            print(f"MPC solve failed: {e}, using fallback joint-space step toward target")
            alpha = 0.05
            q_next = (1.0 - alpha) * q_current + alpha * q_target
            q_traj = np.tile(q_next, (self.horizon + 1, 1))
            return q_next, q_traj

    # ------------------------------------------------------------------
    # Collision checking helpers (MuJoCo geometric heuristic, outside MPC)
    # ------------------------------------------------------------------

    def _trajectory_collides(self, trajectory, model, data_scratch, site_id):
        """
        Check if a trajectory (sequence of joint positions) collides
        with any obstacles using MuJoCo geometry.

        Args:
            trajectory: array of shape (H+1, n_joints).
            model: MuJoCo model.
            data_scratch: MuJoCo data (scratch).
            site_id: EE site ID (may be None).

        Returns:
            True if a collision is detected along the trajectory; False otherwise.
        """
        if not self.obstacles:
            return False

        H_plus_1 = trajectory.shape[0]
        step_stride = max(1, H_plus_1 // 5)

        for k in range(0, H_plus_1, step_stride):
            data_scratch.qpos[: self.n_joints] = trajectory[k]
            mujoco.mj_kinematics(model, data_scratch)

            points = []

            # EE point
            if site_id is not None:
                points.append(data_scratch.site_xpos[site_id].copy())

            # Selected link bodies
            for bid in self.link_body_ids:
                points.append(data_scratch.xpos[bid].copy())

            if not points:
                continue

            for p in points:
                for obs_pos, obs_size in self.obstacles:
                    dist = self._point_box_distance(p, obs_pos, obs_size)
                    # Heuristic: if point is closer than safety_margin, treat as collision
                    if dist < self.safety_margin:
                        return True

        return False

    @staticmethod
    def _point_box_distance(point, box_center, box_half_size):
        """
        Compute signed distance from a point to an axis-aligned box surface.

        Args:
            point: (3,) array.
            box_center: (3,) array.
            box_half_size: (3,) half-extents.

        Returns:
            Signed distance:
                negative if point is inside the box,
                zero on the surface,
                positive outside (Euclidean distance to surface).
        """
        diff = point - box_center
        # closest point on/in box
        clamped = np.clip(diff, -box_half_size, box_half_size)

        if np.allclose(diff, clamped):
            # inside: distance is negative min clearance to a face
            distances = box_half_size - np.abs(diff)
            return -float(np.min(distances))
        else:
            # outside: Euclidean distance to closest point on box
            return float(np.linalg.norm(diff - clamped))

    def _generate_collision_free_guess(self, q_start, q_goal, model, data_scratch, site_id):
        """
        Generate a heuristic collision-free initial guess trajectory that
        lifts the arm "up" before going toward the goal.

        This is only used to initialize the NLP; it does not guarantee
        collision-free motion by itself.
        """
        H = self.horizon
        n = self.n_joints
        x0 = np.zeros(n * (H + 1), dtype=float)

        if self.obstacles:
            # Heuristic waypoints (UR5e-ish):
            # waypoint1: lift shoulder & elbow
            waypoint1 = q_start.copy()
            if n >= 3:
                waypoint1[1] -= 0.8  # shoulder lift up
                waypoint1[2] -= 0.5  # elbow up / extend

            # waypoint2: closer to goal but still lifted
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
            # Simple straight-line interpolation
            for k in range(H + 1):
                alpha = k / H
                q_k = (1.0 - alpha) * q_start + alpha * q_goal
                x0[k * n : (k + 1) * n] = q_k

        return x0
