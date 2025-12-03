"""
Model Predictive Control (MPC) Controller
Position-space MPC for position-controlled robots
"""

import numpy as np
import casadi as ca
import mujoco
from pathlib import Path


class MPCController:
    """MPC controller for position-controlled robotic arm with optional NN-based obstacle avoidance."""

    def __init__(
        self,
        n_joints=6,
        horizon=30,
        dt=0.05,
        nn_fk_weights_path=None,
        enable_nn_fk=True,
    ):
        """
        Initialize MPC controller.

        Args:
            n_joints: Number of robot joints
            horizon: Prediction horizon (time steps)
            dt: Time step duration (seconds) - should match outer control loop period
            nn_fk_weights_path: Path to neural network FK weights (.npz file).
                                If None, defaults to "data/models/ur5e_fk_nn.npz"
            enable_nn_fk: Whether to use the NN FK inside the MPC cost for obstacle avoidance.
                          If False, only heuristic / MuJoCo-based obstacle checking is used.
        """
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt
        self.enable_nn_fk = enable_nn_fk

        # Cost function weights
        self.Q = np.eye(n_joints) * 500.0      # Position error weight
        self.R = np.eye(n_joints) * 0.1        # Control smoothness weight
        self.Q_terminal = np.eye(n_joints) * 1000.0

        # Obstacle avoidance weight (for soft penalties)
        self.obstacle_weight = 1e4

        # Constraints
        self.joint_limits = (np.array([-2 * np.pi] * n_joints),
                             np.array([ 2 * np.pi] * n_joints))
        self.max_velocity = 3.0  # rad/s

        # Obstacle representation
        self.obstacles = []  # list of (position, size) tuples
        self.safety_margin = 0.12  # meters - conservative clearance
        self.n_max_obstacles = 10

        # Bodies along the arm to check for collision (initialized later)
        self.link_body_ids = []

        # Neural network FK
        self.nn_fk_fun = None
        if nn_fk_weights_path is None:
            nn_fk_weights_path = Path(__file__).parent.parent.parent / "data" / "models" / "ur5e_fk_nn.npz"

        nn_fk_weights_path = Path(nn_fk_weights_path)
        if nn_fk_weights_path.exists() and self.enable_nn_fk:
            from src.control.nn_fk_casadi import build_nn_fk_function
            self.nn_fk_fun = build_nn_fk_function(str(nn_fk_weights_path), n_joints)
            print(f"✓ Loaded NN FK from {nn_fk_weights_path} (enabled in MPC cost)")
        elif nn_fk_weights_path.exists():
            print(f"✓ NN FK weights found at {nn_fk_weights_path}, "
                  f"but NN-based MPC cost is disabled (enable_nn_fk=False).")
        else:
            print(f"⚠ NN FK weights not found at {nn_fk_weights_path}")
            print(f"  Obstacle avoidance will use heuristic / MuJoCo-based checking only.")
            print(f"  To enable NN-based avoidance, run:")
            print(f"    1. python scripts/generate_fk_dataset.py")
            print(f"    2. python scripts/train_nn_fk.py")

        # Solver
        self.solver = None
        self.prev_solution = None
        self.setup_optimization()

    # ------------------------------------------------------------------
    # Public configuration helpers
    # ------------------------------------------------------------------

    def initialize_link_bodies(self, model):
        """
        Initialize which robot bodies (links) are used for collision checks.

        We try a set of common UR5e link names with the "arm_" prefix that is
        added in the demo when attaching the robot.

        If no matches are found, collision checks fall back to EE-only.
        """
        if self.link_body_ids:
            return  # already initialized

        candidate_names = [
            # Common UR5e link names with "arm_" prefix
            "arm_base_link",
            "arm_shoulder_link",
            "arm_upper_arm_link",
            "arm_forearm_link",
            "arm_wrist_1_link",
            "arm_wrist_2_link",
            "arm_wrist_3_link",
            # Some gripper / tool bodies that might exist
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

        if self.link_body_ids:
            print(f"✓ MPC collision checks will use {len(self.link_body_ids)} arm link bodies.")
        else:
            print("⚠ MPC collision checks: no arm link bodies found by name; "
                  "falling back to EE-only distance checks.")

    def set_cost_weights(self, Q, Q_terminal, R):
        """Set cost function weights and rebuild solver."""
        self.Q = np.eye(self.n_joints) * Q
        self.Q_terminal = np.eye(self.n_joints) * Q_terminal
        self.R = np.eye(self.n_joints) * R
        self.prev_solution = None
        self.setup_optimization()

    def set_joint_limits(self, lower, upper):
        """Set joint position limits and rebuild solver."""
        self.joint_limits = (np.array(lower), np.array(upper))
        self.prev_solution = None
        self.setup_optimization()

    def set_velocity_limit(self, max_vel):
        """Set maximum velocity and rebuild solver."""
        self.max_velocity = max_vel
        self.prev_solution = None
        self.setup_optimization()

    def add_obstacle(self, position, size):
        """
        Add an obstacle for avoidance.

        Args:
            position: 3D position [x, y, z] of obstacle center
            size: 3D size [sx, sy, sz] of obstacle (half-extents for box)
        """
        self.obstacles.append((np.array(position), np.array(size)))
        print(f"Added obstacle at {position} with size {size}")

    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles = []
        print("Cleared all obstacles")

    # ------------------------------------------------------------------
    # Optimization problem
    # ------------------------------------------------------------------

    def setup_optimization(self):
        """Set up MPC optimization for position-controlled robot."""
        n = self.n_joints
        H = self.horizon

        # Decision variables: joint positions over the horizon
        q = ca.SX.sym('q', n, H + 1)

        # Parameters
        q_current = ca.SX.sym('q_current', n)  # current joint position
        q_target = ca.SX.sym('q_target', n)    # target joint position

        # Obstacle parameters
        obs_pos = ca.SX.sym('obs_pos', 3, self.n_max_obstacles)   # centers
        obs_size = ca.SX.sym('obs_size', 3, self.n_max_obstacles) # half-extents
        n_active_obs = ca.SX.sym('n_active_obs', 1)               # number of obstacles

        # Cost function
        cost = 0

        # Running cost: tracking + smoothness
        for k in range(H):
            q_error = q[:, k] - q_target
            cost += ca.mtimes([q_error.T, self.Q, q_error])

            if k > 0:
                q_change = q[:, k] - q[:, k - 1]
                cost += ca.mtimes([q_change.T, self.R, q_change])

        # Terminal cost
        q_error_final = q[:, H] - q_target
        cost += ca.mtimes([q_error_final.T, self.Q_terminal, q_error_final])

        # NN FK-based obstacle penalty (optional)
        if self.nn_fk_fun is not None and self.enable_nn_fk:
            print("  Adding NN FK obstacle avoidance to MPC cost...")
            for k in range(H + 1):
                ee_pos_k = self.nn_fk_fun(q[:, k])  # 3x1

                for i in range(self.n_max_obstacles):
                    center = obs_pos[:, i]
                    half_size = obs_size[:, i]

                    inflated = half_size + self.safety_margin
                    diff = ee_pos_k - center
                    u = ca.fabs(diff) - inflated       # negative inside inflated box
                    pen_vec = ca.fmin(0, u)            # only negative parts
                    pen_mag = ca.sqrt(ca.sumsqr(pen_vec))

                    active_mask = ca.if_else(i < n_active_obs, 1.0, 0.0)
                    cost += active_mask * self.obstacle_weight * pen_mag ** 2
        else:
            print("  NN FK is not used in MPC cost; heuristic / MuJoCo checks only.")

        # Constraints
        constraints = []
        lbg = []
        ubg = []

        # Initial condition: q[0] = q_current
        constraints.append(q[:, 0] - q_current)
        lbg.extend([0] * n)
        ubg.extend([0] * n)

        # Velocity constraints
        for k in range(H):
            velocity = (q[:, k + 1] - q[:, k]) / self.dt
            for j in range(n):
                constraints.append(velocity[j])
                lbg.append(-self.max_velocity)
                ubg.append(self.max_velocity)

        # Variable bounds (joint limits)
        lbx = []
        ubx = []
        for _ in range(H + 1):
            lbx.extend(self.joint_limits[0])
            ubx.extend(self.joint_limits[1])

        # Pack variables and parameters
        opt_variables = ca.reshape(q, -1, 1)
        opt_params = ca.vertcat(
            q_current,
            q_target,
            ca.reshape(obs_pos, -1, 1),
            ca.reshape(obs_size, -1, 1),
            n_active_obs,
        )

        # Create NLP
        nlp = {
            'x': opt_variables,
            'p': opt_params,
            'f': cost,
            'g': ca.vertcat(*constraints) if constraints else ca.SX.zeros(0, 1),
        }

        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'print_time': 0,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        self.prev_solution = None

        print(f"✓ MPC initialized: {self.n_joints} joints, horizon={self.horizon}, dt={self.dt}")
        if self.nn_fk_fun is not None and self.enable_nn_fk:
            print(f"  NN-based obstacle avoidance: max {self.n_max_obstacles} obstacles")
            print(f"  Safety margin: {self.safety_margin}m, Penalty weight: {self.obstacle_weight}")
        else:
            print(f"  Heuristic / MuJoCo obstacle avoidance: margin={self.safety_margin}m")

    # ------------------------------------------------------------------
    # MPC solve
    # ------------------------------------------------------------------

    def compute_control(self, current_state, target_state, model=None, data_scratch=None, site_id=None):
        """
        Compute optimal position command using MPC.

        Args:
            current_state: Current [q, dq] (2 * n_joints,)
            target_state: Target joint positions (n_joints,)
            model: MuJoCo model (for FK / collision checking)
            data_scratch: MuJoCo data scratch buffer (for FK)
            site_id: End-effector site ID (for EE collision checking)

        Returns:
            q_next: Target position for next timestep (n_joints,)
            q_traj: Predicted trajectory over horizon (H+1, n_joints)
        """
        q_current = current_state[:self.n_joints]
        q_target = target_state

        # Initial guess
        if self.prev_solution is None:
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                x0[k * self.n_joints:(k + 1) * self.n_joints] = (1 - alpha) * q_current + alpha * q_target
        else:
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            x0[:self.n_joints * self.horizon] = self.prev_solution[self.n_joints:]
            x0[self.n_joints * self.horizon:] = self.prev_solution[-self.n_joints:]

        # Prepare obstacle parameters
        obs_pos_flat = np.zeros(3 * self.n_max_obstacles)
        obs_size_flat = np.zeros(3 * self.n_max_obstacles)
        n_active = min(len(self.obstacles), self.n_max_obstacles)

        for i, (pos, size) in enumerate(self.obstacles[:self.n_max_obstacles]):
            obs_pos_flat[i * 3:(i + 1) * 3] = pos
            obs_size_flat[i * 3:(i + 1) * 3] = size

        params = np.concatenate([q_current, q_target, obs_pos_flat, obs_size_flat, [n_active]])

        try:
            # Use MuJoCo-based geometric check to adjust initial guess if needed
            if (
                model is not None
                and data_scratch is not None
                and (site_id is not None or self.link_body_ids)
            ):
                collision_detected = self._check_trajectory_collision(
                    x0.reshape(self.horizon + 1, self.n_joints),
                    model,
                    data_scratch,
                    site_id,
                )
                if collision_detected:
                    x0 = self._generate_collision_free_guess(q_current, q_target, model, data_scratch, site_id)

            sol = self.solver(
                x0=x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params,
            )

            x_opt = sol['x'].full().flatten()
            self.prev_solution = x_opt.copy()

            q_opt = x_opt.reshape(self.horizon + 1, self.n_joints)
            return q_opt[1], q_opt

        except Exception as e:
            print(f"MPC solve failed: {e}, using fallback step toward target")
            alpha = 0.05
            q_next = (1 - alpha) * q_current + alpha * q_target
            return q_next, np.tile(q_next, (self.horizon + 1, 1))

    # ------------------------------------------------------------------
    # Collision checking helpers
    # ------------------------------------------------------------------

    def _check_trajectory_collision(self, trajectory, model, data_scratch, site_id):
        """
        Check if a trajectory collides with any obstacles.

        Args:
            trajectory: (H+1, n_joints) array of joint angles
            model: MuJoCo model
            data_scratch: MuJoCo data for FK
            site_id: EE site ID (may be None)

        Returns:
            True if collision detected, False otherwise.
        """
        if not self.obstacles:
            return False

        step_stride = max(1, len(trajectory) // 5)
        for k in range(0, len(trajectory), step_stride):
            data_scratch.qpos[:self.n_joints] = trajectory[k]
            mujoco.mj_kinematics(model, data_scratch)

            # Points to check: EE + selected arm link bodies
            points = []

            if site_id is not None:
                points.append(data_scratch.site_xpos[site_id].copy())

            for bid in self.link_body_ids:
                points.append(data_scratch.xpos[bid].copy())

            if not points and site_id is None:
                # No points defined, cannot check
                continue

            for p in points:
                for obs_pos, obs_size in self.obstacles:
                    dist = self._point_box_distance(p, obs_pos, obs_size)
                    if dist < self.safety_margin:
                        return True

        return False

    @staticmethod
    def _point_box_distance(point, box_center, box_size):
        """
        Compute minimum distance from point to axis-aligned box surface.

        Args:
            point: 3D point
            box_center: 3D center
            box_size: 3D half-extents

        Returns:
            Signed distance: negative if inside box, positive outside.
        """
        diff = point - box_center
        clamped = np.clip(diff, -box_size, box_size)

        if np.allclose(diff, clamped):
            # Inside box: negative distance to nearest face
            distances = box_size - np.abs(diff)
            return -np.min(distances)
        else:
            # Outside: distance to nearest point on surface
            return np.linalg.norm(diff - clamped)

    def _generate_collision_free_guess(self, q_start, q_goal, model, data_scratch, site_id):
        """
        Generate initial guess trajectory that attempts to avoid obstacles
        by moving high over the workspace and then to the goal.

        Args:
            q_start: starting joint configuration
            q_goal: goal joint configuration
            model, data_scratch, site_id: MuJoCo structures (not heavily used here)

        Returns:
            x0: flattened trajectory (n_joints * (H+1),)
        """
        H = self.horizon
        n = self.n_joints

        if self.obstacles:
            # "Go high" strategy in joint space (UR5e-ish)
            waypoint1 = q_start.copy()
            waypoint1[1] -= 0.8  # shoulder lift up
            waypoint1[2] -= 0.5  # elbow up / extend

            waypoint2 = q_goal.copy()
            waypoint2[1] -= 0.4  # still somewhat lifted near goal

            x0 = np.zeros(n * (H + 1))
            for k in range(H + 1):
                t = k / H
                if t < 0.33:
                    alpha = t / 0.33
                    q_k = (1 - alpha) * q_start + alpha * waypoint1
                elif t < 0.67:
                    alpha = (t - 0.33) / 0.34
                    q_k = (1 - alpha) * waypoint1 + alpha * waypoint2
                else:
                    alpha = (t - 0.67) / 0.33
                    q_k = (1 - alpha) * waypoint2 + alpha * q_goal
                x0[k * n:(k + 1) * n] = q_k
        else:
            # Simple linear interpolation
            x0 = np.zeros(n * (H + 1))
            for k in range(H + 1):
                alpha = k / H
                q_k = (1 - alpha) * q_start + alpha * q_goal
                x0[k * n:(k + 1) * n] = q_k

        return x0
