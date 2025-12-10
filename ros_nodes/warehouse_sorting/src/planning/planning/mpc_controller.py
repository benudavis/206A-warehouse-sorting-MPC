#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import casadi as ca


# ----------------------------------------------------------------------
# Analytic FK: q (6,) -> EE position (3,) in UR7e base_link frame
# Uses Product of Exponentials (POE) method matching forward_kinematics.py
# ----------------------------------------------------------------------
def build_ur7e_fk_function():
    """
    Build a CasADi function fk(q) -> p_ee (3,) in UR7e base_link frame.
    Uses Product of Exponentials method with parameters from forward_kinematics.py.

    Conventions:
        - q is a 6-element vector [q1..q6] in radians.
          (shoulder pan, shoulder lift, elbow, wrist1, wrist2, wrist3)
        - EE position is at wrist_3_link origin (tool frame).
        - Output is in UR7e base frame (for ROS: base_link).
    """
    # Joint vector
    q = ca.SX.sym("q", 6)

    # ------------------------------------------------------------------
    # UR7e kinematic parameters (matching forward_kinematics.py)
    # ------------------------------------------------------------------
    # Points on each joint axis in the zero config
    q0 = ca.SX(3, 6)
    q0[:, 0] = ca.SX([0.,     0.,      0.1625])   # shoulder pan
    q0[:, 1] = ca.SX([0.,     0.,      0.1625])   # shoulder lift
    q0[:, 2] = ca.SX([0.425,  0.,      0.1625])   # elbow
    q0[:, 3] = ca.SX([0.817,  0.1333,  0.1625])   # wrist 1
    q0[:, 4] = ca.SX([0.817,  0.1333,  0.06285])  # wrist 2
    q0[:, 5] = ca.SX([0.817,  0.233,   0.06285])  # wrist 3 (tool frame origin)

    # Axis vectors of each joint axis in the zero config
    w0 = ca.SX(3, 6)
    w0[:, 0] = ca.SX([0.,  0.,  1.])    # shoulder pan
    w0[:, 1] = ca.SX([0.,  1.,  0.])    # shoulder lift
    w0[:, 2] = ca.SX([0.,  1.,  0.])    # elbow
    w0[:, 3] = ca.SX([0.,  1.,  0.])    # wrist 1
    w0[:, 4] = ca.SX([0.,  0., -1.])    # wrist 2 
    w0[:, 5] = ca.SX([0.,  1.,  0.])    # wrist 3

    # Rotation matrix from base_link to wrist_3_link in zero config
    R_zero = ca.SX(3, 3)
    R_zero[0, :] = ca.SX([-1.,  0.,  0.])
    R_zero[1, :] = ca.SX([ 0.,  0.,  1.])
    R_zero[2, :] = ca.SX([ 0.,  1.,  0.])

    # Build twists ξ_i = [v_i; ω_i] with v_i = -ω_i × q_i
    xi = ca.SX(6, 6)
    for i in range(6):
        omega = w0[:, i]
        q_point = q0[:, i]
        # v = -omega × q (cross product: a × b = [a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z, a_x*b_y - a_y*b_x])
        omega_neg = -omega
        v = ca.vertcat(
            omega_neg[1] * q_point[2] - omega_neg[2] * q_point[1],
            omega_neg[2] * q_point[0] - omega_neg[0] * q_point[2],
            omega_neg[0] * q_point[1] - omega_neg[1] * q_point[0]
        )
        xi[0:3, i] = v
        xi[3:6, i] = omega

    # Zero-configuration transform gst(0)
    gst0 = ca.SX.eye(4)
    gst0[0:3, 0:3] = R_zero
    gst0[0:3, 3] = q0[:, 5]  # wrist_3_link origin

    # ------------------------------------------------------------------
    # Helper: Compute exp(ξ_hat * theta) for a single twist
    # ------------------------------------------------------------------
    def exp_twist(xi_vec: ca.SX, theta: ca.SX) -> ca.SX:
        """
        Compute exp(ξ_hat * theta) using product of exponentials formula.
        Returns 4x4 homogeneous transformation matrix.
        All joints are revolute, so omega_norm = 1.
        """
        v = xi_vec[0:3]
        omega = xi_vec[3:6]
        
        # Skew-symmetric matrix for omega
        omega_hat = ca.SX(3, 3)
        omega_hat[0, 0] = 0
        omega_hat[0, 1] = -omega[2]
        omega_hat[0, 2] = omega[1]
        omega_hat[1, 0] = omega[2]
        omega_hat[1, 1] = 0
        omega_hat[1, 2] = -omega[0]
        omega_hat[2, 0] = -omega[1]
        omega_hat[2, 1] = omega[0]
        omega_hat[2, 2] = 0

        # For revolute joints, omega is unit vector, so omega_norm = 1
        # Rotation matrix using Rodrigues' formula: R = I + sin(θ)*ω_hat + (1-cos(θ))*ω_hat^2
        I3 = ca.SX_eye(3)
        R = I3 + ca.sin(theta) * omega_hat + (1 - ca.cos(theta)) * (omega_hat @ omega_hat)
        
        # Translation vector: p = V * v where V = I*θ + (1-cos(θ))*ω_hat + (θ-sin(θ))*ω_hat^2
        V = I3 * theta + (1 - ca.cos(theta)) * omega_hat + (theta - ca.sin(theta)) * (omega_hat @ omega_hat)
        p = V @ v
        
        # Build 4x4 homogeneous transformation
        g = ca.SX.eye(4)
        g[0:3, 0:3] = R
        g[0:3, 3] = p
        
        return g

    # ------------------------------------------------------------------
    # Product of exponentials: g(θ) = exp(ξ_1*θ_1) ... exp(ξ_6*θ_6) * g(0)
    # ------------------------------------------------------------------
    g_theta = ca.SX.eye(4)
    for i in range(6):
        g_theta = g_theta @ exp_twist(xi[:, i], q[i])
    
    # Multiply by zero-configuration transform
    g_theta = g_theta @ gst0

    # Extract position (wrist_3_link origin)
    p_ee = g_theta[0:3, 3]

    fk_fun = ca.Function("fk_ur7e_base", [q], [p_ee])
    return fk_fun


# ----------------------------------------------------------------------
# MPC controller with EE obstacle avoidance using analytic FK
# ----------------------------------------------------------------------
class MPCController:
    """
    MPC controller for a position-controlled arm with mandatory end-effector collision avoidance.
    
    Uses analytic forward kinematics (Product of Exponentials) to compute end-effector positions
    and enforce collision avoidance constraints. FK is always enabled and required.
    """

    def __init__(
        self,
        n_joints: int = 6,
        horizon: int = 6,  # Reduced default for faster computation
        dt: float = 0.15,  # Increased default for faster execution
    ):
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt

        # Cost weights (joint-space tracking + smoothness + velocity)
        self.Q = np.eye(n_joints) * 500.0
        self.R = np.eye(n_joints) * 0.05  # Reduced smoothness penalty for faster motion
        self.Q_terminal = np.eye(n_joints) * 1000.0
        self.Q_v = np.eye(n_joints) * 1.0  # Velocity cost weight - much reduced to allow fast motions

        # Obstacle avoidance (soft margin)
        self.obstacle_weight = 1e4

        # Joint & velocity constraints
        self.joint_limits = (
            np.array([-2 * np.pi] * n_joints),
            np.array([+2 * np.pi] * n_joints),
        )
        self.max_velocity = 8.0  # [rad/s] - increased for much faster motion

        # Obstacles: list of (center[3], half_size[3]) in base_link frame
        self.obstacles = []
        self.n_max_obstacles = 10
        self.safety_margin = 0.12  # [m]

        # Analytic FK - always required for EE collision avoidance
        try:
            self.fk_fun = build_ur7e_fk_function()
            print("✓ Using analytic UR7e FK (Product of Exponentials, base_link frame) for EE collision avoidance.")
        except Exception as e:
            raise RuntimeError(f"Failed to build analytic UR7e FK: {e}. FK is required for MPC EE collision avoidance.")

        self.solver = None
        self.prev_solution = None
        self._setup_optimization()

    # ------------------------------------------------------------------
    # Public configuration helpers
    # ------------------------------------------------------------------
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
        dq_current = ca.SX.sym("dq_current", n)  # Current velocity
        q_target = ca.SX.sym("q_target", n)

        obs_pos = ca.SX.sym("obs_pos", 3, self.n_max_obstacles)
        obs_size = ca.SX.sym("obs_size", 3, self.n_max_obstacles)
        n_active_obs = ca.SX.sym("n_active_obs", 1)

        cost = 0

        # Running cost
        for k in range(H):
            # Position tracking cost
            q_error = q[:, k] - q_target
            cost += ca.mtimes([q_error.T, self.Q, q_error])

            # Position change cost (smoothness)
            if k > 0:
                q_change = q[:, k] - q[:, k - 1]
                cost += ca.mtimes([q_change.T, self.R, q_change])
            
            # Velocity cost (penalize high velocities for smoothness)
            velocity_k = (q[:, k + 1] - q[:, k]) / self.dt
            cost += ca.mtimes([velocity_k.T, self.Q_v, velocity_k])

        # Terminal cost
        q_error_final = q[:, H] - q_target
        cost += ca.mtimes([q_error_final.T, self.Q_terminal, q_error_final])

        # Constraints
        constraints = []
        lbg = []
        ubg = []

        # Initial condition: position
        constraints.append(q[:, 0] - q_current)
        lbg.extend([0.0] * n)
        ubg.extend([0.0] * n)
        
        # Initial velocity constraint (soft - allows some deviation for smoother transitions)
        # This encourages the first step to match current velocity but allows flexibility
        velocity_0 = (q[:, 1] - q[:, 0]) / self.dt
        velocity_error = velocity_0 - dq_current
        # Add as soft cost instead of hard constraint for more flexibility
        cost += 0.1 * ca.sumsqr(velocity_error)  # Small weight to encourage matching current velocity

        # Velocity constraints
        for k in range(H):
            velocity = (q[:, k + 1] - q[:, k]) / self.dt
            for j in range(n):
                constraints.append(velocity[j])
                lbg.append(-self.max_velocity)
                ubg.append(+self.max_velocity)

        # Obstacle constraints - always use FK for EE collision avoidance
        for k in range(H + 1):
            ee_pos_k = self.fk_fun(q[:, k])  # (3,) in base_link

            for i_obs in range(self.n_max_obstacles):
                center = obs_pos[:, i_obs]
                half_size = obs_size[:, i_obs]
                diff = ee_pos_k - center

                # Hard constraint: EE must not be inside obstacle
                inside_clearances = half_size - ca.fabs(diff)
                min_inside = inside_clearances[0]
                min_inside = ca.fmin(min_inside, inside_clearances[1])
                min_inside = ca.fmin(min_inside, inside_clearances[2])

                active_mask = ca.if_else(i_obs < n_active_obs, 1.0, 0.0)
                g_obs = active_mask * min_inside
                constraints.append(g_obs)
                lbg.append(-1e3)
                ubg.append(0.0)  # must not be strictly inside

                # Soft cost: penalize getting too close to obstacles (within safety margin)
                inflated = half_size + self.safety_margin
                u_margin = ca.fabs(diff) - inflated
                pen_vec = ca.fmin(0, u_margin)
                pen_mag = ca.sqrt(ca.sumsqr(pen_vec))
                cost += active_mask * self.obstacle_weight * pen_mag ** 2

        # Variable bounds (joint limits)
        lbx = []
        ubx = []
        for _ in range(H + 1):
            lbx.extend(self.joint_limits[0])
            ubx.extend(self.joint_limits[1])

        opt_variables = ca.reshape(q, -1, 1)
        opt_params = ca.vertcat(
            q_current,
            dq_current,
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
            "ipopt.max_iter": 15,  # Reduced from 30 for faster solves
            "print_time": 0,
            "ipopt.tol": 1e-2,  # Relaxed from 5e-3 for faster convergence
            "ipopt.acceptable_tol": 2e-2,  # Relaxed from 1e-2
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.acceptable_iter": 1,  # Accept solution after just 1 acceptable iteration
            "ipopt.fast_step_computation": "yes",  # Enable fast step computation
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
    def compute_control(self, current_state, target_state):
        """
        Compute optimal next joint command + predicted trajectory.

        Args:
            current_state: [q, dq] array of shape (2*n_joints,).
            target_state: target joint positions (n_joints,).

        Returns:
            q_next: (n_joints,) next-step command.
            q_traj: (H+1, n_joints) full predicted trajectory.
        """
        q_current = np.asarray(current_state[: self.n_joints], dtype=float)
        dq_current = np.asarray(current_state[self.n_joints:], dtype=float)
        q_target = np.asarray(target_state, dtype=float)

        # Warm start: use current velocity to extrapolate motion
        if self.prev_solution is None:
            x0 = np.zeros(self.n_joints * (self.horizon + 1), dtype=float)
            for k in range(self.horizon + 1):
                # Linear interpolation with velocity-aware extrapolation for first few steps
                alpha = k / self.horizon
                if k == 0:
                    # First step: current position
                    x0[k * self.n_joints : (k + 1) * self.n_joints] = q_current
                elif k == 1:
                    # Second step: extrapolate using current velocity
                    x0[k * self.n_joints : (k + 1) * self.n_joints] = (
                        q_current + dq_current * self.dt
                    )
                else:
                    # Remaining steps: interpolate toward target
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
            [q_current, dq_current, q_target, obs_pos_flat, obs_size_flat, np.array([n_active], dtype=float)]
        )

        try:
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
