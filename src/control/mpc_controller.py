"""
Model Predictive Control (MPC) Controller
Position-space MPC for position-controlled robots
"""

import numpy as np
import casadi as ca
from pathlib import Path


class MPCController:
    """MPC controller for position-controlled robotic arm with NN-based obstacle avoidance."""

    def __init__(self, n_joints=6, horizon=30, dt=0.01, nn_fk_weights_path=None):
        """
        Initialize MPC controller.

        Args:
            n_joints: Number of robot joints
            horizon: Prediction horizon (time steps)
            dt: Time step duration (seconds)
            nn_fk_weights_path: Path to neural network FK weights (.npz file).
                               If None, defaults to "data/models/ur5e_fk_nn.npz"
        """
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt

        # Cost function weights
        self.Q = np.eye(n_joints) * 500.0      # Position error weight
        self.R = np.eye(n_joints) * 0.1        # Control smoothness weight
        self.Q_terminal = np.eye(n_joints) * 1000.0
        
        # Obstacle avoidance weight
        self.obstacle_weight = 1e4  # Penalty for being inside obstacle safety margin

        # Constraints
        self.joint_limits = (np.array([-2*np.pi]*n_joints),
                             np.array([ 2*np.pi]*n_joints))
        self.max_velocity = 5.0  # rad/s
        
        # Obstacle avoidance
        self.obstacles = []  # List of (position, size) tuples
        self.safety_margin = 0.10  # meters - minimum clearance from obstacles
        self.n_max_obstacles = 10  # Maximum number of obstacles

        # Neural network FK
        self.nn_fk_fun = None
        if nn_fk_weights_path is None:
            # Default path
            nn_fk_weights_path = Path(__file__).parent.parent.parent / "data" / "models" / "ur5e_fk_nn.npz"
        
        nn_fk_weights_path = Path(nn_fk_weights_path)
        if nn_fk_weights_path.exists():
            from src.control.nn_fk_casadi import build_nn_fk_function
            self.nn_fk_fun = build_nn_fk_function(str(nn_fk_weights_path), n_joints)
            print(f"✓ Loaded NN FK from {nn_fk_weights_path}")
        else:
            print(f"⚠ NN FK weights not found at {nn_fk_weights_path}")
            print(f"  Obstacle avoidance will use heuristic-only approach.")
            print(f"  To enable NN-based avoidance, run:")
            print(f"    1. python scripts/generate_fk_dataset.py")
            print(f"    2. python scripts/train_nn_fk.py")

        # Solver
        self.solver = None
        self.setup_optimization()

    def setup_optimization(self):
        """Set up MPC optimization for position-controlled robot with NN FK obstacle avoidance."""

        n = self.n_joints
        H = self.horizon

        # Decision variables: desired positions at each time step
        q = ca.SX.sym('q', n, H + 1)

        # Parameters
        q_current = ca.SX.sym('q_current', n)  # Current position
        q_target = ca.SX.sym('q_target', n)    # Target position
        
        # Obstacle parameters (position and size for each obstacle)
        obs_pos = ca.SX.sym('obs_pos', 3, self.n_max_obstacles)  # 3D centers
        obs_size = ca.SX.sym('obs_size', 3, self.n_max_obstacles)  # 3D half-extents
        n_active_obs = ca.SX.sym('n_active_obs', 1)  # Number of active obstacles

        # Cost function
        cost = 0

        # Running cost: position error + smoothness
        for k in range(H):
            q_error = q[:, k] - q_target
            cost += ca.mtimes([q_error.T, self.Q, q_error])

            if k > 0:
                q_change = q[:, k] - q[:, k-1]
                cost += ca.mtimes([q_change.T, self.R, q_change])

        # Terminal cost
        q_error_final = q[:, H] - q_target
        cost += ca.mtimes([q_error_final.T, self.Q_terminal, q_error_final])

        # ===================================================================
        # NEURAL NETWORK BASED OBSTACLE AVOIDANCE
        # ===================================================================
        # If NN FK is available, add obstacle avoidance penalties to the cost
        if self.nn_fk_fun is not None:
            print("  Adding NN FK obstacle avoidance to MPC cost...")
            
            for k in range(H + 1):
                # Compute EE position using neural network FK
                ee_pos_k = self.nn_fk_fun(q[:, k])  # 3x1 vector
                
                # For each obstacle, penalize penetration into safety zone
                for i in range(self.n_max_obstacles):
                    center = obs_pos[:, i]      # 3x1
                    half_size = obs_size[:, i]  # 3x1
                    
                    # Inflated box size (adds safety margin)
                    inflated = half_size + self.safety_margin
                    
                    # Distance from EE to obstacle center
                    diff = ee_pos_k - center  # 3x1
                    
                    # Compute signed distance to box surface
                    # u[i] = |diff[i]| - inflated[i]
                    # Negative means inside inflated box
                    u = ca.fabs(diff) - inflated
                    
                    # Penetration vector (only negative components matter)
                    # pen_vec[i] = min(0, u[i])
                    pen_vec = ca.fmin(0, u)
                    
                    # Penetration magnitude (L2 norm of negative parts)
                    pen_mag = ca.sqrt(ca.sumsqr(pen_vec))
                    
                    # Add quadratic penalty for penetration
                    # This creates a strong repulsive force when inside safety zone
                    cost += self.obstacle_weight * pen_mag**2
        else:
            print("  NN FK not available, using heuristic obstacle avoidance only")

        # ===================================================================
        # CONSTRAINTS
        # ===================================================================
        constraints = []
        lbg = []
        ubg = []

        # Initial condition: start from current position
        constraints.append(q[:, 0] - q_current)
        lbg.extend([0] * n)
        ubg.extend([0] * n)

        # Velocity constraints
        for k in range(H):
            velocity = (q[:, k+1] - q[:, k]) / self.dt
            for j in range(n):
                constraints.append(velocity[j])
                lbg.append(-self.max_velocity)
                ubg.append(self.max_velocity)

        # Variable bounds (position limits)
        lbx = []
        ubx = []
        for _ in range(self.horizon + 1):
            lbx.extend(self.joint_limits[0])
            ubx.extend(self.joint_limits[1])

        # Pack variables and parameters
        opt_variables = ca.reshape(q, -1, 1)
        opt_params = ca.vertcat(q_current, q_target, ca.reshape(obs_pos, -1, 1), 
                                ca.reshape(obs_size, -1, 1), n_active_obs)

        # Create NLP
        nlp = {
            'x': opt_variables,
            'p': opt_params,
            'f': cost,
            'g': ca.vertcat(*constraints) if constraints else ca.SX.zeros(0, 1)
        }

        # Solver options (tuned for speed with NN FK)
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,  # Reduced for speed
            'print_time': 0,
            'ipopt.tol': 1e-4,      # Relaxed tolerance
            'ipopt.acceptable_tol': 1e-3,  # Relaxed acceptable tolerance
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',  # Faster convergence
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        # Store for warm starting
        self.prev_solution = None

        print(f"✓ MPC initialized: {self.n_joints} joints, horizon={self.horizon}, dt={self.dt}")
        print(f"  Position weight: {self.Q[0,0]}, Terminal weight: {self.Q_terminal[0,0]}")
        if self.nn_fk_fun is not None:
            print(f"  NN-based obstacle avoidance: max {self.n_max_obstacles} obstacles")
            print(f"  Safety margin: {self.safety_margin}m, Penalty weight: {self.obstacle_weight}")
        else:
            print(f"  Heuristic obstacle avoidance: margin={self.safety_margin}m")

    def compute_control(self, current_state, target_state, model=None, data_scratch=None, site_id=None):
        """
        Compute optimal position command using MPC.

        Args:
            current_state: Current joint positions and velocities [q, dq] (12,)
            target_state: Target joint positions (6,)
            model: MuJoCo model (for FK and obstacle checking)
            data_scratch: MuJoCo data scratch buffer (for FK)
            site_id: End-effector site ID

        Returns:
            optimal_position: Target position for next timestep (6,)
            predicted_trajectory: Predicted trajectory over horizon
        """
        q_current = current_state[:self.n_joints]
        q_target = target_state

        # Initial guess
        if self.prev_solution is None:
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                x0[k*self.n_joints:(k+1)*self.n_joints] = (1-alpha)*q_current + alpha*q_target
        else:
            # Warm start: shift previous solution
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            x0[:self.n_joints*self.horizon] = self.prev_solution[self.n_joints:]
            x0[self.n_joints*self.horizon:] = self.prev_solution[-self.n_joints:]

        # Prepare obstacle parameters
        obs_pos_flat = np.zeros(3 * self.n_max_obstacles)
        obs_size_flat = np.zeros(3 * self.n_max_obstacles)
        n_active = min(len(self.obstacles), self.n_max_obstacles)
        
        for i, (pos, size) in enumerate(self.obstacles[:self.n_max_obstacles]):
            obs_pos_flat[i*3:(i+1)*3] = pos
            obs_size_flat[i*3:(i+1)*3] = size

        # Parameters
        params = np.concatenate([q_current, q_target, obs_pos_flat, obs_size_flat, [n_active]])

        # Solve with obstacle-aware constraints
        try:
            # Check if current trajectory would collide with obstacles
            if model is not None and data_scratch is not None and site_id is not None:
                # Penalize trajectories that get too close to obstacles
                collision_detected = self._check_trajectory_collision(
                    x0.reshape(self.horizon + 1, self.n_joints),
                    model, data_scratch, site_id
                )
                if collision_detected:
                    # Modify initial guess to go around obstacles
                    x0 = self._generate_collision_free_guess(q_current, q_target, model, data_scratch, site_id)
            
            sol = self.solver(
                x0=x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params
            )

            # Extract solution
            x_opt = sol['x'].full().flatten()
            self.prev_solution = x_opt.copy()  # Save for warm start

            # Extract trajectory
            q_opt = x_opt.reshape(self.horizon + 1, self.n_joints)

            # Return next desired position
            return q_opt[1], q_opt
            
        except Exception as e:
            print(f"MPC solve failed: {e}, using fallback")
            # Return a safe fallback (small step toward target)
            alpha = 0.05
            q_next = (1 - alpha) * q_current + alpha * q_target
            return q_next, np.tile(q_next, (self.horizon + 1, 1))

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

    def _check_trajectory_collision(self, trajectory, model, data_scratch, site_id):
        """
        Check if a trajectory collides with any obstacles.
        
        Args:
            trajectory: Array of joint positions (horizon+1, n_joints)
            model: MuJoCo model
            data_scratch: MuJoCo data for FK computation
            site_id: End-effector site ID
            
        Returns:
            True if collision detected, False otherwise
        """
        if not self.obstacles:
            return False
            
        # Sample every few steps in trajectory
        for k in range(0, len(trajectory), max(1, len(trajectory) // 5)):
            # Compute FK for this configuration
            data_scratch.qpos[:self.n_joints] = trajectory[k]
            import mujoco
            mujoco.mj_kinematics(model, data_scratch)
            ee_pos = data_scratch.site_xpos[site_id].copy()
            
            # Check against all obstacles
            for obs_pos, obs_size in self.obstacles:
                if self._point_box_distance(ee_pos, obs_pos, obs_size) < self.safety_margin:
                    return True
        return False

    def _point_box_distance(self, point, box_center, box_size):
        """
        Compute minimum distance from point to box surface.
        
        Args:
            point: 3D point
            box_center: 3D box center
            box_size: 3D box half-extents
            
        Returns:
            Minimum distance (negative if inside box)
        """
        # Vector from box center to point
        diff = point - box_center
        
        # Clamp to box surface
        clamped = np.clip(diff, -box_size, box_size)
        
        # Distance from point to nearest point on box surface
        if np.allclose(diff, clamped):
            # Point is inside box - return negative distance to surface
            distances = box_size - np.abs(diff)
            return -np.min(distances)
        else:
            # Point is outside - return distance to surface
            return np.linalg.norm(diff - clamped)

    def _generate_collision_free_guess(self, q_start, q_goal, model, data_scratch, site_id):
        """
        Generate initial guess that attempts to avoid obstacles.
        Uses multiple strategies to find collision-free path.
        
        Args:
            q_start: Starting joint configuration
            q_goal: Goal joint configuration
            model: MuJoCo model
            data_scratch: MuJoCo data for FK
            site_id: End-effector site ID
            
        Returns:
            Initial guess trajectory
        """
        import mujoco
        
        # Compute start and goal EE positions
        data_scratch.qpos[:self.n_joints] = q_start
        mujoco.mj_kinematics(model, data_scratch)
        start_ee = data_scratch.site_xpos[site_id].copy()
        
        data_scratch.qpos[:self.n_joints] = q_goal
        mujoco.mj_kinematics(model, data_scratch)
        goal_ee = data_scratch.site_xpos[site_id].copy()
        
        # Find obstacle center(s) to determine best avoidance strategy
        if self.obstacles:
            # Compute average obstacle position
            obs_center = np.mean([pos for pos, _ in self.obstacles], axis=0)
            
            # Determine if we should go over, under, or around
            # Check if obstacles are more vertical (wall) or horizontal (ceiling)
            max_obs_height = max(pos[2] + size[2] for pos, size in self.obstacles)
            
            # Strategy: Go high over obstacles
            waypoint = q_start.copy()
            
            # UR5e joint indices (typical):
            # 0: shoulder_pan, 1: shoulder_lift, 2: elbow, 3: wrist_1, 4: wrist_2, 5: wrist_3
            
            # Adjust shoulder and elbow to create high arc
            waypoint[1] -= 0.8  # Shoulder lift - move up (negative is up for UR5e)
            waypoint[2] -= 0.5  # Elbow - extend upward
            
            # Create two-waypoint trajectory for smoother motion
            waypoint2 = q_goal.copy()
            waypoint2[1] -= 0.4  # Partial lift on approach to goal
            
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            
            # Build trajectory: start -> waypoint1 -> waypoint2 -> goal
            for k in range(self.horizon + 1):
                t = k / self.horizon
                if t < 0.33:
                    # First third: start to waypoint1
                    alpha = t / 0.33
                    q_k = (1 - alpha) * q_start + alpha * waypoint
                elif t < 0.67:
                    # Second third: waypoint1 to waypoint2
                    alpha = (t - 0.33) / 0.34
                    q_k = (1 - alpha) * waypoint + alpha * waypoint2
                else:
                    # Final third: waypoint2 to goal
                    alpha = (t - 0.67) / 0.33
                    q_k = (1 - alpha) * waypoint2 + alpha * q_goal
                
                x0[k*self.n_joints:(k+1)*self.n_joints] = q_k
                
        else:
            # No obstacles, use direct interpolation
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                q_k = (1 - alpha) * q_start + alpha * q_goal
                x0[k*self.n_joints:(k+1)*self.n_joints] = q_k
        
        return x0
