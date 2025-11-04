"""
Model Predictive Control (MPC) Controller
Position-space MPC for position-controlled robots
"""

import numpy as np
import casadi as ca


class MPCController:
    """MPC controller for position-controlled robotic arm."""
    
    def __init__(self, n_joints=6, horizon=30, dt=0.01):
        """
        Initialize MPC controller.
        
        Args:
            n_joints: Number of robot joints
            horizon: Prediction horizon (time steps)
            dt: Time step duration (seconds)
        """
        self.n_joints = n_joints
        self.horizon = horizon
        self.dt = dt
        
        # Cost function weights (tuned for excellent performance)
        self.Q = np.eye(n_joints) * 500.0  # Position error weight (HIGH = aggressive tracking)
        self.R = np.eye(n_joints) * 0.1    # Control smoothness weight
        self.Q_terminal = np.eye(n_joints) * 1000.0  # Terminal cost (reach target)
        
        # Constraints
        self.joint_limits = (np.array([-2*np.pi]*n_joints), 
                            np.array([2*np.pi]*n_joints))
        self.max_velocity = 5.0  # rad/s (fast but safe robot velocity)
        
        # Solver
        self.solver = None
        self.setup_optimization()
        
    def setup_optimization(self):
        """
        Set up MPC optimization for position-controlled robot.
        
        For position servos, we command desired positions.
        Cost: smooth trajectory that reaches target.
        """
        # Decision variables: desired positions at each time step
        q = ca.SX.sym('q', self.n_joints, self.horizon + 1)
        
        # Parameters
        q_current = ca.SX.sym('q_current', self.n_joints)  # Current position
        q_target = ca.SX.sym('q_target', self.n_joints)    # Target position
        
        # Cost function
        cost = 0
        
        # Running cost: position error + smoothness
        for k in range(self.horizon):
            # Position error from target
            q_error = q[:, k] - q_target
            cost += ca.mtimes([q_error.T, self.Q, q_error])
            
            # Smoothness: penalize large position changes
            if k > 0:
                q_change = q[:, k] - q[:, k-1]
                cost += ca.mtimes([q_change.T, self.R, q_change])
        
        # Terminal cost: strongly penalize final error
        q_error_final = q[:, self.horizon] - q_target
        cost += ca.mtimes([q_error_final.T, self.Q_terminal, q_error_final])
        
        # Constraints
        constraints = []
        lbg = []
        ubg = []
        
        # Initial condition: start from current position
        constraints.append(q[:, 0] - q_current)
        lbg.extend([0] * self.n_joints)
        ubg.extend([0] * self.n_joints)
        
        # Velocity constraints: limit how fast positions can change
        for k in range(self.horizon):
            velocity = (q[:, k+1] - q[:, k]) / self.dt
            for j in range(self.n_joints):
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
        opt_params = ca.vertcat(q_current, q_target)
        
        # Create NLP
        nlp = {
            'x': opt_variables,
            'p': opt_params,
            'f': cost,
            'g': ca.vertcat(*constraints)
        }
        
        # Solver options - aggressive for fast convergence
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 200,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.warm_start_init_point': 'yes',
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
        
        # Store for warm starting
        self.prev_solution = None
        
        print(f"MPC initialized: {self.n_joints} joints, horizon={self.horizon}, dt={self.dt}")
        print(f"  Position weight: {self.Q[0,0]}, Terminal weight: {self.Q_terminal[0,0]}")
        
    def compute_control(self, current_state, target_state):
        """
        Compute optimal position command using MPC.
        
        Args:
            current_state: Current joint positions and velocities [q, dq] (12,)
            target_state: Target joint positions (6,)
            
        Returns:
            optimal_position: Target position for next timestep (6,)
            predicted_trajectory: Predicted trajectory over horizon
        """
        q_current = current_state[:self.n_joints]
        q_target = target_state
        
        # Initial guess: linear interpolation from current to target
        if self.prev_solution is None:
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            for k in range(self.horizon + 1):
                alpha = k / self.horizon
                x0[k*self.n_joints:(k+1)*self.n_joints] = (1-alpha)*q_current + alpha*q_target
        else:
            # Warm start: shift previous solution
            x0 = np.zeros(self.n_joints * (self.horizon + 1))
            # Shift old solution by one timestep
            x0[:self.n_joints*self.horizon] = self.prev_solution[self.n_joints:]
            # Extend with last value
            x0[self.n_joints*self.horizon:] = self.prev_solution[-self.n_joints:]
        
        # Parameters
        params = np.concatenate([q_current, q_target])
        
        # Solve
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
    
    def set_cost_weights(self, Q, Q_terminal, R):
        """
        Set cost function weights and rebuild solver.
        
        Args:
            Q: Position error weight (higher = more aggressive)
            Q_terminal: Terminal cost weight
            R: Smoothness weight (higher = smoother motion)
        """
        self.Q = np.eye(self.n_joints) * Q
        self.Q_terminal = np.eye(self.n_joints) * Q_terminal
        self.R = np.eye(self.n_joints) * R
        self.prev_solution = None  # Reset warm start
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
