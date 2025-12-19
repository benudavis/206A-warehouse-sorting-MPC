#!/usr/bin/env python3
"""
mpc_controller.py

UR7e NMPC in joint space with:
  - dynamics: q_{k+1} = q_k + dq_k
  - cost: position tracking + (optional) orientation tracking + dq regularization
  - constraints:
      * joint limits (bounds)
      * step limits (dq bounds)
      * obstacle avoidance vs AABB: HARD constraint (NO SLACK)
      * table safety: soft constraint (slack) by default (to avoid infeasibility)
      * facing down: enforced ONLY AT TERMINAL (k = N) as a SOFT constraint (slack)

Key user requirements implemented:
  (1) Obstacle avoidance is a constraint that should never be violated -> HARD inequality d(point,aabb) >= radius.
  (2) Facing down only required at desired location -> constraints only at k=N (terminal), not along horizon.

Returns:
  dq0, q_next, q_traj, dq_traj,
  s_face (terminal slack scalar), s_tab,
  success, solver_status, ipopt_stats
"""

import casadi as ca
import numpy as np


def quat_to_rotmat_xyzw(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix (numpy)."""
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0 - (yy + zz),       xy - wz,       xz + wy],
        [      xy + wz, 1.0 - (xx + zz),       yz - wx],
        [      xz - wy,       yz + wx, 1.0 - (xx + yy)],
    ], dtype=float)


class MPCController:
    def __init__(
        self,
        N=20,
        dt=0.1,
        q_min=None,
        q_max=None,
        dq_min=None,
        dq_max=None,
        cube_side=0.05,
        safety_margin=0.01,
        sphere_offsets_ee=None,
        sphere_radii=None,
        use_lowest_sphere_for_table=True,
        table_margin=0.005,
        # weights
        w_pos=50.0,
        w_rot=0.0,                # NOTE: keep 0 unless you truly want orientation tracking along path
        w_dq=0.5,
        w_terminal_mult=10.0,
        # terminal facing constraint (soft) knobs
        facing_tilt_deg=8.0,
        w_face_slack=5e4,
        # table soft constraint weight
        w_table_slack=5e4,
        # tool transform: wrist_3_link -> tool0
        R_wrist3_tool0=None,
        p_wrist3_tool0=None,
        tool_q_xyzw=None,
        tool_p=None,
        solver_opts=None,
    ):
        self.N = int(N)
        self.dt = float(dt)

        # Joint limits
        if q_min is None:
            q_min = np.array([-2*np.pi]*6, dtype=float)
        if q_max is None:
            q_max = np.array([ 2*np.pi]*6, dtype=float)

        # Step limits
        if dq_min is None:
            dq_min = np.array([-0.15]*6, dtype=float)
        if dq_max is None:
            dq_max = np.array([ 0.15]*6, dtype=float)

        self.q_min = np.asarray(q_min, dtype=float).reshape(6,)
        self.q_max = np.asarray(q_max, dtype=float).reshape(6,)
        self.dq_min = np.asarray(dq_min, dtype=float).reshape(6,)
        self.dq_max = np.asarray(dq_max, dtype=float).reshape(6,)

        self.cube_side = float(cube_side)
        self.safety_margin = float(safety_margin)
        self.use_lowest_sphere_for_table = bool(use_lowest_sphere_for_table)
        self.table_margin = float(table_margin)

        # weights
        self.w_pos = float(w_pos)
        self.w_rot = float(w_rot)
        self.w_dq = float(w_dq)
        self.w_terminal_mult = float(w_terminal_mult)

        # terminal facing params
        self.facing_tilt_deg = float(facing_tilt_deg)
        self.w_face_slack = float(w_face_slack)

        # table soft params
        self.w_table_slack = float(w_table_slack)

        # Proxy spheres offsets/radii
        if sphere_offsets_ee is None:
            sphere_offsets_ee = [
                np.array([0.0, 0.0, 0.00]),
                np.array([0.0, 0.0, 0.02]),
                np.array([0.0, 0.0, 0.03]),
            ]

        r_cube = 0.5*np.sqrt(3.0)*self.cube_side
        if sphere_radii is None:
            sphere_radii = [
                0.035,
                0.030,
                r_cube,
            ]

        assert len(sphere_offsets_ee) == len(sphere_radii), "Offsets and radii length mismatch."
        self.sphere_offsets_ee = [np.asarray(o, dtype=float).reshape(3,) for o in sphere_offsets_ee]
        self.sphere_radii = [float(r) + self.safety_margin for r in sphere_radii]  # inflate

        # Tool transform (wrist3 -> tool0)
        if R_wrist3_tool0 is not None:
            R_tool = np.asarray(R_wrist3_tool0, dtype=float).reshape(3, 3)
        elif tool_q_xyzw is not None:
            qx, qy, qz, qw = tool_q_xyzw
            R_tool = quat_to_rotmat_xyzw(qx, qy, qz, qw)
        else:
            R_tool = np.eye(3)

        if p_wrist3_tool0 is not None:
            p_tool = np.asarray(p_wrist3_tool0, dtype=float).reshape(3,)
        elif tool_p is not None:
            p_tool = np.asarray(tool_p, dtype=float).reshape(3,)
        else:
            p_tool = np.zeros(3)

        self.R_wrist3_tool0 = R_tool
        self.p_wrist3_tool0 = p_tool

        # facing tolerances
        tilt = np.deg2rad(self.facing_tilt_deg)
        self.facing_cos_tol = float(np.cos(tilt))
        self.facing_sin_tol = float(np.sin(tilt))

        # Build FK + distance and solver
        self.fk_T, self.fk_p, self.fk_R = self._build_ur7e_fk_functions_tool0()
        self.dist_point_aabb = self._build_point_aabb_distance_function()
        self._build_solver(solver_opts=solver_opts or {})

        self._last_dq_sol = None

    # -----------------------------
    # Public API
    # -----------------------------
    def solve(self, q0, p_goal, R_goal, aabb_bounds, pickup_height=None):
        """
        Solve NMPC.

        Returns dict:
          dq0, q_next, q_traj, dq_traj,
          s_face (terminal slack scalar),
          s_tab,
          success, solver_status, ipopt_stats
        """
        q0 = np.asarray(q0, dtype=float).reshape(6,)
        p_goal = np.asarray(p_goal, dtype=float).reshape(3,)
        R_goal = np.asarray(R_goal, dtype=float).reshape(3, 3)
        aabb_bounds = np.asarray(aabb_bounds, dtype=float).reshape(6,)

        if pickup_height is None:
            pickup_height = float(np.array(self.fk_p(ca.DM(q0))).reshape(3,)[2])

        p = self._pack_params(q0, p_goal, R_goal, aabb_bounds, float(pickup_height))
        x0 = self._initial_guess(q0)

        sol = None
        status = "unknown"
        stats = {}
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx,
                              lbg=self.lbg, ubg=self.ubg, p=p)
            stats = dict(self.solver.stats())
            status = str(stats.get("return_status", "unknown"))
        except Exception as e:
            return {
                "dq0": np.zeros(6),
                "q_next": q0.copy(),
                "q_traj": np.tile(q0, (self.N+1, 1)),
                "dq_traj": np.zeros((self.N, 6)),
                "s_face": 0.0,
                "s_tab": np.zeros((self.N, len(self.sphere_offsets_ee))) if self.use_lowest_sphere_for_table else np.zeros((self.N,)),
                "success": False,
                "solver_status": f"solve_failed: {e}",
                "ipopt_stats": {"exception": str(e)},
            }

        ok_statuses = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
        if sol is None or status not in ok_statuses:
            return {
                "dq0": np.zeros(6),
                "q_next": q0.copy(),
                "q_traj": np.tile(q0, (self.N+1, 1)),
                "dq_traj": np.zeros((self.N, 6)),
                "s_face": 0.0,
                "s_tab": np.zeros((self.N, len(self.sphere_offsets_ee))) if self.use_lowest_sphere_for_table else np.zeros((self.N,)),
                "success": False,
                "solver_status": status,
                "ipopt_stats": stats,
            }

        x_opt = np.array(sol["x"]).reshape(-1)
        q_traj, dq_traj, s_face_term, s_tab = self._unpack_decision_full(x_opt)
        self._last_dq_sol = dq_traj.copy()

        dq0 = dq_traj[0].copy()
        q_next = q0 + dq0

        return {
            "dq0": dq0,
            "q_next": q_next,
            "q_traj": q_traj,
            "dq_traj": dq_traj,
            "s_face": float(s_face_term),
            "s_tab": s_tab,
            "success": True,
            "solver_status": status,
            "ipopt_stats": stats,
        }

    # -----------------------------
    # Solver construction
    # -----------------------------
    def _build_solver(self, solver_opts):
        N = self.N
        M = len(self.sphere_offsets_ee)

        # Parameters
        q0_p = ca.SX.sym("q0", 6)
        p_goal = ca.SX.sym("p_goal", 3)
        R_goal = ca.SX.sym("R_goal", 3, 3)
        aabb = ca.SX.sym("aabb", 6)      # [xmin,xmax,ymin,ymax,zmin,zmax]
        h_pick = ca.SX.sym("h_pick", 1)  # table reference height
        P = ca.vertcat(q0_p, p_goal, ca.reshape(R_goal, 9, 1), aabb, h_pick)

        # Decision vars
        q_vars = ca.SX.sym("q_traj", 6*(N+1))
        dq_vars = ca.SX.sym("dq_traj", 6*N)

        # Terminal facing slack (>= 0)
        s_face_term = ca.SX.sym("s_face_term", 1)

        # Table slack(s) (>= 0)
        if self.use_lowest_sphere_for_table:
            s_tab = ca.SX.sym("s_tab", N*M)
        else:
            s_tab = ca.SX.sym("s_tab", N)

        X = ca.vertcat(q_vars, dq_vars, s_face_term, s_tab)

        def qk(k): return q_vars[6*k:6*(k+1)]
        def dqk(k): return dq_vars[6*k:6*(k+1)]
        def stab(k, i=None):
            if self.use_lowest_sphere_for_table:
                return s_tab[k*M + int(i)]
            return s_tab[k]

        # Bounds on decision vars
        lbx, ubx = [], []

        for _ in range(N+1):
            lbx.extend(self.q_min.tolist()); ubx.extend(self.q_max.tolist())
        for _ in range(N):
            lbx.extend(self.dq_min.tolist()); ubx.extend(self.dq_max.tolist())

        # s_face_term >= 0
        lbx.append(0.0); ubx.append(1e6)

        # s_tab >= 0
        if self.use_lowest_sphere_for_table:
            lbx.extend([0.0]*(N*M)); ubx.extend([1e6]*(N*M))
        else:
            lbx.extend([0.0]*N); ubx.extend([1e6]*N)

        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)

        # Cost weights
        Qp = self.w_pos * ca.SX.eye(3)
        Qr = self.w_rot * ca.SX.eye(3)
        Rw = self.w_dq * ca.SX.eye(6)
        QpN = self.w_terminal_mult * Qp
        QrN = self.w_terminal_mult * Qr

        cos_tol = self.facing_cos_tol
        sin_tol = self.facing_sin_tol

        g = []
        lbg = []
        ubg = []
        J = 0

        # q0 equality
        g.append(qk(0) - q0_p)
        lbg += [0.0]*6
        ubg += [0.0]*6

        for k in range(N):
            q_k = qk(k)
            dq_k = dqk(k)
            q_kp1 = qk(k+1)

            # dynamics
            g.append(q_kp1 - (q_k + dq_k))
            lbg += [0.0]*6
            ubg += [0.0]*6

            # pose costs (position always; rotation only if w_rot > 0)
            p_k = self.fk_p(q_k)
            e_p = p_k - p_goal
            J += ca.mtimes([e_p.T, Qp, e_p])
            if self.w_rot > 0.0:
                R_k = self.fk_R(q_k)
                e_R = self._orientation_error_log(R_k, R_goal)
                J += ca.mtimes([e_R.T, Qr, e_R])

            # control regularization
            J += ca.mtimes([dq_k.T, Rw, dq_k])

            # -----------------------------
            # HARD obstacle avoidance (NO SLACK)
            # For each proxy sphere center: dist(center, aabb) >= radius
            # -----------------------------
            centers = self._sphere_centers_from_q(q_k)  # stacked length 3*M
            for i in range(M):
                c_i = centers[3*i:3*(i+1)]
                d_i = self.dist_point_aabb(c_i, aabb)
                g.append(d_i)
                lbg.append(self.sphere_radii[i])
                ubg.append(1e19)

                # Table safety: soft (keep slack so IPOPT doesn't die on occasional infeas)
                if self.use_lowest_sphere_for_table:
                    st = stab(k, i)
                    threshold = h_pick + self.table_margin + self.sphere_radii[i]
                    g.append((c_i[2] + st) - threshold)
                    lbg.append(0.0)
                    ubg.append(1e19)
                    J += self.w_table_slack * (st*st)

            if not self.use_lowest_sphere_for_table:
                st = stab(k)
                threshold = h_pick + self.table_margin
                g.append((p_k[2] + st) - threshold)
                lbg.append(0.0)
                ubg.append(1e19)
                J += self.w_table_slack * (st*st)

        # -----------------------------
        # Terminal cost + terminal-only facing constraint
        # -----------------------------
        q_N = qk(N)
        p_N = self.fk_p(q_N)
        e_pN = p_N - p_goal
        J += ca.mtimes([e_pN.T, QpN, e_pN])

        if self.w_rot > 0.0:
            R_N = self.fk_R(q_N)
            e_RN = self._orientation_error_log(R_N, R_goal)
            J += ca.mtimes([e_RN.T, QrN, e_RN])

        # terminal-only facing-down (soft)
        R_N = self.fk_R(q_N)
        z_ee_N = R_N[:, 2]
        sfN = s_face_term[0]

        # -z_ee[2] + sfN >= cos_tol
        g.append(-z_ee_N[2] + sfN)
        lbg.append(cos_tol)
        ubg.append(1e19)

        # norm([zx,zy]) <= sin_tol + sfN  => norm - (sin_tol+sfN) <= 0
        g.append(ca.norm_2(z_ee_N[0:2]) - (sin_tol + sfN))
        lbg.append(-1e19)
        ubg.append(0.0)

        J += self.w_face_slack * (sfN*sfN)

        nlp = {"x": X, "f": J, "g": ca.vertcat(*g), "p": P}

        default_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 250,
            "ipopt.tol": 5e-4,
        }
        default_opts.update(solver_opts)

        self.solver = ca.nlpsol("mpc_solver", "ipopt", nlp, default_opts)
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)

        # Cache sizes for unpacking
        self._nx_q = 6*(N+1)
        self._nx_dq = 6*N
        self._nx_face = 1
        self._nx_tab = (N * M) if self.use_lowest_sphere_for_table else N

    # -----------------------------
    # Packing / Unpacking
    # -----------------------------
    def _pack_params(self, q0, p_goal, R_goal, aabb_bounds, pickup_height):
        return np.concatenate([
            np.asarray(q0).reshape(6,),
            np.asarray(p_goal).reshape(3,),
            np.asarray(R_goal).reshape(9,),
            np.asarray(aabb_bounds).reshape(6,),
            np.asarray([pickup_height]).reshape(1,)
        ]).astype(float)

    def _initial_guess(self, q0):
        N = self.N
        M = len(self.sphere_offsets_ee)
        q0 = np.asarray(q0, dtype=float).reshape(6,)

        if self._last_dq_sol is None:
            dq_guess = np.zeros((N, 6))
        else:
            dq_guess = np.vstack([self._last_dq_sol[1:], self._last_dq_sol[-1:]])

        q_guess = np.zeros((N+1, 6))
        q_guess[0] = q0
        for k in range(N):
            q_guess[k+1] = q_guess[k] + dq_guess[k]

        s_face_term_guess = np.array([0.0], dtype=float)

        if self.use_lowest_sphere_for_table:
            s_tab_guess = np.zeros((N*M,), dtype=float)
        else:
            s_tab_guess = np.zeros((N,), dtype=float)

        return np.concatenate([
            q_guess.reshape(-1),
            dq_guess.reshape(-1),
            s_face_term_guess.reshape(-1),
            s_tab_guess.reshape(-1),
        ]).astype(float)

    def _unpack_decision_full(self, x):
        """
        Returns:
          q_traj: (N+1,6)
          dq_traj:(N,6)
          s_face_term: scalar
          s_tab:  (N,M) if use_lowest_sphere_for_table else (N,)
        """
        N = self.N
        M = len(self.sphere_offsets_ee)

        i0 = 0
        q_flat = x[i0:i0 + self._nx_q]; i0 += self._nx_q
        dq_flat = x[i0:i0 + self._nx_dq]; i0 += self._nx_dq
        face_flat = x[i0:i0 + self._nx_face]; i0 += self._nx_face
        tab_flat = x[i0:i0 + self._nx_tab]; i0 += self._nx_tab

        q_traj = q_flat.reshape(N+1, 6)
        dq_traj = dq_flat.reshape(N, 6)
        s_face_term = float(face_flat.reshape(1,)[0])

        if self.use_lowest_sphere_for_table:
            s_tab = tab_flat.reshape(N, M)
        else:
            s_tab = tab_flat.reshape(N,)

        return q_traj, dq_traj, s_face_term, s_tab

    # -----------------------------
    # Geometry & Errors
    # -----------------------------
    def _sphere_centers_from_q(self, q_k):
        """
        Returns stacked centers [c1; c2; ...] where
          c_i(q) = p(q) + R(q) * offset_i
        """
        p = self.fk_p(q_k)
        R = self.fk_R(q_k)
        centers = []
        for off in self.sphere_offsets_ee:
            off_sx = ca.SX(off)
            centers.append(p + R @ off_sx)
        return ca.vertcat(*centers)

    def _orientation_error_log(self, R, R_goal):
        """ e_R = Log( R_goal^T * R ) in R^3 (axis-angle vector) """
        Rerr = R_goal.T @ R
        tr = ca.trace(Rerr)
        cos_theta = (tr - 1) / 2
        cos_theta = ca.fmin(1.0, ca.fmax(-1.0, cos_theta))
        theta = ca.acos(cos_theta)

        vee = ca.vertcat(
            Rerr[2, 1] - Rerr[1, 2],
            Rerr[0, 2] - Rerr[2, 0],
            Rerr[1, 0] - Rerr[0, 1]
        )

        eps = 1e-6
        scale = ca.if_else(theta < eps, 0.5, theta / (2 * ca.sin(theta)))
        return scale * vee

    # -----------------------------
    # CasADi model builders
    # -----------------------------
    def _build_point_aabb_distance_function(self, name="dist_point_aabb"):
        p = ca.SX.sym("p", 3)
        b = ca.SX.sym("b", 6)  # xmin,xmax,ymin,ymax,zmin,zmax

        xmin, xmax, ymin, ymax, zmin, zmax = b[0], b[1], b[2], b[3], b[4], b[5]

        cx = ca.fmin(ca.fmax(p[0], xmin), xmax)
        cy = ca.fmin(ca.fmax(p[1], ymin), ymax)
        cz = ca.fmin(ca.fmax(p[2], zmin), zmax)

        c = ca.vertcat(cx, cy, cz)
        d = ca.norm_2(p - c)
        return ca.Function(name, [p, b], [d], ["p", "bounds"], ["d"])

    def _build_ur7e_fk_functions_tool0(self):
        """
        POE FK -> wrist_3_link, then apply wrist3->tool0 transform.

        Returns:
          fk_T(q)->4x4, fk_p(q)->3x1, fk_R(q)->3x3 for tool0 in base_link frame.
        """
        q = ca.SX.sym("q", 6)

        # UR7e geometry (same as your original)
        q0 = ca.DM.zeros(3, 6)
        q0[:, 0] = ca.vertcat(0.,     0.,      0.1625)
        q0[:, 1] = ca.vertcat(0.,     0.,      0.1625)
        q0[:, 2] = ca.vertcat(0.425,  0.,      0.1625)
        q0[:, 3] = ca.vertcat(0.817,  0.1333,  0.1625)
        q0[:, 4] = ca.vertcat(0.817,  0.1333,  0.06285)
        q0[:, 5] = ca.vertcat(0.817,  0.233,   0.06285)

        w0 = ca.DM.zeros(3, 6)
        w0[:, 0] = ca.vertcat(0., 0.,  1.)
        w0[:, 1] = ca.vertcat(0., 1.,  0.)
        w0[:, 2] = ca.vertcat(0., 1.,  0.)
        w0[:, 3] = ca.vertcat(0., 1.,  0.)
        w0[:, 4] = ca.vertcat(0., 0., -1.)
        w0[:, 5] = ca.vertcat(0., 1.,  0.)

        R_zero = ca.DM([
            [-1., 0., 0.],
            [ 0., 0., 1.],
            [ 0., 1., 0.]
        ])

        gst0 = ca.DM.eye(4)
        gst0[0:3, 0:3] = R_zero
        gst0[0:3, 3] = q0[:, 5]

        # Twists
        xi = ca.SX.zeros(6, 6)
        for i in range(6):
            omega = ca.SX(w0[:, i])
            q_point = ca.SX(q0[:, i])
            v = -ca.cross(omega, q_point)
            xi[0:3, i] = v
            xi[3:6, i] = omega

        def exp_twist(xi_vec, theta):
            v = xi_vec[0:3]
            omega = xi_vec[3:6]

            wx = ca.SX.zeros(3, 3)
            wx[0, 1] = -omega[2]
            wx[0, 2] =  omega[1]
            wx[1, 0] =  omega[2]
            wx[1, 2] = -omega[0]
            wx[2, 0] = -omega[1]
            wx[2, 1] =  omega[0]

            I3 = ca.SX.eye(3)
            R = I3 + ca.sin(theta) * wx + (1 - ca.cos(theta)) * (wx @ wx)
            V = I3 * theta + (1 - ca.cos(theta)) * wx + (theta - ca.sin(theta)) * (wx @ wx)
            p = V @ v

            g = ca.SX.eye(4)
            g[0:3, 0:3] = R
            g[0:3, 3] = p
            return g

        T = ca.SX.eye(4)
        for i in range(6):
            T = T @ exp_twist(xi[:, i], q[i])

        T_wrist3 = T @ ca.SX(gst0)

        R_tool = ca.DM(self.R_wrist3_tool0)
        p_tool = ca.DM(self.p_wrist3_tool0)

        T_wrist3_tool0 = ca.DM.eye(4)
        T_wrist3_tool0[0:3, 0:3] = R_tool
        T_wrist3_tool0[0:3, 3] = p_tool

        T_tool0 = T_wrist3 @ ca.SX(T_wrist3_tool0)

        p_ee = T_tool0[0:3, 3]
        R_ee = T_tool0[0:3, 0:3]

        fk_T = ca.Function("fk_T_tool0_base", [q], [T_tool0], ["q"], ["T_ee"])
        fk_p = ca.Function("fk_p_tool0_base", [q], [p_ee], ["q"], ["p_ee"])
        fk_R = ca.Function("fk_R_tool0_base", [q], [R_ee], ["q"], ["R_ee"])
        return fk_T, fk_p, fk_R


if __name__ == "__main__":
    mpc = MPCController(N=15, cube_side=0.05, safety_margin=0.01)
    q0 = np.zeros(6)
    p_goal = np.array([0.5, 0.0, 0.25])
    R_goal = np.eye(3)
    aabb = np.array([0.3, 0.4, -0.1, 0.1, 0.0, 0.25])
    out = mpc.solve(q0, p_goal, R_goal, aabb, pickup_height=0.0)
    print("success:", out["success"], "status:", out["solver_status"])
    print("dq0:", out["dq0"])
    print("q_next:", out["q_next"])
    print("s_face_term:", out["s_face"])
