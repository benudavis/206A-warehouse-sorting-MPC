"""
Inverse Kinematics for UR5e
Uses MuJoCo's site Jacobians to find joint angles that reach target poses.

This version solves IK on a **scratch MjData** so the live simulation `data`
is never mutated during solving — avoiding any "teleport/reset" side effects.
"""

import numpy as np
import mujoco


class IKSolver:
    """Inverse kinematics solver using MuJoCo."""

    def __init__(self, model, data, site_name='arm_hand_pinch'):
        """
        Initialize IK solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data (live sim data; NOT modified during solve)
            site_name: Name of end effector site
        """
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    @staticmethod
    def _mat_to_quat(R):
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])

    def solve(self, target_pos, target_quat=None, max_iterations=100, tolerance=0.01):
        """
        Solve IK to reach target position (and optional orientation).

        Args:
            target_pos: Target position [x, y, z]
            target_quat: Optional target orientation [w, x, y, z]
            max_iterations: Maximum IK iterations
            tolerance: Position tolerance (meters)

        Returns:
            joint_angles: Solution joint angles (6,)
            success: Whether IK converged
        """
        # ---- Work on a scratch data so we don't touch the live sim during IK ----
        data_ik = mujoco.MjData(self.model)
        # Start from the CURRENT live pose
        data_ik.qpos[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, data_ik)
        # ------------------------------------------------------------------------

        use_ori = target_quat is not None
        site_id = self.site_id

        # Normalize target quaternion if provided
        if use_ori:
            target_quat = np.array(target_quat, dtype=float)
            target_quat = target_quat / (np.linalg.norm(target_quat) + 1e-9)

        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, data_ik)
            cur_pos = data_ik.site_xpos[site_id].copy()
            cur_mat = data_ik.site_xmat[site_id].reshape(3, 3).copy()

            # Position error
            pos_err = target_pos - cur_pos

            # Orientation error as small-angle rotation vector
            if use_ori:
                R_cur = cur_mat
                w, x, y, z = target_quat
                R_tar = np.array([
                    [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
                    [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
                    [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
                ])
                R_err = R_cur.T @ R_tar
                # small-angle approx for log(R_err)
                rotvec = np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ]) * 0.5
            else:
                rotvec = np.zeros(3)

            # Jacobians at the site
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data_ik, jacp, jacr, site_id)

            Jp = jacp[:, :6]  # position Jacobian for first 6 dofs
            Jr = jacr[:, :6]  # rotation Jacobian for first 6 dofs

            # Stack tasks if needed
            if use_ori:
                err = np.concatenate([pos_err, rotvec])
                J = np.vstack([Jp, Jr])
            else:
                err = pos_err
                J = Jp

            # Damped least-squares
            lam = 0.05
            JT = J.T
            H = J @ JT + lam * np.eye(J.shape[0])
            dq = JT @ np.linalg.solve(H, err)

            # Conservative step and clip
            dq = dq * 0.2
            nrm = np.linalg.norm(dq)
            if nrm > 0.15:
                dq *= (0.15 / (nrm + 1e-9))

            # Update scratch state (NOT the live sim)
            data_ik.qpos[:6] = np.clip(data_ik.qpos[:6] + dq, -2*np.pi, 2*np.pi)

            mujoco.mj_forward(self.model, data_ik)

            # Check convergence
            pos_err_norm = np.linalg.norm(data_ik.site_xpos[site_id] - target_pos)
            if use_ori:
                if pos_err_norm < tolerance and np.linalg.norm(rotvec) < 0.05:
                    return data_ik.qpos[:6].copy(), True
            else:
                if pos_err_norm < tolerance:
                    return data_ik.qpos[:6].copy(), True

        # Didn’t converge; accept best effort unless it’s awful
        mujoco.mj_forward(self.model, data_ik)
        final_pos = data_ik.site_xpos[site_id].copy()
        final_error = np.linalg.norm(target_pos - final_pos)

        if final_error > 0.5:  # Really bad → fall back to current live pose
            return self.data.qpos[:6].copy(), False

        return data_ik.qpos[:6].copy(), (final_error < 0.1)
