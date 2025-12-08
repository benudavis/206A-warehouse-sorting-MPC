"""Inverse Kinematics solver using MuJoCo Jacobians."""

import numpy as np
import mujoco


class IKSolver:
    """IK solver for end-effector position/orientation."""

    def __init__(self, model, data, site_name='arm_hand_pinch'):
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
        """Solve IK for target position/orientation."""
        data_ik = mujoco.MjData(self.model)
        data_ik.qpos[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, data_ik)

        use_ori = target_quat is not None
        site_id = self.site_id
        if use_ori:
            target_quat = np.array(target_quat) / (np.linalg.norm(target_quat) + 1e-9)
        
        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, data_ik)
            cur_pos = data_ik.site_xpos[site_id].copy()
            pos_err = target_pos - cur_pos
            
            if use_ori:
                cur_mat = data_ik.site_xmat[site_id].reshape(3, 3)
                w, x, y, z = target_quat
                R_tar = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
                ])
                R_err = cur_mat.T @ R_tar
                rotvec = np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]]) * 0.5
            else:
                rotvec = np.zeros(3)
            
            jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data_ik, jacp, jacr, site_id)
            
            if use_ori:
                err = np.concatenate([pos_err, rotvec])
                J = np.vstack([jacp[:, :6], jacr[:, :6]])
            else:
                err, J = pos_err, jacp[:, :6]
            
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(J.shape[0]), err)
            dq = np.clip(dq * 0.2, -0.15, 0.15)
            data_ik.qpos[:6] = np.clip(data_ik.qpos[:6] + dq, -2*np.pi, 2*np.pi)
            
            if np.linalg.norm(pos_err) < tolerance:
                if use_ori and np.linalg.norm(rotvec) >= 0.05:
                    continue
                return data_ik.qpos[:6].copy(), True
        
        mujoco.mj_forward(self.model, data_ik)
        final_error = np.linalg.norm(target_pos - data_ik.site_xpos[site_id])
        if final_error > 0.5:
            return self.data.qpos[:6].copy(), False
        return data_ik.qpos[:6].copy(), (final_error < 0.1)
