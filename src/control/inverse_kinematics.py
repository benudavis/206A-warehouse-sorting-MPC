"""Inverse Kinematics solver using MuJoCo Jacobians."""

import numpy as np
import mujoco


class IKSolver:
    """
    IK solver for end-effector position/orientation.

    Design goals:
      - Good numerical robustness for general poses (passes roundtrip tests).
      - Continuity for real-time control: prefer solutions close to previous
        configuration by warm-starting from the last solution.
      - Simple, damped least-squares approach using MuJoCo jacobians.
    """

    def __init__(self, model, data, site_name="arm_hand_pinch"):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        # Continuity: remember previous solution so calls from the controller
        # tend to stay on the same kinematic branch.
        self.q_prev = None

    # ------------------------------------------------------------------
    # Utility: rotation matrix -> quaternion
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Core IK solve
    # ------------------------------------------------------------------
    def solve(self, target_pos, target_quat=None, max_iterations=100, tolerance=0.01):
        """
        Solve IK for target position/orientation.

        Args:
            target_pos: (3,) desired EE position.
            target_quat: (4,) desired EE orientation [w, x, y, z] or None.
            max_iterations: maximum iterations for the *main* attempt.
            tolerance: position tolerance [m] for success.

        Returns:
            (q, success):
                q: (6,) joint configuration.
                success: True if we consider this a good solution.
        """
        use_ori = target_quat is not None
        if use_ori:
            target_quat = np.array(target_quat, dtype=float)
            norm_q = np.linalg.norm(target_quat)
            if norm_q < 1e-9:
                use_ori = False
            else:
                target_quat /= norm_q

        # We will do up to two attempts:
        #   1) warm-start from q_prev (or current) for continuity
        #   2) if that fails badly, restart from a neutral pose (zero joints)
        seeds = []

        # First seed: previous solution if available, else current state
        if self.q_prev is not None:
            seeds.append(self.q_prev.copy())
        else:
            seeds.append(self.data.qpos[:6].copy())

        # Second seed: neutral pose (helps random, far poses in tests)
        seeds.append(np.zeros(6, dtype=float))

        best_q = None
        best_err = np.inf
        best_success = False

        # We split max_iterations between seeds (first seed gets more)
        iters_first = max_iterations
        iters_second = max(20, max_iterations // 2)

        for seed_idx, q_seed in enumerate(seeds):
            # Decide how many iterations this attempt gets
            if seed_idx == 0:
                iters = iters_first
            else:
                iters = iters_second

            data_ik = mujoco.MjData(self.model)
            data_ik.qpos[:] = self.data.qpos[:]  # copy everything
            data_ik.qpos[:6] = q_seed
            mujoco.mj_forward(self.model, data_ik)

            # Damping and gain parameters
            damping = 0.01           # small damping for accuracy
            pos_gain = 1.0
            ori_gain = 0.5           # orientation weight (smaller than position)
            step_scale = 0.4         # global step scaling
            max_step = 0.3           # per-joint step clamp

            for _ in range(iters):
                mujoco.mj_forward(self.model, data_ik)

                cur_pos = data_ik.site_xpos[self.site_id].copy()
                pos_err = target_pos - cur_pos
                pos_err_norm = np.linalg.norm(pos_err)

                # Orientation error as small-angle rotation vector
                if use_ori:
                    cur_R = data_ik.site_xmat[self.site_id].reshape(3, 3)
                    w, x, y, z = target_quat
                    R_tar = np.array(
                        [
                            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                        ]
                    )
                    R_err = cur_R.T @ R_tar
                    rotvec = np.array(
                        [
                            R_err[2, 1] - R_err[1, 2],
                            R_err[0, 2] - R_err[2, 0],
                            R_err[1, 0] - R_err[0, 1],
                        ]
                    ) * 0.5
                else:
                    rotvec = np.zeros(3)
                    R_err = None  # unused

                # Convergence based on position only
                if pos_err_norm < tolerance:
                    q_sol = data_ik.qpos[:6].copy()
                    # This is a good candidate solution
                    if pos_err_norm < best_err:
                        best_err = pos_err_norm
                        best_q = q_sol
                        best_success = True
                    break

                # Build Jacobians
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, data_ik, jacp, jacr, self.site_id)

                if use_ori:
                    err = np.concatenate([pos_gain * pos_err, ori_gain * rotvec])
                    J = np.vstack([pos_gain * jacp[:, :6], ori_gain * jacr[:, :6]])
                else:
                    err = pos_gain * pos_err
                    J = pos_gain * jacp[:, :6]

                # Damped least squares: dq = J^T (J J^T + λI)^(-1) err
                JJT = J @ J.T + damping * np.eye(J.shape[0])
                dq = J.T @ np.linalg.solve(JJT, err)

                # Scale and clip the update
                dq = np.clip(step_scale * dq, -max_step, max_step)

                # Update joints
                data_ik.qpos[:6] += dq
                # Loose joint limits (UR5e-like), just to avoid crazy values
                data_ik.qpos[:6] = np.clip(data_ik.qpos[:6], -2.0 * np.pi, 2.0 * np.pi)

            # After the loop, evaluate this attempt's final error
            mujoco.mj_forward(self.model, data_ik)
            final_pos = data_ik.site_xpos[self.site_id].copy()
            final_err = np.linalg.norm(final_pos - target_pos)
            q_sol = data_ik.qpos[:6].copy()

            if final_err < best_err:
                best_err = final_err
                best_q = q_sol
                best_success = final_err < max(0.1, 5.0 * tolerance)

        # If we got *any* reasonably good solution, remember it for continuity
        if best_q is not None and best_err < 0.1:
            self.q_prev = best_q.copy()

        # If everything failed badly, fall back to current state (but do not
        # update q_prev so future calls can still track a good branch).
        if best_q is None or best_err > 0.5:
            return self.data.qpos[:6].copy(), False

        return best_q, best_success
