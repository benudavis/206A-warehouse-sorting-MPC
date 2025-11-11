"""
Simulation State Extractor
Gets robot and object states from MuJoCo simulation
"""

import numpy as np
import mujoco


class SimulationState:
    """Extract state information from MuJoCo simulation."""
    
    def __init__(self, model, data):
        """
        Initialize state extractor.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
    def get_robot_state(self):
        """
        Get current robot joint state.
        
        Returns:
            state: Joint positions and velocities (12,) [q, dq]
        """
        # Get joint positions and velocities for arm (first 6 joints)
        q = self.data.qpos[:6].copy()
        dq = self.data.qvel[:6].copy()
        return np.concatenate([q, dq])
    
    def get_end_effector_pose(self):
        """
        Get end effector (TCP) pose.
        
        Returns:
            pose: Position and orientation (7,) [x, y, z, qw, qx, qy, qz]
        """
        # Get site for end effector
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'arm_tool0')
            pos = self.data.site_xpos[site_id].copy()
            quat = self.data.site_xmat[site_id].copy().reshape(3, 3)
            # Convert rotation matrix to quaternion
            quat_wxyz = self._mat_to_quat(quat)
            return np.concatenate([pos, quat_wxyz])
        except:
            # If site doesn't exist, return zeros
            return np.zeros(7)
    
    def get_object_pose(self, object_name='obj_cube'):
        """
        Get object pose.
        
        Args:
            object_name: Name of object body
            
        Returns:
            pose: Position and orientation (7,) [x, y, z, qw, qx, qy, qz]
        """
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()
            return np.concatenate([pos, quat])
        except:
            # If object doesn't exist, return zeros
            return np.zeros(7)
    
    def get_full_state(self):
        """
        Get full state observation for learning.
        
        Returns:
            observation: Dict with all relevant state information
        """
        return {
            'robot_state': self.get_robot_state(),  # (12,)
            'ee_pose': self.get_end_effector_pose(),  # (7,)
            'object_pose': self.get_object_pose(),  # (7,)
        }
    
    def get_observation_vector(self, target_q=None):
        """
        Get flattened observation vector for neural network.
        
        Args:
            target_q: Target joint positions (6,) - REQUIRED for learning!
        
        Returns:
            obs: Observation vector (32,) = 12 (robot) + 7 (ee) + 7 (object) + 6 (target)
        """
        state = self.get_full_state()
        obs_parts = [
            state['robot_state'],
            state['ee_pose'],
            state['object_pose']
        ]
        
        if target_q is not None:
            obs_parts.append(target_q)
        else:
            obs_parts.append(np.zeros(6))  # Fallback if no target provided
        
        return np.concatenate(obs_parts)
    
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

