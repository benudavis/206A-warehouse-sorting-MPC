#!/usr/bin/env python3
"""Forward kinematics helper that is *exactly* consistent with MuJoCo.

This deliberately avoids hand-written DH parameters, because those almost
never match the exact frames used in a specific URDF / MJCF (base offsets,
tool frames, etc.).

Instead we:
  - keep a scratch `MjData`,
  - write q into it,
  - call `mj_forward`,
  - read out body / site positions directly.

This guarantees that:
    compute_ee_position(q) == data.site_xpos[ee_site]   (up to FP noise)
for the same `q`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import mujoco
import numpy as np


@dataclass
class LinkNames:
    """Names of the arm bodies we care about for diagnostics / collision."""
    shoulder: str = "arm_shoulder_link"
    elbow: str = "arm_forearm_link"
    wrist: str = "arm_wrist_3_link"   # last wrist link is usually most useful


class FKSolver:
    """Forward kinematics using the MuJoCo model as ground truth."""

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "arm_hand_pinch",
        joint_slice: slice = slice(0, 6),
        link_names: LinkNames | None = None,
    ) -> None:
        """
        Args:
            model: compiled `MjModel` that contains the robot.
            site_name: name of the EE site (e.g. "arm_hand_pinch").
            joint_slice: range of qpos indices corresponding to the arm joints.
            link_names: optional override for shoulder / elbow / wrist bodies.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_slice = joint_slice

        # End-effector site
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self.site_id < 0:
            raise ValueError(f"EE site '{site_name}' not found in model")

        # Link bodies
        self.names = link_names or LinkNames()

        self.shoulder_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.names.shoulder
        )
        self.elbow_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.names.elbow
        )
        self.wrist_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.names.wrist
        )

        missing: list[str] = []
        if self.shoulder_id < 0:
            missing.append(self.names.shoulder)
        if self.elbow_id < 0:
            missing.append(self.names.elbow)
        if self.wrist_id < 0:
            missing.append(self.names.wrist)
        if missing:
            print(
                "[FKSolver] Warning: could not find link bodies "
                + ", ".join(missing)
                + " – only EE FK will be available."
            )

    # ---- internal helper -------------------------------------------------

    def _forward(self, q: np.ndarray) -> None:
        """Write q into scratch data and run mj_forward."""
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.joint_slice.stop - self.joint_slice.start:
            raise ValueError(
                f"Expected {self.joint_slice.stop - self.joint_slice.start} joints, "
                f"got shape {q.shape}"
            )
        self.data.qpos[:] = 0.0
        self.data.qpos[self.joint_slice] = q
        mujoco.mj_forward(self.model, self.data)

    # ---- public API ------------------------------------------------------

    def compute_ee_position(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector position (3,) in world frame for joint vector q."""
        self._forward(q)
        return self.data.site_xpos[self.site_id].copy()

    def compute_shoulder_position(self, q: np.ndarray) -> np.ndarray:
        """Return shoulder link world position (3,)."""
        if self.shoulder_id < 0:
            raise RuntimeError("Shoulder body not available in model")
        self._forward(q)
        return self.data.xpos[self.shoulder_id].copy()

    def compute_elbow_position(self, q: np.ndarray) -> np.ndarray:
        """Return elbow link world position (3,)."""
        if self.elbow_id < 0:
            raise RuntimeError("Elbow body not available in model")
        self._forward(q)
        return self.data.xpos[self.elbow_id].copy()

    def compute_wrist_position(self, q: np.ndarray) -> np.ndarray:
        """Return wrist link world position (3,)."""
        if self.wrist_id < 0:
            raise RuntimeError("Wrist body not available in model")
        self._forward(q)
        return self.data.xpos[self.wrist_id].copy()

    def compute_all_link_positions(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """Convenience: get all important link positions in one call."""
        self._forward(q)
        out: Dict[str, np.ndarray] = {
            "ee": self.data.site_xpos[self.site_id].copy()
        }
        if self.shoulder_id >= 0:
            out["shoulder"] = self.data.xpos[self.shoulder_id].copy()
        if self.elbow_id >= 0:
            out["elbow"] = self.data.xpos[self.elbow_id].copy()
        if self.wrist_id >= 0:
            out["wrist"] = self.data.xpos[self.wrist_id].copy()
        return out
