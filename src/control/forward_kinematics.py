#!/usr/bin/env python3

"""
Analytic forward kinematics for UR5e using POE / DH-derived parameters.
Exposes:
    build_ur5e_fk_function() -> casadi.Function

which maps q (6,) -> ee position (3,) at arm_hand_pinch site.
This is fully symbolic in CasADi and can be used inside the MPC instead
of the neural-network FK.
"""

from __future__ import annotations
import numpy as np
import casadi as ca

def build_ur5e_fk_function():
    """
    Build a CasADi function fk(q) -> p_ee (3,).
    Conventions:
        - q is a 6-element vector [q1..q6] in radians.
        - EE position is at arm_hand_pinch site (gripper pinch point),
          in the MuJoCo *world* frame for the assembled scene.xml.
        - Uses UR5e kinematic parameters + fixed tool0->pinch offset
          + base->world mounting transform.
    """
    # Joint vector
    q = ca.SX.sym("q", 6)

    # Basis vectors
    ex = ca.SX([1.0, 0.0, 0.0])
    ey = ca.SX([0.0, 1.0, 0.0])
    ez = ca.SX([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    # Kinematic parameters (H, P, R_6T) from Elias' UR5e model
    # ------------------------------------------------------------------
    # Joint axes H (3x6), each column is an axis in base frame at zero config
    H = ca.SX(3, 6)
    H[:, 0] = ez
    H[:, 1] = -ey
    H[:, 2] = -ey
    H[:, 3] = -ey
    H[:, 4] = -ez
    H[:, 5] = -ey

    # Displacement vectors P (3x7)
    # Optimized to match MuJoCo arm_hand_pinch site (see scripts/optimize_fk_parameters.py)
    P = ca.SX(3, 7)
    # p01
    P[:, 0] = ca.SX([-0.0002000011246430994, 0.12381999433286241, 0.13800000017695882])
    # p12
    P[:, 1] = ca.SX([-2.5310463076446646e-09, -0.00017500068716176992, -0.024499999824680645])
    # p23
    P[:, 2] = ca.SX([-0.4250000085106623, -0.00017500068667997845, -9.379979883136704e-09])
    # p34
    P[:, 3] = ca.SX([-0.39200000521049083, -0.00017500068973860236, -2.1210003962969342e-09])
    # p45
    P[:, 4] = ca.SX([-2.5087870245593798e-09, -0.13347500069013965, -0.09985000022399897])
    # p56
    P[:, 5] = ca.SX([-2.0967273265274825e-09, -0.06155999960694523, -0.00015000022379548536])
    # p6T (tool offset)
    P[:, 6] = ca.SX([-3.7045055566769567e-09, -0.16115999966160519, 0.04929999208756993])

    # Tool rotation R_6T = Rot_x(+pi/2)
    th_tool = ca.pi / 2
    c = ca.cos(th_tool)
    s = ca.sin(th_tool)
    R_6T = ca.SX(3, 3)
    R_6T[0, 0] = 1.0
    R_6T[0, 1] = 0.0
    R_6T[0, 2] = 0.0
    R_6T[1, 0] = 0.0
    R_6T[1, 1] = c
    R_6T[1, 2] = -s
    R_6T[2, 0] = 0.0
    R_6T[2, 1] = s
    R_6T[2, 2] = c

    # ------------------------------------------------------------------
    # Helper: Rodrigues rotation around axis w by angle theta
    # ------------------------------------------------------------------
    def rot_axis(w: ca.SX, theta: ca.SX) -> ca.SX:
        wx, wy, wz = w[0], w[1], w[2]
        zero = ca.SX(0)
        
        # Skew-symmetric matrix W
        row1 = ca.hcat([zero, -wz, wy])
        row2 = ca.hcat([wz, zero, -wx])
        row3 = ca.hcat([-wy, wx, zero])
        W = ca.vertcat(row1, row2, row3)
        
        I3 = ca.SX_eye(3)
        return I3 + ca.sin(theta) * W + (1.0 - ca.cos(theta)) * (W @ W)

    # ------------------------------------------------------------------
    # Forward kinematics computation
    # ------------------------------------------------------------------
    # Start at base: R_00 = I, p_0 = p01
    R = ca.SX_eye(3)
    p = P[:, 0]  # p01

    # Go through joints 1..6
    for i in range(6):
        R = R @ rot_axis(H[:, i], q[i])
        if i < 5:
            # p += R_0(i+1) * p_(i+1)(i+2)
            p = p + R @ P[:, i + 1]

    # After loop:
    #   R = R_06
    #   p = p_06
    R_06 = R
    p_06 = p

    # End-effector pose at tool0 frame:
    p_0T = p_06 + R_06 @ P[:, 6]
    R_0T = R_06 @ R_6T
    
    # ------------------------------------------------------------------
    # Add offset from tool0 to pinch site (arm_hand_pinch) in UR base frame
    # Computed from MuJoCo model at zero config (see scripts/compute_tool0_to_pinch_offset.py)
    # ------------------------------------------------------------------
    TOOL0_TO_PINCH = np.array([0.0, -0.0493, 0.03308], dtype=float)
    offset_pinch = ca.SX(TOOL0_TO_PINCH)  # (3,)
    
    # Pinch position in UR base frame
    p_pinch_base = p_0T + R_0T @ offset_pinch  # (3,)
    
    # ------------------------------------------------------------------
    # Fixed transform: UR base frame -> MuJoCo world frame
    # (measured at q = 0 from the assembled scene)
    # The robot is mounted in the scene with this fixed offset.
    # world_pinch = base_pinch + BASE_TO_WORLD_OFFSET
    # ------------------------------------------------------------------
    BASE_TO_WORLD_OFFSET = np.array([0.0002, -0.12382, 0.4495], dtype=float)
    base_to_world_offset = ca.SX(BASE_TO_WORLD_OFFSET)
    
    # Final pinch position in MuJoCo world frame
    p_pinch_world = p_pinch_base + base_to_world_offset  # (3,)

    fk_fun = ca.Function("fk_ur5e_pinch", [q], [p_pinch_world])
    return fk_fun
