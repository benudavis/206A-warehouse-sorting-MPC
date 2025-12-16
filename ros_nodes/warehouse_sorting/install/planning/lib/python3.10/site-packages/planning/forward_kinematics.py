#!/usr/bin/env python

import numpy as np
from planning import kin_func_skeleton as kfs
from sensor_msgs.msg import JointState


def ur7e_forward_kinematics_from_angles(joint_angles):
    """
    Calculate the orientation of the ur7e's end-effector tool given
    the joint angles of each joint in radians

    Parameters:
    ------------
    joint_angles ((6,) np.ndarray): 6 joint angles 
        (shoulder pan, shoulder lift, elbow, wrist1, wrist2, wrist3)

    Returns: 
    ------------
    (4x4) np.ndarray: homogeneous transformation matrix gst(theta)
    """
    # Points on each joint axis in the zero config
    q0 = np.zeros((3, 6))  

    # Axis vector of each joint axis in the zero config
    w0 = np.zeros((3, 6))  

    q0[:, 0] = [0.,     0.,      0.1625]   # shoulder pan
    q0[:, 1] = [0.,     0.,      0.1625]   # shoulder lift
    q0[:, 2] = [0.425,  0.,      0.1625]   # elbow
    q0[:, 3] = [0.817,  0.1333,  0.1625]   # wrist 1
    q0[:, 4] = [0.817,  0.1333,  0.06285]  # wrist 2
    q0[:, 5] = [0.817,  0.233,   0.06285]  # wrist 3 (tool frame origin)

    w0[:, 0] = [0.,  0.,  1.]    # shoulder pan
    w0[:, 1] = [0.,  1.,  0.]    # shoulder lift
    w0[:, 2] = [0.,  1.,  0.]    # elbow
    w0[:, 3] = [0.,  1.,  0.]    # wrist 1
    w0[:, 4] = [0.,  0., -1.]    # wrist 2 
    w0[:, 5] = [0.,  1.,  0.]    # wrist 3

    # Rotation matrix from base_link to wrist_3_link in zero config
    R = np.array([
        [-1.,  0.,  0.],
        [ 0.,  0.,  1.], 
        [ 0.,  1.,  0.]
    ])

    # --- Build twists ξ_i = [v_i; ω_i] with v_i = -ω_i × q_i ---
    xi = np.zeros((6, 6))
    for i in range(6):
        omega = w0[:, i]
        q = q0[:, i]
        v = -np.cross(omega, q)
        xi[0:3, i] = v
        xi[3:6, i] = omega

    # --- Zero-configuration transform gst(0) ---
    gst0 = np.eye(4)
    gst0[0:3, 0:3] = R
    # Use wrist_3_link origin as tool position at zero config
    gst0[0:3, 3] = q0[:, 5]

    # Ensure joint_angles is a 1D array of length 6
    joint_angles = np.asarray(joint_angles).reshape(6,)

    # --- Product of exponentials ---
    g_theta = kfs.prod_exp(xi, joint_angles) @ gst0

    return g_theta


def ur7e_forward_kinematics_from_joint_state(joint_state):
    """
    Computes the orientation of the ur7e's end-effector given the joint
    state.

    Parameters
    ----------
    joint_state (sensor_msgs.msg.JointState): JointState of ur7e robot

    Returns
    -------
    (4x4) np.ndarray: homogeneous transformation matrix
    """
    # Desired joint order for FK
    joint_order = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    angles = np.zeros(6)

    # Map from joint_state.name -> position
    name_to_pos = {name: pos for name, pos in zip(joint_state.name,
                                                  joint_state.position)}

    for i, jname in enumerate(joint_order):
        if jname in name_to_pos:
            angles[i] = name_to_pos[jname]
        else:
            # If a joint is missing, you might want to log or raise an error
            # For now, we leave it at 0 and optionally print a warning.
            # print(f"Warning: joint {jname} not found in JointState!")
            angles[i] = 0.0

    # Call the FK from angles
    gst = ur7e_forward_kinematics_from_angles(angles)

    return gst
