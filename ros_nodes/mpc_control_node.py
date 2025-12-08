#!/usr/bin/env python3
"""
ROS 2 node that wraps the existing MPC + IK stack and publishes joint
trajectory commands for a UR5e + Robotiq hand.

Usage (example):
    ros2 run <your_package> mpc_control_node.py \
        --ros-args -p model_path:=/path/to/scene.xml \
        -p command_topic:=/scaled_joint_trajectory_controller/joint_trajectory
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Make sure local src/ is importable when the node is used in an overlay workspace.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.control.inverse_kinematics import IKSolver  # noqa: E402
from src.control.mpc_controller import MPCController  # noqa: E402


def _duration_from_seconds(dt: float) -> Duration:
    """Helper to build a ROS duration from floating seconds."""
    sec = int(dt)
    nsec = int((dt - sec) * 1e9)
    return Duration(sec=sec, nanosec=nsec)


class MPCControlNode(Node):
    """ROS 2 bridge for the MPC/IK controllers."""

    def __init__(self) -> None:
        super().__init__("mpc_control_node")

        # ---- Parameters (can be set via YAML/launch) ----
        default_model = str(REPO_ROOT / "sim" / "models" / "scene.xml")
        self.declare_parameter("model_path", default_model)
        self.declare_parameter("ee_site", "arm_hand_pinch")
        self.declare_parameter(
            "joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        self.declare_parameter("home_position", [0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter(
            "command_topic",
            "/scaled_joint_trajectory_controller/joint_trajectory",
        )
        self.declare_parameter("target_pose_topic", "/mpc/target_pose")
        self.declare_parameter("control_rate_hz", 20.0)  # 50 ms loop
        self.declare_parameter("mpc_horizon", 20)
        self.declare_parameter("mpc_dt", 0.05)
        self.declare_parameter("mpc_max_velocity", 1.5)
        self.declare_parameter("mpc_position_weight", 500.0)
        self.declare_parameter("mpc_terminal_weight", 1000.0)
        self.declare_parameter("mpc_smoothness_weight", 0.1)
        self.declare_parameter(
            "obstacle_boxes",
            [],  # list of strings "cx cy cz sx sy sz" (center + half-size)
        )
        self.declare_parameter("ik_tolerance", 0.01)

        model_path = Path(
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        ee_site = self.get_parameter("ee_site").get_parameter_value().string_value
        self.joint_names: List[str] = (
            self.get_parameter("joint_names").get_parameter_value().string_array_value
        )
        self.home_position = np.array(
            self.get_parameter("home_position").get_parameter_value().double_array_value,
            dtype=float,
        )
        self.control_rate_hz = float(
            self.get_parameter("control_rate_hz").get_parameter_value().double_value
        )

        # ---- Load MuJoCo model for kinematics/collision heuristics ----
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.data_scratch = mujoco.MjData(self.model)
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site
        )

        # ---- Controllers ----
        horizon = int(self.get_parameter("mpc_horizon").get_parameter_value().integer_value)
        mpc_dt = float(self.get_parameter("mpc_dt").get_parameter_value().double_value)
        control_dt = 1.0 / max(1e-3, self.control_rate_hz)
        if abs(control_dt - mpc_dt) > 1e-4:
            self.get_logger().warn(
                f"control_rate_hz ({self.control_rate_hz:.2f} Hz) and mpc_dt ({mpc_dt:.3f} s) differ; using mpc_dt."
            )
        self.dt = mpc_dt
        self.control_rate_hz = 1.0 / self.dt

        self.mpc = MPCController(
            n_joints=len(self.joint_names),
            horizon=horizon,
            dt=mpc_dt,
        )
        self.mpc.set_cost_weights(
            self.get_parameter("mpc_position_weight").value,
            self.get_parameter("mpc_terminal_weight").value,
            self.get_parameter("mpc_smoothness_weight").value,
        )
        self.mpc.set_velocity_limit(self.get_parameter("mpc_max_velocity").value)
        self.mpc.initialize_link_bodies(self.model)
        self._load_obstacles_from_param()

        self.ik = IKSolver(self.model, self.data, site_name=ee_site)
        self.ik_tolerance = float(self.get_parameter("ik_tolerance").value)

        # ---- State ----
        self.current_q: Optional[np.ndarray] = None
        self.current_dq: Optional[np.ndarray] = None
        self.target_q: Optional[np.ndarray] = None
        self.latest_pose: Optional[PoseStamped] = None

        # ---- ROS interfaces ----
        joint_state_topic = (
            self.get_parameter("joint_state_topic").get_parameter_value().string_value
        )
        target_pose_topic = (
            self.get_parameter("target_pose_topic").get_parameter_value().string_value
        )
        self.command_topic = (
            self.get_parameter("command_topic").get_parameter_value().string_value
        )

        self.create_subscription(JointState, joint_state_topic, self._joint_state_cb, 10)
        self.create_subscription(PoseStamped, target_pose_topic, self._target_pose_cb, 10)
        self.cmd_pub = self.create_publisher(JointTrajectory, self.command_topic, 10)
        self.home_srv = self.create_service(Trigger, "go_home", self._handle_go_home)

        self.control_timer = self.create_timer(self.dt, self._control_step)

        self.get_logger().info(
            f"MPC control node ready. Model: {model_path}, command topic: {self.command_topic}"
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def _load_obstacles_from_param(self) -> None:
        """Parse obstacle list from ROS parameter."""
        box_strings = self.get_parameter("obstacle_boxes").get_parameter_value().string_array_value
        self.mpc.clear_obstacles()
        for spec in box_strings:
            try:
                parts = [float(x) for x in spec.split()]
                if len(parts) != 6:
                    raise ValueError
                center = parts[:3]
                half_size = parts[3:]
                self.mpc.add_obstacle(center, half_size)
            except Exception:
                self.get_logger().warn(
                    f"Ignoring obstacle spec '{spec}' (expected 'cx cy cz sx sy sz')."
                )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        q = []
        dq = []
        for joint in self.joint_names:
            if joint in name_to_idx and name_to_idx[joint] < len(msg.position):
                idx = name_to_idx[joint]
                q.append(msg.position[idx])
                if idx < len(msg.velocity):
                    dq.append(msg.velocity[idx])
                else:
                    dq.append(0.0)
            else:
                # Missing joint -> hold previous if possible
                if self.current_q is not None:
                    q.append(float(self.current_q[len(q)]))
                    dq.append(0.0)
                else:
                    q.append(0.0)
                    dq.append(0.0)
        self.current_q = np.array(q, dtype=float)
        self.current_dq = np.array(dq, dtype=float)

    def _target_pose_cb(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def _handle_go_home(self, _, response):
        """Service to reset target to the configured home position."""
        self.target_q = self.home_position.copy()
        response.success = True
        response.message = "Target set to home configuration."
        return response

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def _control_step(self) -> None:
        if self.current_q is None or self.current_dq is None:
            return

        # Update IK target if a new pose was received
        if self.latest_pose is not None:
            pos = self.latest_pose.pose.position
            ori = self.latest_pose.pose.orientation
            target_pos = np.array([pos.x, pos.y, pos.z], dtype=float)
            target_quat = np.array([ori.w, ori.x, ori.y, ori.z], dtype=float)

            # Seed IK with the current joint estimate
            self.data.qpos[: len(self.joint_names)] = self.current_q
            mujoco.mj_forward(self.model, self.data)
            q_sol, success = self.ik.solve(
                target_pos, target_quat, tolerance=self.ik_tolerance
            )
            if success:
                self.target_q = q_sol
            else:
                self.get_logger().warn("IK failed, holding previous target.")

            self.latest_pose = None  # consume

        # If no target yet, hold current pose
        if self.target_q is None:
            self.target_q = self.current_q.copy()

        # Build state vector [q, dq] for MPC
        state = np.concatenate([self.current_q, self.current_dq])
        q_next, _ = self.mpc.compute_control(
            state,
            self.target_q,
            model=self.model,
            data_scratch=self.data_scratch,
            site_id=self.ee_site_id,
        )
        self._publish_trajectory(q_next)

    # ------------------------------------------------------------------
    # Publishing helper
    # ------------------------------------------------------------------
    def _publish_trajectory(self, q_cmd: Sequence[float]) -> None:
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = list(q_cmd)
        point.time_from_start = _duration_from_seconds(self.dt)

        traj.points.append(point)
        self.cmd_pub.publish(traj)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = MPCControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
