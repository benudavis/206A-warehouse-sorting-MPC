#!/usr/bin/env python3
"""
Minimal ROS 2 node to command a Robotiq 2F-85 gripper via a position
controller topic.

Exposes services:
  - /open_gripper  (std_srvs/Trigger)
  - /close_gripper (std_srvs/Trigger)

The node simply republishes the latest command at a fixed rate to keep
the ros2_control hardware interface latched.
"""

from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_srvs.srv import Trigger


class RobotiqGripperNode(Node):
    """Lightweight publisher + services for Robotiq position control."""

    def __init__(self) -> None:
        super().__init__("robotiq_gripper_node")

        self.declare_parameter("command_topic", "/robotiq_gripper/command")
        self.declare_parameter("open_position", 0.0)    # meters at fingertip or normalized 0..1
        self.declare_parameter("closed_position", 0.8)  # meters at fingertip or normalized 0..1
        self.declare_parameter("hold_rate_hz", 5.0)

        self.command_topic = (
            self.get_parameter("command_topic").get_parameter_value().string_value
        )
        self.open_pos = float(self.get_parameter("open_position").value)
        self.closed_pos = float(self.get_parameter("closed_position").value)
        self.hold_dt = 1.0 / max(1e-3, float(self.get_parameter("hold_rate_hz").value))

        self.cmd_pub = self.create_publisher(Float64, self.command_topic, 10)
        self.current_command = self.open_pos

        self.create_service(Trigger, "open_gripper", self._handle_open)
        self.create_service(Trigger, "close_gripper", self._handle_close)

        self.timer = self.create_timer(self.hold_dt, self._republish_command)
        self.get_logger().info(
            f"Robotiq gripper node ready. Topic: {self.command_topic}"
        )

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------
    def _handle_open(self, _, response):
        self.current_command = self.open_pos
        self._publish(self.current_command)
        response.success = True
        response.message = "Gripper opening."
        return response

    def _handle_close(self, _, response):
        self.current_command = self.closed_pos
        self._publish(self.current_command)
        response.success = True
        response.message = "Gripper closing."
        return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _republish_command(self) -> None:
        self._publish(self.current_command)

    def _publish(self, value: float) -> None:
        msg = Float64()
        msg.data = float(value)
        self.cmd_pub.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = RobotiqGripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
