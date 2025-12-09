#!/usr/bin/env python3
"""
Simple node that publishes box positions.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String
import json


class BoxPublisherNode(Node):
    """Publishes box coordinates."""

    def __init__(self):
        super().__init__('box_publisher')
        
        # Box positions (matching your demo)
        self.boxes = [
            {"name": "red_1", "pos": [0.35, -0.28, 0.52], "color": "red"},
            {"name": "red_2", "pos": [0.40, -0.32, 0.52], "color": "red"},
            {"name": "red_3", "pos": [0.45, -0.28, 0.52], "color": "red"},
            {"name": "blue_1", "pos": [0.35, -0.42, 0.52], "color": "blue"},
            {"name": "blue_2", "pos": [0.40, -0.38, 0.52], "color": "blue"},
            {"name": "blue_3", "pos": [0.45, -0.42, 0.52], "color": "blue"},
        ]
        
        # Publisher
        self.box_data_pub = self.create_publisher(
            String,
            '/box_coordinates',
            10
        )
        
        # Timer to publish periodically
        self.timer = self.create_timer(1.0, self.publish_boxes)
        
        self.get_logger().info('Box Publisher Node ready')
        self.get_logger().info(f'Publishing {len(self.boxes)} box positions to /box_coordinates')

    def publish_boxes(self):
        """Publish all box coordinates."""
        msg = String()
        msg.data = json.dumps(self.boxes, indent=2)
        self.box_data_pub.publish(msg)
        
        # Also log to console
        self.get_logger().info('Box Coordinates:')
        for box in self.boxes:
            self.get_logger().info(f"  {box['name']}: {box['pos']} ({box['color']})")


def main(args=None):
    rclpy.init(args=args)
    node = BoxPublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
