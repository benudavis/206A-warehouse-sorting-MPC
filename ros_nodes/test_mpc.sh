#!/bin/bash
# Quick test script for MPC with box positions and obstacle avoidance
# Run this after starting docker-compose-mujoco.yml

set -e

echo "=========================================="
echo "MPC Test - Box Positions + Obstacle Avoidance"
echo "=========================================="
echo ""
echo "This test will:"
echo "  1. Subscribe to box positions from box_publisher"
echo "  2. Use MPC to plan trajectories avoiding obstacles"
echo "  3. Publish trajectories to /mpc/joint_trajectory"
echo ""
echo "Make sure docker-compose-mujoco.yml is running first!"
echo ""

# Check if we're in a container or need to run one
if [ -f /.dockerenv ]; then
    # Already in container
    echo "Running MPC test node..."
    source /opt/ros/humble/setup.bash
    source /ros2_ws/install/setup.bash
    ros2 run warehouse_sorting mpc_test_node.py
else
    # Need to run in container
    echo "Starting MPC test in Docker container..."
    docker run -it --rm \
      --network ros_nodes_ros_network \
      -e ROS_DOMAIN_ID=42 \
      --name mpc_test \
      $(docker build -q -f Dockerfile ..) \
      bash -c "source /opt/ros/humble/setup.bash && \
               source /ros2_ws/install/setup.bash && \
               ros2 run warehouse_sorting mpc_test_node.py"
fi
