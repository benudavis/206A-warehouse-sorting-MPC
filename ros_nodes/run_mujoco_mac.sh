#!/bin/bash
# Script to run MuJoCo viewer on Mac
# This connects to ROS2 topics from Docker

set -e

echo "=========================================="
echo "MuJoCo Viewer for Mac"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Connect to ROS2 topics from Docker"
echo "  2. Visualize robot and boxes in MuJoCo"
echo ""
echo "Make sure:"
echo "  • Docker containers are running (docker-compose up)"
echo "  • ROS_DOMAIN_ID=42 matches Docker"
echo "  • Virtual environment is set up"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv_mujoco"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Installing packages (this may take a few minutes)..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install rclpy numpy mujoco
    echo "✓ Virtual environment created and packages installed"
else
    echo "Using existing virtual environment..."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check if packages are installed
if ! python3 -c "import rclpy" 2>/dev/null; then
    echo "Error: rclpy not installed in virtual environment"
    echo "Installing packages..."
    pip install rclpy numpy mujoco
fi

if ! python3 -c "import mujoco" 2>/dev/null; then
    echo "Error: mujoco not installed in virtual environment"
    echo "Installing packages..."
    pip install rclpy numpy mujoco
fi

# Set ROS domain ID
export ROS_DOMAIN_ID=42

# Run viewer
echo "Starting MuJoCo viewer..."
cd "$SCRIPT_DIR"
python3 warehouse_sorting/warehouse_sorting/mujoco_viewer_mac.py
