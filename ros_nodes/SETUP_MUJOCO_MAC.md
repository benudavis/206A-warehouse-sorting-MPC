# MuJoCo Setup Guide

Complete guide for running MuJoCo simulation in Docker with visualization on Mac.

## Architecture

```
┌─────────────────────────────────────┐
│  Docker Container (ROS2)            │
│  • MuJoCo sim (headless)            │
│  • Publishes /joint_states          │
│  • Publishes /box_coordinates       │
│  • Executes trajectories            │
└─────────────────────────────────────┘
         │
         │ ROS2 Topics (network)
         │
┌────────▼─────────────────────────────┐
│  Mac Machine                          │
│  • MuJoCo viewer (native)             │
│  • Subscribes to ROS2 topics         │
│  • Visualizes robot/boxes            │
└───────────────────────────────────────┘
```

## Step 1: Install Requirements on Mac

**IMPORTANT**: `rclpy` is part of ROS2 and cannot be installed via pip alone.

### Option A: Install ROS2 on Mac (Recommended for MuJoCo Viewer)

```bash
# Install ROS2 Humble
brew install ros-humble-desktop

# Source ROS2
source /opt/ros/humble/setup.bash

# Add to ~/.zshrc
echo "source /opt/ros/humble/setup.bash" >> ~/.zshrc

# Install other packages (in virtual environment)
python3 -m venv venv_mujoco
source venv_mujoco/bin/activate
pip install numpy mujoco
```

### Option B: Use Virtual Environment

The `run_mujoco_mac.sh` script will automatically create a virtual environment and install packages (except rclpy, which requires ROS2).

**Note**: To use the MuJoCo viewer on Mac, you need ROS2 Humble installed. The viewer connects to ROS2 topics from Docker via the network.

## Step 2: Start Docker Containers

```bash
cd ros_nodes

# Start MuJoCo sim and other nodes (headless)
docker-compose -f docker-compose-mujoco.yml up -d

# Check containers are running
docker ps

# You should see:
# - mujoco_sim
# - mpc_controller (optional)
# - ik_solver
# - task_planner (optional)
```

## Step 3: Run MuJoCo Viewer on Mac

```bash
cd ros_nodes

# Set ROS domain ID (must match Docker!)
export ROS_DOMAIN_ID=42

# Run viewer
./run_mujoco_mac.sh

# OR manually:
python3 warehouse_sorting/warehouse_sorting/mujoco_viewer_mac.py
```

## Step 4: Verify Everything Works

### Check ROS2 Topics

In a new terminal (on Mac, if you have ROS2 installed):
```bash
export ROS_DOMAIN_ID=42
ros2 topic list

# Should see:
# /joint_states
# /box_coordinates
# /mpc/joint_trajectory
```

### Check Docker Logs

```bash
# Watch MuJoCo sim logs
docker logs -f mujoco_sim

# Watch other nodes
docker logs -f mpc_controller
docker logs -f task_planner
```

## What You'll See

1. **MuJoCo Viewer Window** opens on your Mac
2. **Robot arm** appears (UR5e)
3. **Boxes** appear (red and blue on table)
4. **Obstacles** appear (wall)
5. **Baskets** appear (red and blue)
6. **Real-time updates** as robot moves

## Testing the Setup

### Test 1: Check Box Positions

```bash
# In Docker container
docker exec -it mujoco_sim bash
source /opt/ros/humble/setup.bash
ros2 topic echo /box_coordinates
```

### Test 2: Send Test Trajectory

```bash
# In Docker container
docker exec -it mujoco_sim bash
source /opt/ros/humble/setup.bash

# Publish test trajectory
ros2 topic pub /mpc/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'base_link'}, \
   joint_names: ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', \
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'], \
   points: [{positions: [0.5, -1.2, 1.3, -1.67, -1.57, 0.0], \
             time_from_start: {sec: 2, nanosec: 0}}]}"
```

You should see the robot move in the MuJoCo viewer on your Mac!

### Test 3: Run MPC Test

```bash
# Start MPC test node in Docker
docker run -it --rm \
  --network ros_nodes_ros_network \
  -e ROS_DOMAIN_ID=42 \
  --name mpc_test \
  $(docker build -q -f Dockerfile ..) \
  bash -c "source /opt/ros/humble/setup.bash &&
           source /ros2_ws/install/setup.bash &&
           ros2 run warehouse_sorting mpc_test_node.py"
```

Watch the MuJoCo viewer - you should see MPC trajectories being executed!

## Troubleshooting

### ROS2 Not Installed on Mac

**Problem**: `rclpy` not found

**Solution**: Install ROS2 Humble on Mac:
```bash
brew install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

### Viewer Doesn't Connect

**Problem**: Viewer can't see ROS2 topics

**Solution**:
```bash
# Make sure ROS_DOMAIN_ID matches
export ROS_DOMAIN_ID=42

# Check if topics are available (requires ROS2 on Mac)
source /opt/ros/humble/setup.bash
ros2 topic list

# Check Docker network
docker network inspect ros_nodes_ros_network
```

### Viewer Shows No Robot

**Problem**: Robot doesn't appear in viewer

**Solution**:
```bash
# Check if joint states are being published
docker logs mujoco_sim | grep "joint_states"

# Check viewer logs for errors
./run_mujoco_mac.sh
```

### Box Positions Not Updating

**Problem**: Boxes don't move in viewer

**Solution**:
```bash
# Check box publisher
docker logs mujoco_sim | grep "box_coordinates"

# Verify box positions topic
docker exec -it mujoco_sim bash
source /opt/ros/humble/setup.bash
ros2 topic echo /box_coordinates
```

## Running Full System

### Start Everything

```bash
# Terminal 1: Start Docker containers
cd ros_nodes
docker-compose -f docker-compose-mujoco.yml up

# Terminal 2: Start MuJoCo viewer on Mac
cd ros_nodes
export ROS_DOMAIN_ID=42
./run_mujoco_mac.sh
```

### Run Task Planner

```bash
# In Docker (or add to docker-compose)
docker run -it --rm \
  --network ros_nodes_ros_network \
  -e ROS_DOMAIN_ID=42 \
  --name task_planner \
  $(docker build -q -f Dockerfile ..) \
  bash -c "source /opt/ros/humble/setup.bash &&
           source /ros2_ws/install/setup.bash &&
           ros2 run warehouse_sorting task_planner_node.py"
```

Watch the MuJoCo viewer - you'll see the robot pick and place boxes!

## Virtual Environment Setup

Since macOS uses an externally-managed Python environment, use a virtual environment:

```bash
# Create virtual environment (or use run_mujoco_mac.sh which does this automatically)
python3 -m venv venv_mujoco
source venv_mujoco/bin/activate

# Install packages (rclpy requires ROS2, but numpy and mujoco can be installed here)
pip install --upgrade pip
pip install numpy mujoco

# Note: rclpy must be installed via ROS2 Humble installation
```

## Quick Reference

```bash
# Start Docker containers
docker-compose -f docker-compose-mujoco.yml up -d

# View logs
docker logs -f mujoco_sim

# Stop containers
docker-compose -f docker-compose-mujoco.yml down

# Run viewer on Mac (requires ROS2 installed)
export ROS_DOMAIN_ID=42
./run_mujoco_mac.sh
```

## Next Steps

1. Install ROS2 Humble on Mac (for viewer)
2. Test basic setup (robot appears in viewer)
3. Test trajectory execution (robot moves)
4. Test MPC trajectories (obstacle avoidance)
5. Test full pick-and-place sequence
