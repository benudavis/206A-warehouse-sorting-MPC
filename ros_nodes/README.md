# ROS 2 Nodes for Warehouse Sorting with MPC

This package provides ROS 2 nodes for MPC-based warehouse sorting with the UR5e robot using MuJoCo simulation.

## Quick Start

### With MuJoCo Visualization (Recommended)

```bash
# 1. Start Docker containers
docker-compose -f docker-compose-mujoco.yml up -d

# 2. Run MuJoCo viewer on Mac
export ROS_DOMAIN_ID=42
./run_mujoco_mac.sh
```

See [SETUP_MUJOCO_MAC.md](SETUP_MUJOCO_MAC.md) for detailed setup instructions.

## Package Structure

```
warehouse_sorting/
├── warehouse_sorting/           # Python package
│   ├── mpc_controller_node.py   # MPC trajectory optimization
│   ├── ik_solver_node.py        # MoveIt-based IK service
│   ├── gripper_controller_node.py # Robotiq gripper control
│   ├── task_planner_node.py     # High-level task sequencer
│   ├── mujoco_sim_node.py       # MuJoCo simulation (headless)
│   ├── mujoco_viewer_mac.py     # MuJoCo viewer for Mac
│   └── mpc_test_node.py         # MPC testing node
├── launch/
│   ├── warehouse_sorting.launch.py
│   └── mujoco_sim.launch.py
└── config/
    ├── mpc_params.yaml
    └── ik_params.yaml
```

## Nodes

### 1. MPC Controller Node (`mpc_controller_node.py`)

**Subscribes:**
- `/joint_states` - Current robot state
- `/mpc/target_joint_state` - Target configuration

**Publishes:**
- `/mpc/joint_trajectory` - Optimized trajectory with obstacle avoidance

**Services:**
- `~/add_obstacle` - Add obstacle to MPC avoidance

### 2. IK Solver Node (`ik_solver_node.py`)

Uses MoveIt services for inverse kinematics.

**Services:**
- `~/solve_ik` - Solve IK for target pose
- `~/plan_kinematic_path` - Plan collision-free trajectory

### 3. MuJoCo Simulation Node (`mujoco_sim_node.py`)

Runs headless MuJoCo simulation in Docker.

**Publishes:**
- `/joint_states` - Robot joint states from simulation
- `/box_coordinates` - Box positions from simulation

**Subscribes:**
- `/mpc/joint_trajectory` - Executes trajectories
- `/gripper_events` - Handles box attachment

### 4. Task Planner Node (`task_planner_node.py`)

Orchestrates pick-and-place sequence using MoveIt IK + MPC.

**Subscribes:**
- `/joint_states` - Current robot state
- `/box_coordinates` - Box positions

**Publishes:**
- `/gripper_events` - Grasp/release commands
- `/mpc/target_joint_state` - Target for MPC

## Building

```bash
cd ros_nodes
colcon build --packages-select warehouse_sorting
source install/setup.bash
```

## Running

### Launch All Nodes

```bash
ros2 launch warehouse_sorting mujoco_sim.launch.py
```

### Run Individual Nodes

```bash
# MuJoCo simulation (in Docker)
ros2 run warehouse_sorting mujoco_sim_node.py

# MPC controller
ros2 run warehouse_sorting mpc_controller_node.py

# Task planner
ros2 run warehouse_sorting task_planner_node.py
```

## Testing MPC

```bash
# Test MPC with box positions and obstacles
./test_mpc.sh
```

## Configuration

Edit parameters in `config/*.yaml` files or pass via launch arguments:

```bash
ros2 launch warehouse_sorting mujoco_sim.launch.py horizon:=20
```

## Architecture

```
┌─────────────────────────────────────┐
│  Docker Container (ROS2)            │
│  • MuJoCo sim (headless)             │
│  • MPC controller                    │
│  • IK solver                         │
│  • Task planner                      │
└─────────────────────────────────────┘
         │
         │ ROS2 Topics
         │
┌────────▼─────────────────────────────┐
│  Mac Machine                          │
│  • MuJoCo viewer (native)             │
└───────────────────────────────────────┘
```

## Dependencies

**In Docker:**
- ROS 2 Humble
- MuJoCo
- MoveIt (for IK)

**On Mac (for viewer):**
- Python 3.10+
- rclpy (via ROS2 Humble installation)
- numpy, mujoco (via pip in virtual environment)

See [SETUP_MUJOCO_MAC.md](SETUP_MUJOCO_MAC.md) for installation instructions.
