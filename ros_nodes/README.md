# ROS 2 Nodes

Lightweight ROS 2 wrappers around the MPC/IK stack for lab bring-up. These
scripts can be run directly with `python3` after sourcing your ROS 2 distro, or
dropped into a ROS 2 workspace as a Python package. All paths default to the
UR5e + Robotiq MuJoCo model in `sim/models/scene.xml`.

## Nodes
- `mpc_control_node.py`: Subscribes to joint states and a target pose, solves IK
  + MPC, and publishes joint trajectories (UR driver compatible).
- `robotiq_gripper_node.py`: Publishes a position command for the 2F-85 and
  exposes `open_gripper`/`close_gripper` services.

## Quick Start (direct run)
```bash
source /opt/ros/humble/setup.bash   # or your ROS 2 distro
export PYTHONPATH=$PWD:$PYTHONPATH  # make src/ importable

# Arm control
python3 ros_nodes/mpc_control_node.py --ros-args \
  -p command_topic:=/scaled_joint_trajectory_controller/joint_trajectory \
  -p joint_state_topic:=/joint_states \
  -p model_path:=$(pwd)/sim/models/scene.xml \
  -p obstacle_boxes:="['0.10 -0.35 0.75 0.25 0.02 0.20','0.0 -0.30 0.85 0.20 0.15 0.02']"

# Gripper (assumes std_msgs/Float64 position interface)
python3 ros_nodes/robotiq_gripper_node.py --ros-args \
  -p command_topic:=/robotiq_gripper/command \
  -p open_position:=0.0 -p closed_position:=0.8
```

## Interfaces
- **mpc_control_node**
  - Subscribes: `sensor_msgs/JointState` (`joint_state_topic`, default `/joint_states`)
  - Subscribes: `geometry_msgs/PoseStamped` (`target_pose_topic`, default `/mpc/target_pose`)
  - Publishes: `trajectory_msgs/JointTrajectory` (`command_topic`, default `/scaled_joint_trajectory_controller/joint_trajectory`)
  - Service: `go_home` (`std_srvs/Trigger`) to reset the target to `home_position`
  - Params: `model_path`, `ee_site`, `joint_names`, `home_position`,
    `mpc_dt`, `mpc_horizon`, cost weights, `obstacle_boxes` (list of strings
    `"cx cy cz sx sy sz"` for center and half-extents).

- **robotiq_gripper_node**
  - Publishes: `std_msgs/Float64` position command (`command_topic`)
  - Services: `open_gripper`, `close_gripper` (both `std_srvs/Trigger`)
  - Params: `open_position`, `closed_position`, `hold_rate_hz`.

## Lab Notes
- The MPC timer uses `mpc_dt` (default 0.05 s → 20 Hz). Keep ROS control rates
  aligned with the UR `scaled_joint_trajectory_controller` update period.
- Default joint names match the UR driver (`shoulder_pan_joint` ... `wrist_3_joint`).
- Obstacles are optional; if provided they are passed into the MPC hard/soft
  box constraints.
- Set `MUJOCO_GL=osmesa` or `egl` if the machine has no display.
- Send a `PoseStamped` with orientation `[w,x,y,z]` → message order is x,y,z,w;
  the node handles the reordering for IK.
