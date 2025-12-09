# Dockerfile for warehouse sorting ROS 2 nodes
FROM ros:humble-ros-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-control-msgs \
    ros-humble-trajectory-msgs \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    casadi>=3.6.0 \
    mujoco==3.3.6 \
    pyyaml>=6.0

# Create workspace
WORKDIR /ros2_ws

# Copy project source (context is now parent directory)
COPY src /ros2_ws/src_project
COPY ros_nodes/warehouse_sorting /ros2_ws/src/warehouse_sorting
COPY sim /ros2_ws/sim

# Build ROS workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --packages-select warehouse_sorting

# Source ROS workspace in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Set environment
ENV PYTHONPATH=/ros2_ws/src_project:$PYTHONPATH
ENV ROS_PYTHON_LOG_CONFIG_FILE=

# Source ROS in entrypoint
RUN echo '#!/bin/bash\nset -e\nsource /opt/ros/humble/setup.bash\nsource /ros2_ws/install/setup.bash\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
