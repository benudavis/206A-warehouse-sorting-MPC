cmake_minimum_required(VERSION 3.8)
project(warehouse_sorting)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoxDetection.msg"
  "msg/ObstacleArray.msg"
  "srv/SolveIK.srv"
  "srv/AddObstacle.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

# Install Python scripts as executables  
install(PROGRAMS
  warehouse_sorting/mpc_controller_node.py
  warehouse_sorting/ik_solver_node.py
  warehouse_sorting/gripper_controller_node.py
  warehouse_sorting/task_planner_node.py
  warehouse_sorting/visualization_node.py
  warehouse_sorting/box_publisher_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config files
install(DIRECTORY
  launch
  config
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
