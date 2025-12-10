from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    EmitEvent,
)
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    # --------------------------------------------------
    # RealSense camera
    # --------------------------------------------------
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '1920x1080x30',
        }.items(),
    )

    # --------------------------------------------------
    # Plane parameters for tabletop segmentation
    # --------------------------------------------------
    plane_a_launch_arg = DeclareLaunchArgument(
        'plane_a',
        default_value='0.0'
    )
    plane_b_launch_arg = DeclareLaunchArgument(
        'plane_b',
        default_value='1.0'
    )
    plane_c_launch_arg = DeclareLaunchArgument(
        'plane_c',
        default_value='0.0'
    )
    plane_d_launch_arg = DeclareLaunchArgument(
        'plane_d',
        default_value='-0.075'
    )

    plane_a = LaunchConfiguration('plane_a')
    plane_b = LaunchConfiguration('plane_b')
    plane_c = LaunchConfiguration('plane_c')
    plane_d = LaunchConfiguration('plane_d')

    # --------------------------------------------------
    # Perception node: process pointcloud -> /cube_pose
    # --------------------------------------------------
    perception_node = Node(
        package='perception',
        executable='process_pointcloud',
        name='process_pointcloud',
        output='screen',
        parameters=[{
            'plane.a': plane_a,
            'plane.b': plane_b,
            'plane.c': plane_c,
            'plane.d': plane_d,
            'max_distance': 0.6,  # optional, uses default if omitted
        }]
    )

    # --------------------------------------------------
    # ArUco marker detection
    # --------------------------------------------------
    aruco_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros2_aruco'),
                'launch',
                'aruco_recognition.launch.py'
            )
        )
    )

    # Which AR marker to treat as reference
    ar_marker_launch_arg = DeclareLaunchArgument(
        'ar_marker',
        default_value='ar_marker_8'  # match the marker ID most commonly detected in logs
    )
    ar_marker = LaunchConfiguration('ar_marker')

    # --------------------------------------------------
    # Static TF from AR marker to base_link
    # (implemented in planning/static_tf_transform.py)
    # --------------------------------------------------
    planning_tf_node = Node(
        package='planning',
        executable='tf',  # maps to static_tf_transform.py via setup.py
        name='tf_node',
        output='screen',
        parameters=[{
            'ar_marker': ar_marker,
        }]
    )

    # --------------------------------------------------
    # Static TF: base_link -> world
    # --------------------------------------------------
    static_base_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_base_world',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'base_link', 'world'],
        output='screen',
    )

    # --------------------------------------------------
    # UR MoveIt bringup
    # --------------------------------------------------
    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true")

    moveit_launch_file = os.path.join(
        get_package_share_directory("ur_moveit_config"),
        "launch",
        "ur_moveit.launch.py"
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz,
        }.items(),
    )

    # --------------------------------------------------
    # Transform /cube_pose (camera frame) -> /cube_pose_in_base (base_link)
    # --------------------------------------------------
    transform_cube_pose_node = Node(
        package='planning',
        executable='transform_cube_pose',
        name='transform_cube_pose',
        output='screen',
    )

    # --------------------------------------------------
    # Pick-and-place node (uses MPC for trajectory planning)
    # Uses:
    #   - /cube_pose_in_base
    #   - /joint_states
    #   - MoveIt services via IKPlanner
    #   - /scaled_joint_trajectory_controller/follow_joint_trajectory
    #   - /toggle_gripper
    # --------------------------------------------------
    pick_place_node = Node(
        package='planning',
        executable='main',  # from setup.py entry_points
        name='pick_place',
        output='screen',
    )

    # --------------------------------------------------
    # Global shutdown on any process exit
    # --------------------------------------------------
    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='SOMETHING BONKED'))]
        )
    )

    return LaunchDescription([
        # Launch args
        ar_marker_launch_arg,
        plane_a_launch_arg,
        plane_b_launch_arg,
        plane_c_launch_arg,
        plane_d_launch_arg,

        # Perception / camera / TF / MoveIt
        realsense_launch,
        aruco_launch,
        perception_node,
        planning_tf_node,
        static_base_world,
        moveit_launch,
        transform_cube_pose_node,

        # Pick-and-place controller node
        pick_place_node,

        # Shutdown handling
        shutdown_on_any_exit,
    ])
