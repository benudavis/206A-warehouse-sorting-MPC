"""Launch file for warehouse sorting system."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch import conditions


def generate_launch_description():
    """Generate launch description for warehouse sorting nodes."""
    
    # Declare arguments
    use_mpc_arg = DeclareLaunchArgument(
        'use_mpc',
        default_value='true',
        description='Whether to use MPC controller'
    )
    
    horizon_arg = DeclareLaunchArgument(
        'horizon',
        default_value='10',
        description='MPC horizon'
    )
    
    # Nodes
    mpc_controller = Node(
        package='warehouse_sorting',
        executable='mpc_controller_node.py',
        name='mpc_controller',
        output='screen',
        parameters=[{
            'horizon': LaunchConfiguration('horizon'),
            'dt': 0.05,
            'enable_fk': True,
        }],
        condition=conditions.IfCondition(LaunchConfiguration('use_mpc'))
    )
    
    ik_solver = Node(
        package='warehouse_sorting',
        executable='ik_solver_node.py',
        name='ik_solver',
        output='screen',
        parameters=[{
            'site_name': 'arm_hand_pinch',
        }]
    )
    
    gripper_controller = Node(
        package='warehouse_sorting',
        executable='gripper_controller_node.py',
        name='gripper_controller',
        output='screen'
    )
    
    task_planner = Node(
        package='warehouse_sorting',
        executable='task_planner_node.py',
        name='task_planner',
        output='screen'
    )
    
    return LaunchDescription([
        use_mpc_arg,
        horizon_arg,
        mpc_controller,
        ik_solver,
        gripper_controller,
        task_planner,
    ])
