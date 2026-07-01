"""Launch graphnav_planner, Nav2, and the PoseStamped action bridge."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import SetRemap
from nav2_common.launch import RewrittenYaml


def generate_launch_description() -> LaunchDescription:
    """Build the complete scheme-A execution launch description."""
    package_share = get_package_share_directory('graphnav_nav2_bridge')
    nav2_share = get_package_share_directory('nav2_bringup')
    graphnav_share = get_package_share_directory('graphnav_planner')

    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    start_graphnav_planner = LaunchConfiguration('start_graphnav_planner')
    start_bridge = LaunchConfiguration('start_bridge')
    goal_topic = LaunchConfiguration('goal_topic')
    action_name = LaunchConfiguration('action_name')
    min_update_distance = LaunchConfiguration('min_update_distance')
    min_update_yaw = LaunchConfiguration('min_update_yaw')
    min_update_period = LaunchConfiguration('min_update_period')
    goal_input_timeout = LaunchConfiguration('goal_input_timeout')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    log_level = LaunchConfiguration('log_level')

    declarations = [
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use the Gazebo /clock for all launched nodes',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically activate Nav2 lifecycle nodes',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=f'{package_share}/config/nav2_a300_odom.yaml',
            description='Nav2 parameters for the A300 odom-frame stack',
        ),
        DeclareLaunchArgument(
            'start_graphnav_planner',
            default_value='false',
            description='Also start graphnav_planner and its path follower',
        ),
        DeclareLaunchArgument(
            'start_bridge',
            default_value='true',
            description='Start the rate-limited PoseStamped-to-action bridge',
        ),
        DeclareLaunchArgument(
            'goal_topic',
            default_value='/goal_pose',
            description='PoseStamped look-ahead goal produced by path_follower',
        ),
        DeclareLaunchArgument(
            'action_name',
            default_value='/graphnav_navigate_to_pose',
            description='Dedicated Nav2 action, isolated from legacy clients',
        ),
        DeclareLaunchArgument('min_update_distance', default_value='0.5'),
        DeclareLaunchArgument('min_update_yaw', default_value='0.35'),
        DeclareLaunchArgument('min_update_period', default_value='1.0'),
        DeclareLaunchArgument('goal_input_timeout', default_value='2.0'),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/a300_0000/cmd_vel',
            description='Final TwistStamped output from Collision Monitor',
        ),
        DeclareLaunchArgument('log_level', default_value='info'),
    ]

    configured_params = RewrittenYaml(
        source_file=params_file,
        param_rewrites={'cmd_vel_out_topic': cmd_vel_topic},
        convert_types=True,
    )

    graphnav = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            f'{graphnav_share}/launch/graphnav_planner.launch.yml'
        ),
        condition=IfCondition(start_graphnav_planner),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    nav2 = GroupAction(
        actions=[
            SetRemap(src='navigate_to_pose', dst=action_name),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    f'{nav2_share}/launch/navigation_launch.py'
                ),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'autostart': autostart,
                    'params_file': configured_params,
                    'use_composition': 'False',
                    'log_level': log_level,
                }.items(),
            ),
        ]
    )

    bridge = Node(
        package='graphnav_nav2_bridge',
        executable='goal_pose_to_nav2',
        name='goal_pose_to_nav2',
        output='screen',
        condition=IfCondition(start_bridge),
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'goal_topic': goal_topic,
                'action_name': action_name,
                'min_update_distance': min_update_distance,
                'min_update_yaw': min_update_yaw,
                'min_update_period': min_update_period,
                'goal_input_timeout': goal_input_timeout,
                'flatten_to_2d': True,
            }
        ],
        arguments=['--ros-args', '--log-level', log_level],
    )

    return LaunchDescription(declarations + [graphnav, nav2, bridge])
