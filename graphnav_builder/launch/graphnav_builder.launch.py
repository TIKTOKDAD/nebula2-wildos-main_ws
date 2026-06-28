"""``graphnav_builder`` 的 ROS 2 启动描述。

参数默认从 YAML 读取，使部署配置只有一个事实来源；启动参数只负责命名空间、
时钟、日志等级、配置路径和全局坐标系等典型运行时覆盖项。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """构建 ``graphnav_builder`` 节点的 ROS 2 启动描述。"""
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')
    log_level = LaunchConfiguration('log_level')
    global_frame = LaunchConfiguration('global_frame')

    # 使用包 share 目录，而不是工作目录相对路径，以支持 install 后运行。
    default_config_file = PathJoinSubstitution([
        FindPackageShare('graphnav_builder'),
        'config',
        'graphnav_builder.yaml',
    ])

    return LaunchDescription([
        # 命名空间会与 YAML 中的相对话题名共同决定最终 ROS 话题路径。
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='机器人命名空间；会前缀化相对话题名',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='为 true 时使用仿真时钟',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='graphnav_builder ROS 参数 YAML 文件路径',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='INFO',
            description='日志等级：DEBUG、INFO、WARN、ERROR 或 FATAL',
        ),
        DeclareLaunchArgument(
            'global_frame',
            default_value='odom',
            description='所有发布导航图数据使用的全局坐标系',
        ),
        # 将命令行覆盖项作为第二个参数字典，优先级高于 YAML 默认配置。
        Node(
            package='graphnav_builder',
            executable='graphnav_builder',
            name='graphnav_builder',
            namespace=namespace,
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[
                config_file,
                {
                    'use_sim_time': use_sim_time,
                    'global_frame': global_frame,
                },
            ],
        ),
    ])
