"""启动 A300 实际行驶轨迹记录节点。"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    """创建 launch 描述，并允许在命令行替换参数文件和仿真时间设置。"""

    # 默认配置安装在本包 share/config 目录中。
    package_share = Path(get_package_share_directory('a300_path_visualization'))
    default_params = package_share / 'config' / 'a300_path_visualization.yaml'

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        # 可通过 params_file:=... 使用另一套话题或采样参数。
        DeclareLaunchArgument(
            'params_file',
            default_value=str(default_params),
            description='A300 实际轨迹记录节点的 YAML 参数文件',
        ),

        # Gazebo 默认使用仿真时间；连接实车时可传 use_sim_time:=false。
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='是否使用 Gazebo 发布的 /clock 仿真时间',
        ),

        Node(
            package='a300_path_visualization',
            executable='a300_path_recorder',
            name='a300_path_recorder',
            output='screen',
            parameters=[
                params_file,
                # launch 参数放在 YAML 后面，因此命令行值可覆盖 YAML 默认值。
                {'use_sim_time': use_sim_time},
            ],
        ),
    ])
