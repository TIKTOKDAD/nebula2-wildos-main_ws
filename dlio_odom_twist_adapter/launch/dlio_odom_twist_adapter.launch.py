"""启动 DLIO Odometry twist 坐标转换节点."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """声明可覆盖接口参数，并创建单个转换节点."""
    package_share = get_package_share_directory("dlio_odom_twist_adapter")

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    input_odom_topic = LaunchConfiguration("input_odom_topic")
    output_odom_topic = LaunchConfiguration("output_odom_topic")
    output_child_frame_id = LaunchConfiguration("output_child_frame_id")
    expected_world_frame = LaunchConfiguration("expected_world_frame")
    strict_frame_check = LaunchConfiguration("strict_frame_check")
    log_level = LaunchConfiguration("log_level")

    declarations = [
        # 仿真默认使用 Gazebo 时钟；实车应显式传 use_sim_time:=false。
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "params_file",
            default_value=f"{package_share}/config/dlio_odom_twist_adapter.yaml",
        ),
        DeclareLaunchArgument(
            "input_odom_topic",
            default_value="/dlio/odom_node/odom",
        ),
        DeclareLaunchArgument(
            "output_odom_topic",
            default_value="/dlio/odom_node/odom_body_twist",
        ),
        DeclareLaunchArgument("output_child_frame_id", default_value="base_link"),
        DeclareLaunchArgument("expected_world_frame", default_value="odom"),
        DeclareLaunchArgument("strict_frame_check", default_value="false"),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]

    adapter = Node(
        package="dlio_odom_twist_adapter",
        executable="dlio_odom_twist_adapter",
        name="dlio_odom_twist_adapter",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "input_odom_topic": input_odom_topic,
                "output_odom_topic": output_odom_topic,
                "output_child_frame_id": output_child_frame_id,
                "expected_world_frame": expected_world_frame,
                "strict_frame_check": strict_frame_check,
            },
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )
    return LaunchDescription(declarations + [adapter])
