from pathlib import Path


LAUNCH_PATH = (
    Path(__file__).resolve().parents[1]
    / 'launch'
    / 'graphnav_nav2.launch.py'
)


def test_nav2_does_not_consume_high_rate_goal_pose_directly():
    launch_source = LAUNCH_PATH.read_text(encoding='utf-8')

    assert "SetRemap(src='goal_pose', dst='nav2_goal_pose_unused')" in launch_source


def test_bridge_defaults_are_responsive_for_moving_lookahead_goals():
    launch_source = LAUNCH_PATH.read_text(encoding='utf-8')

    assert "DeclareLaunchArgument('min_update_distance', default_value='0.8')" in launch_source
    assert "DeclareLaunchArgument('min_update_yaw', default_value='0.45')" in launch_source
    assert "DeclareLaunchArgument('min_update_period', default_value='0.5')" in launch_source
    assert "DeclareLaunchArgument('urgent_update_distance', default_value='1.6')" in launch_source
    assert "DeclareLaunchArgument('urgent_update_yaw', default_value='0.9')" in launch_source
    assert "DeclareLaunchArgument('urgent_update_period', default_value='0.2')" in launch_source


def test_nav2_uses_fast_replanning_behavior_tree():
    launch_source = LAUNCH_PATH.read_text(encoding='utf-8')

    assert 'navigate_to_pose_w_fast_replanning_and_recovery.xml' in launch_source
    assert "'default_nav_to_pose_bt_xml': bt_xml_file" in launch_source


def test_old_bridge_starts_dlio_twist_adapter_for_nav2():
    launch_source = LAUNCH_PATH.read_text(encoding='utf-8')

    assert "package='dlio_odom_twist_adapter'" in launch_source
    assert "executable='dlio_odom_twist_adapter'" in launch_source
    assert "'odom_topic': control_odom_topic" in launch_source
    assert "default_value='/dlio/odom_node/odom_body_twist'" in launch_source
