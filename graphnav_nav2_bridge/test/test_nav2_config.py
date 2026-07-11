from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / 'config' / 'nav2_a300_odom.yaml'
)
BT_PATH = (
    Path(__file__).resolve().parents[1]
    / 'behavior_trees'
    / 'navigate_to_pose_w_fast_replanning_and_recovery.xml'
)


def test_a300_topics_and_frames_are_wired():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))

    bt = config['bt_navigator']['ros__parameters']
    assert bt['global_frame'] == 'odom'
    assert bt['robot_base_frame'] == 'base_link'
    corrected_odom = '/dlio/odom_node/odom_body_twist'
    assert bt['odom_topic'] == corrected_odom
    assert config['controller_server']['ros__parameters']['odom_topic'] == corrected_odom
    assert config['velocity_smoother']['ros__parameters']['odom_topic'] == corrected_odom

    scan_topic = '/a300_0000/sensors/lidar3d_0/scan'
    local = config['local_costmap']['local_costmap']['ros__parameters']
    global_costmap = config['global_costmap']['global_costmap']['ros__parameters']
    assert local['obstacle_layer']['scan']['topic'] == scan_topic
    assert global_costmap['obstacle_layer']['scan']['topic'] == scan_topic

    collision = config['collision_monitor']['ros__parameters']
    assert collision['scan']['topic'] == scan_topic
    assert collision['cmd_vel_out_topic'] == '/a300_0000/cmd_vel'


def test_velocity_chain_uses_twist_stamped():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    for node_name in (
        'controller_server',
        'behavior_server',
        'velocity_smoother',
        'collision_monitor',
        'docking_server',
    ):
        assert config[node_name]['ros__parameters']['enable_stamped_cmd_vel']


def test_lookahead_goals_use_loose_goal_tolerances():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    goal_checker = (
        config['controller_server']['ros__parameters']['general_goal_checker']
    )

    assert goal_checker['xy_goal_tolerance'] >= 0.75
    assert goal_checker['yaw_goal_tolerance'] >= 1.5


def test_progress_checker_accepts_in_place_rotation():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    progress_checker = (
        config['controller_server']['ros__parameters']['progress_checker']
    )

    assert progress_checker['plugin'] == 'nav2_controller::PoseProgressChecker'
    assert progress_checker['required_movement_angle'] > 0.0
    assert progress_checker['movement_time_allowance'] >= 20.0


def test_follow_path_rotates_to_path_heading_before_mppi():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    follow_path = config['controller_server']['ros__parameters']['FollowPath']

    assert follow_path['plugin'] == (
        'nav2_rotation_shim_controller::RotationShimController'
    )
    assert follow_path['primary_controller'] == (
        'nav2_mppi_controller::MPPIController'
    )
    assert follow_path['angular_dist_threshold'] > 0.0
    assert follow_path['rotate_to_heading_angular_vel'] > 0.0


def test_bt_navigator_declares_rewritable_forward_only_tree_key():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    bt_params = config['bt_navigator']['ros__parameters']

    assert 'default_nav_to_pose_bt_xml' in bt_params


def test_bt_navigator_only_starts_navigate_to_pose_for_graphnav_bridge():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    bt_params = config['bt_navigator']['ros__parameters']

    assert bt_params['navigators'] == ['navigate_to_pose']
    assert 'navigate_through_poses' not in bt_params


def test_a300_graphnav_nav2_stack_is_forward_only():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))

    follow_path = config['controller_server']['ros__parameters']['FollowPath']
    assert follow_path['vx_min'] >= 0.0
    assert follow_path['vx_max'] > 0.0

    velocity_smoother = config['velocity_smoother']['ros__parameters']
    assert velocity_smoother['min_velocity'][0] >= 0.0
    assert velocity_smoother['max_velocity'][0] > 0.0

    behavior_server = config['behavior_server']['ros__parameters']
    assert 'backup' not in behavior_server['behavior_plugins']

    behavior_tree = ET.parse(BT_PATH)
    assert not behavior_tree.findall('.//BackUp')


def test_high_speed_limits_are_consistent_across_control_chain():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))

    follow_path = config['controller_server']['ros__parameters']['FollowPath']
    velocity_smoother = config['velocity_smoother']['ros__parameters']

    assert follow_path['vx_max'] == 5.0
    assert follow_path['wz_max'] == 3.1
    assert velocity_smoother['max_velocity'] == [5.0, 0.0, 3.1]
    assert velocity_smoother['min_velocity'] == [0.0, 0.0, -3.1]

    local = config['local_costmap']['local_costmap']['ros__parameters']
    predicted_forward_distance = (
        follow_path['vx_max'] * follow_path['time_steps'] * follow_path['model_dt']
    )
    assert local['width'] / 2.0 > predicted_forward_distance
    assert local['obstacle_layer']['scan']['obstacle_max_range'] > (
        predicted_forward_distance
    )


def test_mppi_horizon_matches_seven_meter_lookahead_cruise_speed():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))

    follow_path = config['controller_server']['ros__parameters']['FollowPath']
    horizon = follow_path['time_steps'] * follow_path['model_dt']

    assert horizon == pytest.approx(2.35)
    assert 7.0 / horizon == pytest.approx(3.0, rel=0.01)
