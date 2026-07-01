from pathlib import Path

import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / 'config' / 'nav2_a300_odom.yaml'
)


def test_a300_topics_and_frames_are_wired():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))

    bt = config['bt_navigator']['ros__parameters']
    assert bt['global_frame'] == 'odom'
    assert bt['robot_base_frame'] == 'base_link'
    assert bt['odom_topic'] == '/dlio/odom_node/odom'

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
