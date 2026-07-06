from pathlib import Path


SETUP_PATH = Path(__file__).resolve().parents[1] / 'setup.py'


def test_simple_bridge_console_script_is_registered():
    setup_source = SETUP_PATH.read_text(encoding='utf-8')

    assert 'simple_goal_pose_to_nav2 = ' in setup_source
    assert 'graphnav_nav2_bridge.simple_goal_pose_to_nav2:main' in setup_source
