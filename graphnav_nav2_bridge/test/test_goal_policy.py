import math

from geometry_msgs.msg import PoseStamped

from graphnav_nav2_bridge.goal_pose_to_nav2 import flatten_goal
from graphnav_nav2_bridge.goal_pose_to_nav2 import goal_changed
from graphnav_nav2_bridge.goal_pose_to_nav2 import planar_distance


def make_goal(x=0.0, y=0.0, z=1.0, yaw=0.0):
    goal = PoseStamped()
    goal.header.frame_id = 'odom'
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation.z = math.sin(yaw * 0.5)
    goal.pose.orientation.w = math.cos(yaw * 0.5)
    return goal


def test_planar_distance_ignores_height():
    assert planar_distance(make_goal(z=0.0), make_goal(z=9.0)) == 0.0


def test_goal_change_uses_distance_or_yaw():
    previous = make_goal()
    assert not goal_changed(make_goal(x=0.49), previous, 0.5, 0.35)
    assert goal_changed(make_goal(x=0.5), previous, 0.5, 0.35)
    assert goal_changed(make_goal(yaw=0.35), previous, 0.5, 0.35)


def test_flatten_goal_preserves_xy_and_yaw():
    original = make_goal(x=2.0, y=-3.0, z=4.0, yaw=1.2)
    flattened = flatten_goal(original)
    assert flattened.pose.position.x == 2.0
    assert flattened.pose.position.y == -3.0
    assert flattened.pose.position.z == 0.0
    assert flattened.pose.orientation.x == 0.0
    assert flattened.pose.orientation.y == 0.0
    assert math.isclose(flattened.pose.orientation.z, math.sin(0.6))
    assert math.isclose(flattened.pose.orientation.w, math.cos(0.6))
    assert original.pose.position.z == 4.0
