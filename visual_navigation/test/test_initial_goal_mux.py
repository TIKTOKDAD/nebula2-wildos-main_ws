"""Tests for the initial coarse-goal multiplexer."""

import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from visual_navigation.initial_goal_mux import InitialGoalMux


def spin_until(executor, predicate, timeout=3.0):
    """Spin until a predicate becomes true or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_initial_goal_is_replaced_by_triangulated_goal():
    """The first triangulated goal must replace the configured coarse goal."""
    rclpy.init(args=[
        '--ros-args',
        '-p', 'real_goal_topic:=/test/triangulated_goal',
        '-p', 'output_goal_topic:=/test/active_goal',
        '-p', 'initial_goal_x:=12.5',
        '-p', 'initial_goal_y:=-4.0',
        '-p', 'publish_rate:=20.0',
    ])
    mux = InitialGoalMux()
    peer = Node('initial_goal_mux_test_peer')
    received = []
    peer.create_subscription(
        PoseStamped, '/test/active_goal', received.append, 10)
    real_goal_pub = peer.create_publisher(
        PoseStamped, '/test/triangulated_goal', 10)

    executor = SingleThreadedExecutor()
    executor.add_node(mux)
    executor.add_node(peer)

    try:
        assert spin_until(executor, lambda: len(received) >= 1)
        assert received[0].pose.position.x == 12.5
        assert received[0].pose.position.y == -4.0

        real_goal = PoseStamped()
        real_goal.header.frame_id = 'odom'
        real_goal.pose.position.x = 3.0
        real_goal.pose.position.y = 7.0
        real_goal.pose.orientation.w = 1.0

        assert spin_until(
            executor,
            lambda: real_goal_pub.get_subscription_count() > 0,
        )
        real_goal_pub.publish(real_goal)

        assert spin_until(executor, lambda: len(received) >= 2)
        assert received[-1].pose.position.x == 3.0
        assert received[-1].pose.position.y == 7.0
        assert mux.using_real_goal
    finally:
        executor.remove_node(peer)
        executor.remove_node(mux)
        peer.destroy_node()
        mux.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
