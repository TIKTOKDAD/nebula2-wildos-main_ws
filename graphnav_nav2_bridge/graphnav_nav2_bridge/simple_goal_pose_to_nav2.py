#!/usr/bin/env python3
"""Simple PoseStamped-to-Nav2 NavigateToPose bridge.

This intentionally small bridge is useful for quick experiments. The main
``goal_pose_to_nav2`` executable contains additional protections for moving
look-ahead goals, stale input, yaw filtering, and result tracking.
"""

import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


class GoalPoseToNav2(Node):
    def __init__(self):
        super().__init__("simple_goal_pose_to_nav2")

        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter("min_update_dist", 0.5)
        self.declare_parameter("min_update_sec", 2.0)

        goal_topic = self.get_parameter("goal_topic").value
        action_name = self.get_parameter("action_name").value

        self.min_update_dist = float(
            self.get_parameter("min_update_dist").value
        )
        self.min_update_sec = float(
            self.get_parameter("min_update_sec").value
        )

        self.client = ActionClient(self, NavigateToPose, action_name)
        self.sub = self.create_subscription(
            PoseStamped,
            goal_topic,
            self.goal_callback,
            10,
        )

        self.last_goal = None
        self.last_send_time = 0.0

        self.get_logger().info(f"Subscribing: {goal_topic}")
        self.get_logger().info(f"Sending goals to: {action_name}")

    def distance(self, a: PoseStamped, b: PoseStamped):
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        return math.sqrt(dx * dx + dy * dy)

    def goal_callback(self, msg: PoseStamped):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_goal is not None:
            dist = self.distance(msg, self.last_goal)
            if (
                dist < self.min_update_dist
                and (now - self.last_send_time) < self.min_update_sec
            ):
                return

        if not self.client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warn(
                "Nav2 /navigate_to_pose action server not ready"
            )
            return

        goal = NavigateToPose.Goal()
        goal.pose = msg

        self.client.send_goal_async(goal)
        self.last_goal = msg
        self.last_send_time = now

        self.get_logger().info(
            f"Sent Nav2 goal: frame={msg.header.frame_id}, "
            f"x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}"
        )


def main():
    rclpy.init()
    node = GoalPoseToNav2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
