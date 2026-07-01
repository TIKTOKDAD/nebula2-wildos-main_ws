#!/usr/bin/env python3
"""Forward filtered PoseStamped look-ahead goals to Nav2 NavigateToPose."""

from copy import deepcopy
import math
from typing import Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


def quaternion_yaw(message: PoseStamped) -> float:
    """Return the planar yaw represented by a PoseStamped quaternion."""
    quaternion = message.pose.orientation
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        ),
    )


def angular_distance(first: float, second: float) -> float:
    """Return the unsigned shortest angular distance in radians."""
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def planar_distance(first: PoseStamped, second: PoseStamped) -> float:
    """Return XY distance between two goals."""
    return math.hypot(
        first.pose.position.x - second.pose.position.x,
        first.pose.position.y - second.pose.position.y,
    )


def goal_changed(
    candidate: PoseStamped,
    previous: PoseStamped,
    minimum_distance: float,
    minimum_yaw: float,
) -> bool:
    """Report whether a candidate differs enough to replace a Nav2 goal."""
    if candidate.header.frame_id != previous.header.frame_id:
        return True
    return (
        planar_distance(candidate, previous) >= minimum_distance
        or angular_distance(
            quaternion_yaw(candidate), quaternion_yaw(previous)
        ) >= minimum_yaw
    )


def flatten_goal(message: PoseStamped) -> PoseStamped:
    """Copy a 3-D graph goal into the planar form expected by Nav2."""
    flattened = deepcopy(message)
    yaw = quaternion_yaw(flattened)
    flattened.pose.position.z = 0.0
    flattened.pose.orientation.x = 0.0
    flattened.pose.orientation.y = 0.0
    flattened.pose.orientation.z = math.sin(yaw * 0.5)
    flattened.pose.orientation.w = math.cos(yaw * 0.5)
    return flattened


class GoalPoseToNav2(Node):
    """Rate-limit moving look-ahead goals and send them to NavigateToPose."""

    def __init__(self) -> None:
        super().__init__('goal_pose_to_nav2')

        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter(
            'action_name', '/graphnav_navigate_to_pose'
        )
        self.declare_parameter('min_update_distance', 0.5)
        self.declare_parameter('min_update_yaw', 0.35)
        self.declare_parameter('min_update_period', 1.0)
        self.declare_parameter('goal_input_timeout', 2.0)
        self.declare_parameter('server_retry_period', 0.5)
        self.declare_parameter('flatten_to_2d', True)

        goal_topic = str(self.get_parameter('goal_topic').value)
        action_name = str(self.get_parameter('action_name').value)
        self.min_update_distance = float(
            self.get_parameter('min_update_distance').value
        )
        self.min_update_yaw = float(
            self.get_parameter('min_update_yaw').value
        )
        self.min_update_period = float(
            self.get_parameter('min_update_period').value
        )
        self.goal_input_timeout = float(
            self.get_parameter('goal_input_timeout').value
        )
        retry_period = float(
            self.get_parameter('server_retry_period').value
        )
        self.flatten_to_2d = bool(
            self.get_parameter('flatten_to_2d').value
        )

        if self.min_update_distance < 0.0:
            raise ValueError('min_update_distance must be non-negative')
        if self.min_update_yaw < 0.0:
            raise ValueError('min_update_yaw must be non-negative')
        if self.min_update_period < 0.0:
            raise ValueError('min_update_period must be non-negative')
        if self.goal_input_timeout <= 0.0:
            raise ValueError('goal_input_timeout must be positive')
        if retry_period <= 0.0:
            raise ValueError('server_retry_period must be positive')

        self.client = ActionClient(self, NavigateToPose, action_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            PoseStamped, goal_topic, self.goal_callback, qos
        )
        self.retry_timer = self.create_timer(retry_period, self.timer_callback)

        self.pending_goal: Optional[PoseStamped] = None
        self.last_sent_goal: Optional[PoseStamped] = None
        self.last_send_time = float('-inf')
        self.last_input_time: Optional[float] = None
        self.send_in_progress = False
        self.current_sequence = 0
        self.current_goal_handle = None
        self.cancel_requested = False
        self.server_warning_printed = False

        self.get_logger().info(
            f'Listening on {goal_topic}; forwarding to {action_name}; '
            f'distance threshold={self.min_update_distance:.2f} m, '
            f'period={self.min_update_period:.2f} s'
        )

    def now_seconds(self) -> float:
        """Return this node's clock in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def valid_goal(self, goal: PoseStamped) -> bool:
        """Reject goals that Nav2 cannot safely interpret."""
        values = (
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z,
            goal.pose.orientation.x,
            goal.pose.orientation.y,
            goal.pose.orientation.z,
            goal.pose.orientation.w,
        )
        if not goal.header.frame_id:
            self.get_logger().warning('Ignoring goal with an empty frame_id')
            return False
        if not all(math.isfinite(value) for value in values):
            self.get_logger().warning('Ignoring goal containing NaN or infinity')
            return False
        quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
        if quaternion_norm < 1e-6:
            self.get_logger().warning('Ignoring goal with a zero quaternion')
            return False
        return True

    def goal_callback(self, message: PoseStamped) -> None:
        """Keep only the newest materially different look-ahead goal."""
        if not self.valid_goal(message):
            return
        self.last_input_time = self.now_seconds()
        candidate = (
            flatten_goal(message)
            if self.flatten_to_2d
            else deepcopy(message)
        )

        reference = self.pending_goal or self.last_sent_goal
        if reference is not None and not goal_changed(
            candidate,
            reference,
            self.min_update_distance,
            self.min_update_yaw,
        ):
            return

        self.pending_goal = candidate
        self.try_send()

    def timer_callback(self) -> None:
        """Run stale-input protection and retry pending action goals."""
        self.cancel_if_input_stale()
        self.try_send()

    def cancel_if_input_stale(self) -> None:
        """Cancel motion when the upstream look-ahead stream stops."""
        if (
            self.current_goal_handle is None
            or self.cancel_requested
            or self.last_input_time is None
        ):
            return
        age = self.now_seconds() - self.last_input_time
        if age <= self.goal_input_timeout:
            return
        self.cancel_requested = True
        self.current_goal_handle.cancel_goal_async()
        self.get_logger().warning(
            f'No goal input for {age:.2f} s; canceling active Nav2 goal'
        )

    def try_send(self) -> None:
        """Send the pending goal when Nav2 and the rate limiter are ready."""
        if self.pending_goal is None or self.send_in_progress:
            return
        if self.now_seconds() - self.last_send_time < self.min_update_period:
            return
        if not self.client.server_is_ready():
            if not self.server_warning_printed:
                self.get_logger().warning(
                    'NavigateToPose server is not ready; retaining latest goal'
                )
                self.server_warning_printed = True
            return

        self.server_warning_printed = False
        candidate = self.pending_goal
        self.pending_goal = None
        self.last_sent_goal = deepcopy(candidate)
        self.last_send_time = self.now_seconds()
        self.current_sequence += 1
        sequence = self.current_sequence
        self.send_in_progress = True

        goal = NavigateToPose.Goal()
        goal.pose = candidate
        future = self.client.send_goal_async(goal)
        future.add_done_callback(
            lambda completed, seq=sequence: self.goal_response_callback(
                completed, seq
            )
        )
        self.get_logger().info(
            f'Sending Nav2 goal #{sequence}: frame={candidate.header.frame_id}, '
            f'x={candidate.pose.position.x:.2f}, '
            f'y={candidate.pose.position.y:.2f}'
        )

    def goal_response_callback(self, future, sequence: int) -> None:
        """Track acceptance without blocking incoming look-ahead updates."""
        self.send_in_progress = False
        try:
            goal_handle = future.result()
        except Exception as error:  # rclpy futures propagate transport failures
            self.get_logger().error(
                f'Failed to send Nav2 goal #{sequence}: {error}'
            )
            if sequence == self.current_sequence:
                self.last_sent_goal = None
            self.try_send()
            return

        if not goal_handle.accepted:
            self.get_logger().warning(f'Nav2 rejected goal #{sequence}')
            if sequence == self.current_sequence:
                self.last_sent_goal = None
            self.try_send()
            return

        if sequence == self.current_sequence:
            self.current_goal_handle = goal_handle
            self.cancel_requested = False
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed, seq=sequence: self.result_callback(completed, seq)
        )
        self.try_send()

    def result_callback(self, future, sequence: int) -> None:
        """Report the current Nav2 result and permit retry after failures."""
        try:
            wrapped_result = future.result()
            status = wrapped_result.status
            error_code = wrapped_result.result.error_code
            error_message = wrapped_result.result.error_msg
        except Exception as error:  # rclpy futures propagate transport failures
            self.get_logger().error(
                f'Could not read Nav2 result for goal #{sequence}: {error}'
            )
            status = GoalStatus.STATUS_ABORTED
            error_code = 100
            error_message = str(error)

        if sequence != self.current_sequence:
            return

        self.current_goal_handle = None
        self.cancel_requested = False
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Nav2 goal #{sequence} succeeded')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'Nav2 goal #{sequence} was replaced/canceled')
            self.last_sent_goal = None
        else:
            self.get_logger().warning(
                f'Nav2 goal #{sequence} failed: status={status}, '
                f"error_code={error_code}, message={error_message or 'n/a'}"
            )
            self.last_sent_goal = None
        self.try_send()


def main(args=None) -> None:
    """Run the bridge node."""
    rclpy.init(args=args)
    node = GoalPoseToNav2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
