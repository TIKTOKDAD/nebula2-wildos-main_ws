"""Select an initial coarse goal until object triangulation provides a goal."""

import math

from geometry_msgs.msg import PoseStamped, Quaternion
import rclpy
from rclpy.node import Node


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert a planar yaw angle to a quaternion."""
    half_yaw = 0.5 * yaw
    return Quaternion(z=math.sin(half_yaw), w=math.cos(half_yaw))


class InitialGoalMux(Node):
    """Publish one coarse goal, then forward every triangulated object goal."""

    def __init__(self) -> None:
        super().__init__('initial_goal_mux')

        self.declare_parameter(
            'real_goal_topic', '/triangulated_imgnav_waypoint')
        self.declare_parameter('output_goal_topic', '/imgnav_waypoint')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('initial_goal_x', 100.0)
        self.declare_parameter('initial_goal_y', 0.0)
        self.declare_parameter('initial_goal_z', 0.0)
        self.declare_parameter('initial_goal_yaw', 0.0)
        self.declare_parameter('publish_rate', 1.0)

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.initial_goal_x = float(
            self.get_parameter('initial_goal_x').value)
        self.initial_goal_y = float(
            self.get_parameter('initial_goal_y').value)
        self.initial_goal_z = float(
            self.get_parameter('initial_goal_z').value)
        self.initial_goal_yaw = float(
            self.get_parameter('initial_goal_yaw').value)

        real_goal_topic = str(self.get_parameter('real_goal_topic').value)
        output_goal_topic = str(
            self.get_parameter('output_goal_topic').value)
        publish_rate = max(
            0.1, float(self.get_parameter('publish_rate').value))

        if real_goal_topic == output_goal_topic:
            raise ValueError(
                'real_goal_topic and output_goal_topic must be different '
                'to avoid a feedback loop')

        self.using_real_goal = False
        self.initial_goal_published = False

        self.goal_pub = self.create_publisher(
            PoseStamped, output_goal_topic, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            real_goal_topic,
            self.real_goal_callback,
            10,
        )
        self.timer = self.create_timer(
            1.0 / publish_rate, self.publish_initial_goal_once)

        self.get_logger().info(
            'Waiting to publish initial coarse goal '
            f'({self.initial_goal_x:.2f}, {self.initial_goal_y:.2f}, '
            f'{self.initial_goal_z:.2f}) in {self.frame_id}; '
            f'triangulated goals from {real_goal_topic} will replace it on '
            f'{output_goal_topic}.')

    def real_goal_callback(self, msg: PoseStamped) -> None:
        """Switch permanently to the triangulated goal and forward it."""
        position = msg.pose.position
        if not all(math.isfinite(value) for value in (
                position.x, position.y, position.z)):
            self.get_logger().warning(
                'Ignoring triangulated goal with non-finite coordinates.')
            return

        if not self.using_real_goal:
            self.get_logger().info(
                'Received triangulated object goal; replacing initial goal.')

        self.using_real_goal = True
        self.initial_goal_published = True
        self.stop_initial_timer()

        goal = PoseStamped()
        goal.header = msg.header
        goal.pose = msg.pose
        self.goal_pub.publish(goal)

    def make_initial_goal(self) -> PoseStamped:
        """Build the configured initial coarse goal."""
        goal = PoseStamped()
        goal.header.frame_id = self.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = self.initial_goal_x
        goal.pose.position.y = self.initial_goal_y
        goal.pose.position.z = self.initial_goal_z
        goal.pose.orientation = yaw_to_quaternion(self.initial_goal_yaw)
        return goal

    def stop_initial_timer(self) -> None:
        """Stop waiting to publish the initial goal."""
        if self.timer is not None:
            self.timer.cancel()
            self.destroy_timer(self.timer)
            self.timer = None

    def publish_initial_goal_once(self) -> None:
        """Publish the initial goal after its first subscriber appears."""
        if self.initial_goal_published or self.using_real_goal:
            self.stop_initial_timer()
            return

        if self.goal_pub.get_subscription_count() == 0:
            return

        self.goal_pub.publish(self.make_initial_goal())
        self.initial_goal_published = True
        self.stop_initial_timer()
        self.get_logger().info('Published initial coarse goal once.')


def main(args=None) -> None:
    """Run the initial-goal multiplexer node."""
    rclpy.init(args=args)
    node = InitialGoalMux()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
