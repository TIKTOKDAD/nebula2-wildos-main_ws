"""在三角定位给出真实目标前，先使用配置中的初始粗目标."""

import math

from geometry_msgs.msg import PoseStamped, Quaternion
import rclpy
from rclpy.node import Node


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """将平面偏航角转换为四元数."""
    half_yaw = 0.5 * yaw
    return Quaternion(z=math.sin(half_yaw), w=math.cos(half_yaw))


class InitialGoalMux(Node):
    """先发布一次粗目标，随后转发所有三角定位目标."""

    def __init__(self) -> None:
        super().__init__('initial_goal_mux')

        # 输入话题接收三角定位结果；输出话题提供给 graphnav_planner。
        self.declare_parameter(
            'real_goal_topic', '/triangulated_imgnav_waypoint')
        self.declare_parameter('output_goal_topic', '/imgnav_waypoint')

        # 在首次收到三角定位结果之前使用的初始粗目标。
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('initial_goal_x', 100.0)
        self.declare_parameter('initial_goal_y', 0.0)
        self.declare_parameter('initial_goal_z', 0.0)
        self.declare_parameter('initial_goal_yaw', 0.0)

        # 这里只控制等待订阅者时的检查频率，不会循环发布粗目标。
        self.declare_parameter('publish_rate', 1.0)

        # 读取 launch 传入的 ROS 参数；未传入时使用上面的默认值。
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

        # 输入输出不能是同一话题，否则节点会订阅到自己发布的消息，
        # 从而形成无限消息回环。
        if real_goal_topic == output_goal_topic:
            raise ValueError(
                'real_goal_topic and output_goal_topic must be different '
                'to avoid a feedback loop')

        # 一旦切换到真实目标便保持该状态，不会因目标暂时丢失而回退。
        self.using_real_goal = False
        self.initial_goal_published = False

        # /imgnav_waypoint 的发布者。
        self.goal_pub = self.create_publisher(
            PoseStamped, output_goal_topic, 10)

        # 订阅三角定位产生的原始目标点。
        self.goal_sub = self.create_subscription(
            PoseStamped,
            real_goal_topic,
            self.real_goal_callback,
            10,
        )

        # 等待 graphnav_planner 建立订阅后，再发布一次初始目标，避免消息
        # 在规划器启动前丢失。
        self.timer = self.create_timer(
            1.0 / publish_rate, self.publish_initial_goal_once)

        self.get_logger().info(
            'Waiting to publish initial coarse goal '
            f'({self.initial_goal_x:.2f}, {self.initial_goal_y:.2f}, '
            f'{self.initial_goal_z:.2f}) in {self.frame_id}; '
            f'triangulated goals from {real_goal_topic} will replace it on '
            f'{output_goal_topic}.')

    def real_goal_callback(self, msg: PoseStamped) -> None:
        """永久切换到三角定位目标，并将其转发给规划器."""
        position = msg.pose.position

        # 拒绝 NaN 或无穷大坐标，防止无效目标进入规划器。
        if not all(math.isfinite(value) for value in (
                position.x, position.y, position.z)):
            self.get_logger().warning(
                'Ignoring triangulated goal with non-finite coordinates.')
            return

        if not self.using_real_goal:
            self.get_logger().info(
                'Received triangulated object goal; replacing initial goal.')

        # 第一条有效三角定位消息会自动覆盖初始粗目标；后续消息继续更新
        # 真实目标位置。
        self.using_real_goal = True
        self.initial_goal_published = True
        self.stop_initial_timer()

        goal = PoseStamped()
        goal.header = msg.header
        goal.pose = msg.pose
        self.goal_pub.publish(goal)

    def make_initial_goal(self) -> PoseStamped:
        """根据配置构造初始粗目标消息."""
        goal = PoseStamped()
        goal.header.frame_id = self.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = self.initial_goal_x
        goal.pose.position.y = self.initial_goal_y
        goal.pose.position.z = self.initial_goal_z
        goal.pose.orientation = yaw_to_quaternion(self.initial_goal_yaw)
        return goal

    def stop_initial_timer(self) -> None:
        """停止等待发布初始目标."""
        if self.timer is not None:
            self.timer.cancel()
            self.destroy_timer(self.timer)
            self.timer = None

    def publish_initial_goal_once(self) -> None:
        """输出话题出现订阅者后，只发布一次初始粗目标."""
        if self.initial_goal_published or self.using_real_goal:
            self.stop_initial_timer()
            return

        # graphnav_planner 尚未订阅时继续等待，不提前丢出一次性消息。
        if self.goal_pub.get_subscription_count() == 0:
            return

        self.goal_pub.publish(self.make_initial_goal())
        self.initial_goal_published = True
        self.stop_initial_timer()
        self.get_logger().info('Published initial coarse goal once.')


def main(args=None) -> None:
    """运行初始目标选择节点."""
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
