#!/usr/bin/env python3
"""DLIO 世界系 twist 到 base_link 车体系 twist 的 ROS 2 适配节点."""

import math
import time

from dlio_odom_twist_adapter.twist_transform import odometry_with_body_twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


class DlioOdomTwistAdapter(Node):
    """按输入频率修正 DLIO Odometry 的 twist 坐标表达."""

    def __init__(self) -> None:
        super().__init__("dlio_odom_twist_adapter")

        # 输入保留原始 DLIO 话题；输出使用新话题，禁止输入输出同名形成回环。
        self.declare_parameter("input_odom_topic", "/dlio/odom_node/odom")
        self.declare_parameter(
            "output_odom_topic",
            "/dlio/odom_node/odom_body_twist",
        )
        # 输出 twist 的目标坐标系。A300/Nav2 的车体坐标系为 base_link。
        self.declare_parameter("output_child_frame_id", "base_link")
        # 只用于部署检查，不参与旋转计算；实际旋转由消息中的 pose 四元数决定。
        self.declare_parameter("expected_world_frame", "odom")
        self.declare_parameter("strict_frame_check", False)
        # 较大的输入队列可吸收短时间调度抖动，但节点始终逐消息立即转换发布。
        self.declare_parameter("qos_depth", 50)
        # 周期诊断用于确认转弯时世界 y 速度被正确转换成车体 x 速度。
        self.declare_parameter("diagnostic_log_period", 5.0)

        self.input_topic = str(self.get_parameter("input_odom_topic").value)
        self.output_topic = str(self.get_parameter("output_odom_topic").value)
        self.output_child_frame = str(
            self.get_parameter("output_child_frame_id").value
        )
        self.expected_world_frame = str(
            self.get_parameter("expected_world_frame").value
        )
        self.strict_frame_check = bool(
            self.get_parameter("strict_frame_check").value
        )
        qos_depth = int(self.get_parameter("qos_depth").value)
        self.diagnostic_log_period = float(
            self.get_parameter("diagnostic_log_period").value
        )

        if not self.input_topic or not self.output_topic:
            raise ValueError("input_odom_topic and output_odom_topic must not be empty")
        if self.input_topic == self.output_topic:
            raise ValueError("input and output odometry topics must be different")
        if not self.output_child_frame:
            raise ValueError("output_child_frame_id must not be empty")
        if qos_depth <= 0:
            raise ValueError("qos_depth must be positive")
        if self.diagnostic_log_period < 0.0:
            raise ValueError("diagnostic_log_period must be non-negative")

        # BEST_EFFORT 输入可以同时连接 BEST_EFFORT 和 RELIABLE 的 DLIO 发布端。
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # 输出使用 RELIABLE，兼容 Nav2 默认的里程计订阅可靠性要求。
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            Odometry,
            self.output_topic,
            output_qos,
        )
        self.subscription = self.create_subscription(
            Odometry,
            self.input_topic,
            self.odom_callback,
            input_qos,
        )

        self.last_diagnostic_time = float("-inf")
        self.last_warning_times: dict[str, float] = {}
        self.get_logger().info(
            f"Converting DLIO twist {self.input_topic} -> {self.output_topic}; "
            f"output child_frame_id={self.output_child_frame}"
        )

    def warn_throttled(self, key: str, message: str, period: float = 5.0) -> None:
        """按问题类型节流告警，避免每个高频 Odometry 都重复刷屏."""
        now = time.monotonic()
        if now - self.last_warning_times.get(key, float("-inf")) < period:
            return
        self.last_warning_times[key] = now
        self.get_logger().warning(message)

    def odom_callback(self, message: Odometry) -> None:
        """转换并发布一条 Odometry；无效四元数不会产生错误速度."""
        if (
            self.expected_world_frame
            and message.header.frame_id != self.expected_world_frame
        ):
            warning = (
                f"Expected input frame '{self.expected_world_frame}', got "
                f"'{message.header.frame_id}'"
            )
            self.warn_throttled("world_frame", warning)
            if self.strict_frame_check:
                return

        if message.child_frame_id and message.child_frame_id != self.output_child_frame:
            self.warn_throttled(
                "child_frame",
                f"Input child_frame_id is '{message.child_frame_id}', overriding it "
                f"with '{self.output_child_frame}' after twist conversion",
            )

        try:
            output = odometry_with_body_twist(
                message,
                output_child_frame_id=self.output_child_frame,
            )
        except ValueError as exc:
            self.warn_throttled("invalid_message", f"Dropping invalid odometry: {exc}")
            return

        self.publisher.publish(output)
        self.publish_periodic_diagnostic(message, output)

    def publish_periodic_diagnostic(
        self,
        source: Odometry,
        output: Odometry,
    ) -> None:
        """低频记录转换前后速度，便于现场确认坐标方向."""
        if self.diagnostic_log_period <= 0.0:
            return
        now = time.monotonic()
        if now - self.last_diagnostic_time < self.diagnostic_log_period:
            return
        self.last_diagnostic_time = now

        source_linear = source.twist.twist.linear
        body_linear = output.twist.twist.linear
        source_speed = math.sqrt(
            source_linear.x * source_linear.x
            + source_linear.y * source_linear.y
            + source_linear.z * source_linear.z
        )
        self.get_logger().info(
            "twist world=(%.3f, %.3f, %.3f), body=(%.3f, %.3f, %.3f), "
            "speed_norm=%.3f m/s"
            % (
                source_linear.x,
                source_linear.y,
                source_linear.z,
                body_linear.x,
                body_linear.y,
                body_linear.z,
                source_speed,
            )
        )


def main(args=None) -> None:
    """初始化 ROS 2，运行转换节点并可靠释放资源."""
    rclpy.init(args=args)
    node = DlioOdomTwistAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        # SIGTERM/launch 关闭可能已经结束 context；只在仍运行时主动关闭一次。
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
