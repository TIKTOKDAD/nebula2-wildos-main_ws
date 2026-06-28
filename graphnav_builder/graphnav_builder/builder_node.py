"""将 WildOS 算法 1--5 接入 ROS 2 通信、时间同步和 TF 的适配节点。

本模块不在订阅回调中执行重型建图：回调只入队，定时器在获得与消息时间戳对应
的 TF 后顺序处理，从而兼顾吞吐、历史坐标变换正确性和可诊断的背压行为。
"""

import math
import time
from typing import Dict, Optional

from graphnav_builder.graph_builder import SparseGraphBuilder
from graphnav_builder.utils.global_memory import GlobalTraversabilityMemory
from graphnav_builder.utils.graph_messages import navigation_graph_message
from graphnav_builder.utils.message_buffer import MessageBuffer
from graphnav_builder.utils.transforms import (
    planar_transform_from_tf,
    PlanarTransform,
    transform_pose_position,
)
from graphnav_builder.utils.traversability_grid import TraversabilityGrid
from graphnav_builder.utils.viz import (
    global_memory_marker_array,
    local_classification_marker_array,
)
from graphnav_msgs.msg import NavigationGraph
from grid_map_msgs.msg import GridMap
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray


def qos_profile_from_parameters(
    reliability: str,
    durability: str,
    depth: int,
) -> QoSProfile:
    """根据可读字符串参数构造显式 ROS 2 QoS 配置。

    里程计常为 BEST_EFFORT，而地图常为 RELIABLE，故两条输入通道分别配置；
    非法字符串不静默回退，避免因 QoS 不匹配导致“看似正常但收不到消息”。
    """
    reliability_options = {
        'best_effort': ReliabilityPolicy.BEST_EFFORT,
        'reliable': ReliabilityPolicy.RELIABLE,
    }
    durability_options = {
        'volatile': DurabilityPolicy.VOLATILE,
        'transient_local': DurabilityPolicy.TRANSIENT_LOCAL,
    }
    if reliability not in reliability_options:
        raise ValueError(
            'QoS reliability must be best_effort or reliable'
        )
    if durability not in durability_options:
        raise ValueError(
            'QoS durability must be volatile or transient_local'
        )
    if depth <= 0:
        raise ValueError('QoS depth must be positive')
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=reliability_options[reliability],
        durability=durability_options[durability],
    )


class SparseGraphBuilderNode(Node):
    """同步 ROS 输入，维护持久稀疏图并发布 ``NavigationGraph``。

    生命周期内持有一个 ``SparseGraphBuilder``，所以局部 GridMap 移动时图状态
    不会重置。调试全局记忆为可选旁路，绝不参与算法的节点、边或代价决策。
    """

    def __init__(self):
        """初始化参数、纯算法状态、TF/消息管线、发布器和定时器。"""
        # 按“声明→读取→构造依赖→建立通信”的顺序初始化，参数只读一次。
        super().__init__('graphnav_builder')
        self.declare_graph_parameters()
        self.read_parameters()

        # 将 ROS 参数显式投影为纯算法对象，避免算法层依赖 rclpy。
        self.graph_builder = SparseGraphBuilder(
            max_free_radius=self.max_free_radius,
            traversable_radius=self.traversable_radius,
            edge_radius=self.edge_radius,
            num_samples=self.num_samples,
            random_seed=self.random_seed,
            sample_against_new_nodes=self.sample_against_new_nodes,
            frontier_connectivity=self.frontier_connectivity,
            validate_historical_edges=self.validate_historical_edges,
            ensure_robot_anchor=self.ensure_robot_anchor,
            edge_cost_mode=self.edge_cost_mode,
            traversability_cost_weight=self.traversability_cost_weight,
            min_x=self.min_x,
            max_x=self.max_x,
            min_y=self.min_y,
            max_y=self.max_y,
        )
        # TF 可能比消息晚到；缓冲区策略决定满载时偏向实时还是严格顺序。
        self.msg_buffer = MessageBuffer(
            max_size=self.message_buffer_size,
            wait_for_oldest=not self.drop_old_messages,
        )
        # 不需要 TF 的同坐标系部署不创建监听器，减少线程/资源开销。
        self.tf_buffer = (
            Buffer(cache_time=Duration(seconds=self.tf_cache_duration))
            if self.use_tf else None
        )
        self.tf_listener = (
            TransformListener(self.tf_buffer, self, spin_thread=False)
            if self.tf_buffer is not None else None
        )
        # 以下变量均为运行时诊断状态，不会改变论文算法的语义。
        self.global_memory: Optional[GlobalTraversabilityMemory] = None
        self.latest_debug_stamp = None
        self.last_local_classification_publish_time = -math.inf
        self.debug_limit_warned = False
        self.resolution_warning_emitted = False
        self.grid_contract_logged = False
        self.last_grid_geometry = None
        self.unknown_boundary_warning_emitted = False
        self.warning_timestamps = {}
        self.tf_timeout_drop_count = 0
        self.processed_message_count = 0
        self.last_graph_diagnostics_time = 0.0
        self.clbk_cntr = 0

        self.init_publishers()
        self.init_subscribers()
        # 定时处理把昂贵计算移出同步订阅回调，避免阻塞消息过滤器。
        self.processing_timer = self.create_timer(
            self.processing_timer_period,
            self.process_buffer,
        )
        self.debug_timer = (
            self.create_timer(
                self.global_memory_publish_period,
                self.publish_debug_outputs,
            )
            if self.publish_global_memory_markers else None
        )

        self.get_logger().info(
            'Initialized WildOS graph builder with '
            f'local_bounds={self.local_bounds_description()}, '
            f'N_samples={self.num_samples}, '
            f'r_trav={self.traversable_radius:.2f} m, '
            f'r_max^f={self.max_free_radius:.2f} m, '
            f'r_edge={self.edge_radius:.2f} m, '
            f'odom_qos={self.odom_qos_reliability}/'
            f'{self.odom_qos_depth}, '
            f'grid_qos={self.grid_qos_reliability}/'
            f'{self.grid_qos_depth}'
        )

    def declare_graph_parameters(self):
        """声明 ROS 接口、地图合同、算法、TF、诊断和调试参数。

        参数分组与 YAML 配置保持一致；默认值可安全启动，部署时可通过 launch/YAML
        覆盖。此处仅声明，类型转换与缓存统一在 ``read_parameters`` 中完成。
        """
        # 话题与全局坐标系：所有发布节点、边和前沿都使用 global_frame。
        self.declare_parameter('odometry_topic', 'odom')
        self.declare_parameter(
            'traversability_map_topic',
            'traversability_map',
        )
        self.declare_parameter('navigation_graph_topic', 'nav_graph')
        self.declare_parameter('traversability_class', 'default')
        self.declare_parameter('global_frame', 'odom')

        # GridMap 图层合同及数值到 FREE/OBSTACLE/UNKNOWN 的分类规则。
        self.declare_parameter('traversability_layer', 'traversability')
        self.declare_parameter('elevation_layer', 'elevation')
        self.declare_parameter('observed_layer', '')
        self.declare_parameter('observed_threshold', 0.5)
        self.declare_parameter('safe_threshold', 0.5)
        self.declare_parameter(
            'traversability_semantics',
            TraversabilityGrid.HIGHER_IS_SAFER,
        )
        self.declare_parameter(
            'unknown_value_policy',
            TraversabilityGrid.UNKNOWN_NON_FINITE,
        )
        self.declare_parameter('traversability_cost_layer', '')
        self.declare_parameter('cost_min', 0.0)
        self.declare_parameter('cost_max', 1.0)
        self.declare_parameter('cost_higher_is_riskier', True)
        self.declare_parameter('strict_cost_range', False)

        # 论文对应的局部可靠范围、采样、净空和连边参数。
        self.declare_parameter('local_map_radius', 10.0)
        self.declare_parameter('expected_map_resolution', 0.1)
        self.declare_parameter('resolution_tolerance', 0.001)
        self.declare_parameter('crop_to_local_radius', True)
        self.declare_parameter('strict_resolution_check', False)
        self.declare_parameter('max_free_radius', 4.0)
        self.declare_parameter('traversable_radius', 0.5)
        self.declare_parameter('edge_radius', 8.0)
        self.declare_parameter('num_samples', 1000)
        self.declare_parameter('sample_against_new_nodes', True)
        self.declare_parameter('frontier_connectivity', 4)
        self.declare_parameter('validate_historical_edges', True)
        self.declare_parameter('ensure_robot_anchor', True)
        self.declare_parameter('edge_cost_mode', 'euclidean')
        self.declare_parameter('traversability_cost_weight', 1.0)
        self.declare_parameter('random_seed', 7)

        # 两路输入可独立匹配其实际发布端的 QoS。
        self.declare_parameter('odom_qos_reliability', 'best_effort')
        self.declare_parameter('odom_qos_durability', 'volatile')
        self.declare_parameter('odom_qos_depth', 20)
        self.declare_parameter('grid_qos_reliability', 'reliable')
        self.declare_parameter('grid_qos_durability', 'volatile')
        self.declare_parameter('grid_qos_depth', 1)
        self.declare_parameter('syncsub_queue_size', 10)
        self.declare_parameter('syncsub_slop', 0.1)
        self.declare_parameter('message_buffer_size', 1)
        self.declare_parameter('drop_old_messages', True)
        self.declare_parameter('processing_timer_period', 0.05)
        self.declare_parameter('use_tf', True)
        self.declare_parameter('tf_cache_duration', 10.0)
        self.declare_parameter('tf_lookup_timeout', 0.0)
        self.declare_parameter('tf_message_max_age', 1.0)
        self.declare_parameter('tf_max_lookup_attempts', 20)

        # 对上游 GridMap 几何/前沿供给的运行时合同检查与告警阈值。
        self.declare_parameter('log_grid_map_contract', True)
        self.declare_parameter('expected_map_length_x', 0.0)
        self.declare_parameter('expected_map_length_y', 0.0)
        self.declare_parameter('map_length_tolerance', 0.01)
        self.declare_parameter('strict_map_size_check', False)
        self.declare_parameter('warn_if_no_unknown_boundary', True)
        self.declare_parameter('require_unknown_map_boundary', False)
        self.declare_parameter('warn_if_no_frontier_candidates', True)
        self.declare_parameter('require_frontier_candidates', False)

        self.declare_parameter('graph_diagnostics_period', 5.0)
        self.declare_parameter('warn_node_count', 5000)
        self.declare_parameter('warn_edge_count', 50000)
        self.declare_parameter('warn_update_duration', 0.5)

        # 限容量的 RViz 调试旁路；默认关闭，不影响正常建图性能。
        self.declare_parameter('publish_global_memory_markers', False)
        self.declare_parameter(
            'global_memory_marker_topic',
            'global_traversability_markers',
        )
        self.declare_parameter('global_memory_marker_stride', 2)
        self.declare_parameter('global_memory_publish_period', 1.0)
        self.declare_parameter('global_memory_max_cells', 500000)
        self.declare_parameter(
            'publish_local_classification_markers',
            False,
        )
        self.declare_parameter(
            'local_classification_marker_topic',
            'local_traversability_2d_markers',
        )
        self.declare_parameter(
            'local_classification_publish_period',
            0.5,
        )
        self.declare_parameter('local_classification_z_offset', 0.02)

        # 持久节点的可选世界坐标工作空间限制。
        self.declare_parameter('min_x', -math.inf)
        self.declare_parameter('max_x', math.inf)
        self.declare_parameter('min_y', -math.inf)
        self.declare_parameter('max_y', math.inf)

    def read_parameters(self):
        """读取已声明参数并转换为运行期使用的 Python 基本类型。

        使用局部 ``value`` 函数集中访问参数 API；之后的热路径只读取实例属性，
        避免每帧重复调用 rclpy 参数接口。
        """
        def value(name):
            """返回一个已声明 ROS 参数的原始值。"""
            return self.get_parameter(name).value

        # 话题名、图类别和发布坐标系。
        self.odometry_topic = value('odometry_topic')
        self.traversability_map_topic = value('traversability_map_topic')
        self.navigation_graph_topic = value('navigation_graph_topic')
        self.traversability_class = value('traversability_class')
        self.global_frame = value('global_frame')

        # 栅格解码与可选风险成本层的合同。
        self.traversability_layer = value('traversability_layer')
        self.elevation_layer = value('elevation_layer')
        self.observed_layer = value('observed_layer')
        self.observed_threshold = float(value('observed_threshold'))
        self.safe_threshold = float(value('safe_threshold'))
        self.traversability_semantics = value('traversability_semantics')
        self.unknown_value_policy = value('unknown_value_policy')
        self.traversability_cost_layer = value(
            'traversability_cost_layer'
        )
        self.cost_min = float(value('cost_min'))
        self.cost_max = float(value('cost_max'))
        self.cost_higher_is_riskier = bool(
            value('cost_higher_is_riskier')
        )
        self.strict_cost_range = bool(value('strict_cost_range'))

        # 建图几何与算法参数。
        self.local_map_radius = float(value('local_map_radius'))
        self.expected_map_resolution = float(
            value('expected_map_resolution')
        )
        self.resolution_tolerance = float(value('resolution_tolerance'))
        self.crop_to_local_radius = bool(value('crop_to_local_radius'))
        self.strict_resolution_check = bool(
            value('strict_resolution_check')
        )
        self.max_free_radius = float(value('max_free_radius'))
        self.traversable_radius = float(value('traversable_radius'))
        self.edge_radius = float(value('edge_radius'))
        self.num_samples = int(value('num_samples'))
        self.sample_against_new_nodes = bool(
            value('sample_against_new_nodes')
        )
        self.frontier_connectivity = int(value('frontier_connectivity'))
        self.validate_historical_edges = bool(
            value('validate_historical_edges')
        )
        self.ensure_robot_anchor = bool(value('ensure_robot_anchor'))
        self.edge_cost_mode = value('edge_cost_mode')
        self.traversability_cost_weight = float(
            value('traversability_cost_weight')
        )
        self.random_seed = int(value('random_seed'))

        # 同步、缓冲与历史 TF 参数。
        self.odom_qos_reliability = value('odom_qos_reliability')
        self.odom_qos_durability = value('odom_qos_durability')
        self.odom_qos_depth = int(value('odom_qos_depth'))
        self.grid_qos_reliability = value('grid_qos_reliability')
        self.grid_qos_durability = value('grid_qos_durability')
        self.grid_qos_depth = int(value('grid_qos_depth'))
        self.syncsub_queue_size = int(value('syncsub_queue_size'))
        self.syncsub_slop = float(value('syncsub_slop'))
        self.message_buffer_size = int(value('message_buffer_size'))
        self.drop_old_messages = bool(value('drop_old_messages'))
        self.processing_timer_period = float(
            value('processing_timer_period')
        )
        self.use_tf = bool(value('use_tf'))
        self.tf_cache_duration = float(value('tf_cache_duration'))
        self.tf_lookup_timeout = float(value('tf_lookup_timeout'))
        self.tf_message_max_age = float(value('tf_message_max_age'))
        self.tf_max_lookup_attempts = int(
            value('tf_max_lookup_attempts')
        )

        # 输入合同、规模和延迟诊断参数。
        self.log_grid_map_contract = bool(value('log_grid_map_contract'))
        self.expected_map_length_x = float(value('expected_map_length_x'))
        self.expected_map_length_y = float(value('expected_map_length_y'))
        self.map_length_tolerance = float(value('map_length_tolerance'))
        self.strict_map_size_check = bool(value('strict_map_size_check'))
        self.warn_if_no_unknown_boundary = bool(
            value('warn_if_no_unknown_boundary')
        )
        self.require_unknown_map_boundary = bool(
            value('require_unknown_map_boundary')
        )
        self.warn_if_no_frontier_candidates = bool(
            value('warn_if_no_frontier_candidates')
        )
        self.require_frontier_candidates = bool(
            value('require_frontier_candidates')
        )
        self.graph_diagnostics_period = float(
            value('graph_diagnostics_period')
        )
        self.warn_node_count = int(value('warn_node_count'))
        self.warn_edge_count = int(value('warn_edge_count'))
        self.warn_update_duration = float(
            value('warn_update_duration')
        )

        # 可选稀疏调试记忆和工作空间限制。
        self.publish_global_memory_markers = bool(
            value('publish_global_memory_markers')
        )
        self.global_memory_marker_topic = value(
            'global_memory_marker_topic'
        )
        self.global_memory_marker_stride = max(
            1,
            int(value('global_memory_marker_stride')),
        )
        self.global_memory_publish_period = float(
            value('global_memory_publish_period')
        )
        self.global_memory_max_cells = int(
            value('global_memory_max_cells')
        )
        self.publish_local_classification_markers = bool(
            value('publish_local_classification_markers')
        )
        self.local_classification_marker_topic = value(
            'local_classification_marker_topic'
        )
        self.local_classification_publish_period = max(
            0.0,
            float(value('local_classification_publish_period')),
        )
        self.local_classification_z_offset = float(
            value('local_classification_z_offset')
        )
        self.min_x = float(value('min_x'))
        self.max_x = float(value('max_x'))
        self.min_y = float(value('min_y'))
        self.max_y = float(value('max_y'))

    def init_publishers(self):
        """创建导航图发布器，以及按需创建的 RViz 调试发布器。"""
        self.graph_pub = self.create_publisher(
            NavigationGraph,
            self.navigation_graph_topic,
            10,
        )
        # 无人订阅时不会发送标记；关闭开关时连 publisher 也不创建。
        self.global_memory_marker_pub = (
            self.create_publisher(
                MarkerArray,
                self.global_memory_marker_topic,
                1,
            )
            if self.publish_global_memory_markers else None
        )
        self.local_classification_marker_pub = (
            self.create_publisher(
                MarkerArray,
                self.local_classification_marker_topic,
                1,
            )
            if self.publish_local_classification_markers else None
        )

    def init_subscribers(self):
        """创建带独立 QoS 的里程计/GridMap 订阅器并进行近似时间同步。"""
        odom_qos = qos_profile_from_parameters(
            self.odom_qos_reliability,
            self.odom_qos_durability,
            self.odom_qos_depth,
        )
        grid_qos = qos_profile_from_parameters(
            self.grid_qos_reliability,
            self.grid_qos_durability,
            self.grid_qos_depth,
        )
        self.odom_sub = Subscriber(
            self,
            Odometry,
            self.odometry_topic,
            qos_profile=odom_qos,
        )
        self.grid_sub = Subscriber(
            self,
            GridMap,
            self.traversability_map_topic,
            qos_profile=grid_qos,
        )
        # 传感器时间戳通常无法严格相同；slop 限制允许的时间差。
        self.ts = ApproximateTimeSynchronizer(
            [self.odom_sub, self.grid_sub],
            queue_size=self.syncsub_queue_size,
            slop=self.syncsub_slop,
        )
        self.ts.registerCallback(self.listener_callback)

    def listener_callback(
        self,
        odom_msg: Odometry,
        grid_map_msg: GridMap,
    ):
        """仅把一对已同步消息入队，把重型处理留给定时器。

        回调必须短小：TF 查询或距离场计算放在这里会堵塞同步器，进而放大传感器
        延迟。满缓冲拒绝时通过节流警告暴露背压。
        """
        self.clbk_cntr += 1
        accepted = self.msg_buffer.add_msg(
            msg={
                'odom': odom_msg,
                'traversability_map': grid_map_msg,
            },
            stamp=grid_map_msg.header.stamp,
        )
        if not accepted:
            self.warn_throttled(
                'message_buffer_full',
                'Synchronized map/odometry pair rejected because the '
                'processing buffer is full while waiting for its oldest TF',
            )

    def process_buffer(self):
        """在所需历史 TF 可用后，顺序处理一对缓冲消息。

        只查看最旧项以维持时间顺序。TF 暂不可用时保留它重试；超过等待年龄或
        查询次数上限则丢弃，防止永久缺失 TF 让整条管线停摆。
        """
        entry = self.msg_buffer.get_oldest()
        if entry is None:
            return
        msg = entry.msg
        odom_msg = msg['odom']
        grid_map_msg = msg['traversability_map']
        try:
            # 两种输入可来自不同 frame，且都必须转换到同一个 global_frame。
            transforms = {
                'global_from_odom': self.resolve_transform(
                    odom_msg.header.frame_id,
                    odom_msg.header.stamp,
                ),
                'global_from_grid': self.resolve_transform(
                    grid_map_msg.header.frame_id,
                    grid_map_msg.header.stamp,
                ),
            }
        except TransformException as exc:
            # 可恢复错误：保留最旧消息，直到超时/重试上限触发。
            entry.tf_attempts += 1
            entry.last_tf_error = str(exc)
            expired = (
                self.tf_message_max_age > 0.0
                and entry.wait_age() >= self.tf_message_max_age
            )
            attempts_exhausted = (
                self.tf_max_lookup_attempts > 0
                and entry.tf_attempts >= self.tf_max_lookup_attempts
            )
            if expired or attempts_exhausted:
                self.msg_buffer.pop_oldest()
                self.tf_timeout_drop_count += 1
                reason = (
                    'age limit'
                    if expired
                    else 'lookup-attempt limit'
                )
                self.warn_throttled(
                    'tf_message_drop',
                    'Dropping synchronized map/odometry pair after TF '
                    f'{reason}: age={entry.wait_age():.3f}s, '
                    f'attempts={entry.tf_attempts}, error={exc}',
                )
            return
        except ValueError as exc:
            # 配置或 frame_id 错误不可通过重试解决，直接丢弃该对消息。
            self.get_logger().warning(str(exc))
            self.msg_buffer.pop_oldest()
            return

        self.msg_buffer.pop_oldest()
        self.do_processing(msg, transforms)
        self.processed_message_count += 1

    def warn_throttled(
        self,
        key: str,
        message: str,
        period: float = 5.0,
    ):
        """以键为粒度节流告警，同类问题最多每 ``period`` 秒输出一次。"""
        now = time.monotonic()
        last_time = self.warning_timestamps.get(key, -math.inf)
        if now - last_time < period:
            return
        self.warning_timestamps[key] = now
        self.get_logger().warning(message)

    def resolve_transform(self, source_frame: str, stamp) -> PlanarTransform:
        """查询 ``global_frame <- source_frame`` 的消息时间戳平面 TF。

        相同 frame 无需 TF；不同 frame 但禁用 TF 则明确失败。底层转换还会拒绝
        roll/pitch，避免把倾斜三维坐标系错误地投影为 2.5D 地图。
        """
        if not source_frame:
            raise ValueError('Input message frame_id must not be empty')
        if source_frame == self.global_frame:
            return PlanarTransform()
        if not self.use_tf or self.tf_buffer is None:
            raise ValueError(
                f"Input frame '{source_frame}' differs from global frame "
                f"'{self.global_frame}', but use_tf is false"
            )
        # 必须用消息自身时间戳，而不是“最新 TF”，否则运动中会发生空间错配。
        transform = self.tf_buffer.lookup_transform(
            self.global_frame,
            source_frame,
            stamp,
            timeout=Duration(seconds=self.tf_lookup_timeout),
        )
        return planar_transform_from_tf(transform)

    def do_processing(
        self,
        msg: Dict[str, object],
        transforms: Dict[str, PlanarTransform],
    ):
        """解码一帧局部地图，执行算法 1--5，并发布持久图状态。"""
        odom_msg = msg['odom']
        grid_map_msg = msg['traversability_map']
        # 里程计位置和地图都会被独立转换，但最终均位于 global_frame。
        robot_x, robot_y, _ = transform_pose_position(
            odom_msg.pose.pose,
            transforms['global_from_odom'],
        )
        robot_xy = (robot_x, robot_y)

        try:
            # 这一层处理循环缓冲区、地图 yaw、可选观测/成本层和状态分类。
            grid = TraversabilityGrid(
                msg=grid_map_msg,
                traversability_layer=self.traversability_layer,
                safe_threshold=self.safe_threshold,
                elevation_layer=self.elevation_layer,
                observed_layer=self.observed_layer,
                observed_threshold=self.observed_threshold,
                traversability_semantics=self.traversability_semantics,
                unknown_value_policy=self.unknown_value_policy,
                traversability_cost_layer=self.traversability_cost_layer,
                cost_min=self.cost_min,
                cost_max=self.cost_max,
                cost_higher_is_riskier=self.cost_higher_is_riskier,
                strict_cost_range=self.strict_cost_range,
                frame_transform=transforms['global_from_grid'],
                output_frame=self.global_frame,
            )
            self.validate_resolution(grid.resolution)
            if self.crop_to_local_radius:
                # 将矩形 GridMap 外围裁成论文对应的可靠感知圆。
                grid.apply_circular_mask(robot_xy, self.local_map_radius)
            self.validate_grid_contract(grid)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        # 局部二维图展示的就是分类/裁切后的当前算法输入，即使整帧全未知也发布。
        self.publish_local_classification(
            grid,
            grid_map_msg.header.stamp,
        )

        # 全部未知时无法更新节点或边；保留上一帧持久图比制造空图更安全。
        if not grid.free_cells and not grid.obstacle_cells:
            self.get_logger().warning(
                'No observed cells in reliable local map; skipping update'
            )
            return

        # 计时只用于诊断；算法状态完全由输入地图和随机种子决定。
        update_start = time.monotonic()
        state = self.graph_builder.update_navigation_graph(grid, robot_xy)
        update_duration = time.monotonic() - update_start
        self.report_graph_diagnostics(
            self.graph_builder.graph_diagnostics(),
            update_duration,
        )
        if self.publish_global_memory_markers:
            self.update_global_memory(grid)
            self.latest_debug_stamp = grid_map_msg.header.stamp
        self.graph_pub.publish(
            navigation_graph_message(
                nodes=state.nodes,
                edges=state.edges,
                edge_costs=state.edge_costs,
                current_node_idx=state.current_node_idx,
                traversability_class=self.traversability_class,
                frame_id=self.global_frame,
                stamp=grid_map_msg.header.stamp,
            )
        )

    def validate_resolution(self, resolution: float):
        """当输入分辨率偏离预期值时告警，严格模式下拒绝该帧。"""
        if math.isclose(
            resolution,
            self.expected_map_resolution,
            rel_tol=0.0,
            abs_tol=self.resolution_tolerance,
        ):
            return
        message = (
            f'GridMap resolution {resolution:.4f} m differs from expected '
            f'{self.expected_map_resolution:.4f} m'
        )
        if self.strict_resolution_check:
            raise ValueError(message)
        if not self.resolution_warning_emitted:
            self.get_logger().warning(message)
            self.resolution_warning_emitted = True

    def local_bounds_description(self) -> str:
        """返回当前使用矩形地图边界还是圆形可靠掩码的简短描述。"""
        if self.crop_to_local_radius:
            return f'radial({self.local_map_radius:.2f} m)'
        return 'GridMap rectangle'

    def validate_grid_contract(self, grid: TraversabilityGrid):
        """记录地图几何合同，并验证当前帧实际能否产生前沿。

        这些检查不假设上游地图尺寸；它们会先输出观测到的合同，再按可选期望
        尺寸进行告警/拒绝，帮助部署者发现图层、掩码或尺寸配置错误。
        """
        geometry = (
            grid.frame_id,
            grid.length_x,
            grid.length_y,
            grid.height,
            grid.width,
            grid.resolution,
        )
        # 矩形边界状态和前沿候选数能揭示“地图全有限、没有未知区”等上游问题。
        boundary_counts = grid.boundary_state_counts()
        frontier_candidate_count = len(
            grid.unknown_frontier_cells_next_to_free(
                connectivity=getattr(self, 'frontier_connectivity', 4),
            )
        )
        summary = (
            f"GridMap frame='{grid.frame_id}', "
            f'size={grid.length_x:.3f} x {grid.length_y:.3f} m, '
            f'cells={grid.height} x {grid.width}, '
            f'resolution={grid.resolution:.3f} m, '
            'boundary='
            f"unknown:{boundary_counts['unknown']}, "
            f"free:{boundary_counts['free']}, "
            f"obstacle:{boundary_counts['obstacle']}, "
            f'frontier_candidates:{frontier_candidate_count}'
        )
        if self.log_grid_map_contract and not self.grid_contract_logged:
            self.get_logger().info(summary)
            self.grid_contract_logged = True
        if (
            self.last_grid_geometry is not None
            and geometry != self.last_grid_geometry
        ):
            self.get_logger().warning(
                f'GridMap geometry changed at runtime: {summary}'
            )
            self.unknown_boundary_warning_emitted = False
        self.last_grid_geometry = geometry

        size_errors = []
        if (
            self.expected_map_length_x > 0.0
            and not math.isclose(
                grid.length_x,
                self.expected_map_length_x,
                rel_tol=0.0,
                abs_tol=self.map_length_tolerance,
            )
        ):
            size_errors.append(
                f'length_x={grid.length_x:.3f} m, expected '
                f'{self.expected_map_length_x:.3f} m'
            )
        if (
            self.expected_map_length_y > 0.0
            and not math.isclose(
                grid.length_y,
                self.expected_map_length_y,
                rel_tol=0.0,
                abs_tol=self.map_length_tolerance,
            )
        ):
            size_errors.append(
                f'length_y={grid.length_y:.3f} m, expected '
                f'{self.expected_map_length_y:.3f} m'
            )
        if size_errors:
            message = 'GridMap size mismatch: ' + '; '.join(size_errors)
            if self.strict_map_size_check:
                raise ValueError(message)
            self.get_logger().warning(message)

        if frontier_candidate_count == 0:
            # 这里针对的是几何探索前沿，并非普通的地图有效性错误。
            message = (
                'GridMap contains no Free/Unknown frontier candidates after '
                'all configured masking. Exploration requires an observed '
                'mask, upstream unknown cells, or radial frontier cropping.'
            )
            require_frontiers = (
                getattr(self, 'require_frontier_candidates', False)
                or getattr(self, 'require_unknown_map_boundary', False)
            )
            warn_no_frontiers = (
                getattr(self, 'warn_if_no_frontier_candidates', False)
                or getattr(self, 'warn_if_no_unknown_boundary', False)
            )
            if require_frontiers:
                raise ValueError(message)
            if (
                warn_no_frontiers
                and not self.unknown_boundary_warning_emitted
            ):
                self.get_logger().warning(message)
                self.unknown_boundary_warning_emitted = True
        else:
            self.unknown_boundary_warning_emitted = False

    def report_graph_diagnostics(
        self,
        diagnostics: Dict[str, float],
        update_duration: float,
    ):
        """对不安全/异常图状态发出节流告警，并定期记录规模诊断。"""
        # 以下阈值只报告健康度，不会暗中删除节点或边改变规划结果。
        if not diagnostics['current_node_is_safe']:
            self.warn_throttled(
                'unsafe_current_node',
                'No graph node is safely reachable from the current robot '
                'pose; current_node_idx uses the nearest geometric fallback',
            )
        if (
            diagnostics['frontier_nodes'] > 0
            and diagnostics['reachable_frontier_nodes'] == 0
        ):
            self.warn_throttled(
                'unreachable_frontiers',
                'The graph contains frontier nodes, but none are reachable '
                'from current_node_idx',
            )
        if diagnostics['nodes'] >= self.warn_node_count:
            self.warn_throttled(
                'graph_node_limit',
                f"Graph node count {diagnostics['nodes']} exceeds warning "
                f'threshold {self.warn_node_count}',
            )
        if diagnostics['edges'] >= self.warn_edge_count:
            self.warn_throttled(
                'graph_edge_limit',
                f"Graph edge count {diagnostics['edges']} exceeds warning "
                f'threshold {self.warn_edge_count}',
            )
        if update_duration >= self.warn_update_duration:
            stage_summary = ', '.join(
                f'{name}={duration:.3f}s'
                for name, duration in sorted(
                    self.graph_builder.last_stage_durations.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if name != 'total'
            )
            self.warn_throttled(
                'graph_update_slow',
                f'Graph update took {update_duration:.3f}s, exceeding '
                f'{self.warn_update_duration:.3f}s; stages: '
                f'{stage_summary}',
            )

        now = time.monotonic()
        if (
            self.graph_diagnostics_period <= 0.0
            or now - self.last_graph_diagnostics_time
            < self.graph_diagnostics_period
        ):
            return
        self.last_graph_diagnostics_time = now
        self.get_logger().info(
            'Graph diagnostics: '
            f"nodes={diagnostics['nodes']}, "
            f"edges={diagnostics['edges']}, "
            f"components={diagnostics['components']}, "
            f"current_component={diagnostics['current_component_nodes']}, "
            f"frontiers={diagnostics['frontier_nodes']}, "
            f"frontier_points={diagnostics['frontier_points']}, "
            f"frontier_candidates={diagnostics['frontier_candidates']}, "
            f"frontier_path_checks={diagnostics['frontier_path_checks']}, "
            f'frontier_unsafe_approaches='
            f"{diagnostics['frontier_unsafe_approaches']}, "
            f"frontier_owner_nodes={diagnostics['frontier_owner_nodes']}, "
            f'frontier_component_rejects='
            f"{diagnostics['frontier_component_rejects']}, "
            f"edge_candidates={diagnostics['edge_candidates']}, "
            f"edge_validations={diagnostics['edge_validations']}, "
            f'historical_edge_checks='
            f"{diagnostics['historical_edge_checks']}, "
            f'reachable_frontiers='
            f"{diagnostics['reachable_frontier_nodes']}, "
            f"avg_degree={diagnostics['average_degree']:.2f}, "
            f"max_degree={diagnostics['max_degree']}, "
            f'update={update_duration:.3f}s, '
            f'tf_drops={self.tf_timeout_drop_count}, '
            f'buffer_overflow_drops='
            f'{self.msg_buffer.dropped_overflow_count}, '
            f'buffer_rejected={self.msg_buffer.rejected_full_count}'
        )
        self.get_logger().info(
            'Graph stage timings: '
            + ', '.join(
                f'{name}={duration:.4f}s'
                for name, duration in (
                    self.graph_builder.last_stage_durations.items()
                )
            )
        )

    def update_global_memory(self, grid: TraversabilityGrid):
        """仅在启用调试可视化时合并局部观测单元到稀疏全局记忆。"""
        if self.global_memory is None:
            # 首帧延迟创建，避免默认关闭时分配任何调试存储。
            self.global_memory = GlobalTraversabilityMemory(
                grid.resolution,
                self.global_memory_max_cells,
            )
            self.debug_limit_warned = False
        elif not math.isclose(
            self.global_memory.resolution,
            grid.resolution,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            self.get_logger().warning(
                'GridMap resolution changed; resetting sparse debug memory'
            )
            self.global_memory = GlobalTraversabilityMemory(
                grid.resolution,
                self.global_memory_max_cells,
            )
            self.debug_limit_warned = False
        self.global_memory.integrate(grid)
        if self.global_memory.limit_reached and not self.debug_limit_warned:
            self.get_logger().warning(
                'Sparse global debug memory reached global_memory_max_cells'
            )
            self.debug_limit_warned = True

    def publish_debug_outputs(self):
        """以独立低频率发布有界稀疏 RViz 标记。"""
        # 没有订阅者时跳过 marker 构造，降低大地图调试状态的空转开销。
        if (
            self.global_memory_marker_pub is None
            or self.global_memory is None
            or self.latest_debug_stamp is None
            or self.global_memory_marker_pub.get_subscription_count() == 0
        ):
            return
        self.global_memory_marker_pub.publish(
            global_memory_marker_array(
                self.global_memory,
                self.global_frame,
                self.latest_debug_stamp,
                self.global_memory_marker_stride,
            )
        )

    def publish_local_classification(
        self,
        grid: TraversabilityGrid,
        stamp,
    ):
        """按限频发布当前算法帧的平面三态分类，不保留历史单元."""
        if (
            self.local_classification_marker_pub is None
            or self.local_classification_marker_pub.get_subscription_count()
            == 0
        ):
            return
        now = time.monotonic()
        if (
            self.local_classification_publish_period > 0.0
            and now - self.last_local_classification_publish_time
            < self.local_classification_publish_period
        ):
            return
        self.local_classification_marker_pub.publish(
            local_classification_marker_array(
                grid,
                self.global_frame,
                stamp,
                self.local_classification_z_offset,
            )
        )
        self.last_local_classification_publish_time = now


def main(args=None):
    """初始化、运行并在退出时可靠销毁 ``graphnav_builder`` ROS 2 节点。"""
    rclpy.init(args=args)
    node = SparseGraphBuilderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
