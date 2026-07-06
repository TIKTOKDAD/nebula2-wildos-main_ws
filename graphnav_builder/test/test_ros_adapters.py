"""验证消息缓冲、消息序列化和 ROS 适配层边界行为。"""

import math
from test.helpers import make_builder, make_grid, make_node, make_node_at

from builtin_interfaces.msg import Time
from graphnav_builder.builder_node import (
    qos_profile_from_parameters,
    SparseGraphBuilderNode,
)
from graphnav_builder.utils.graph_messages import navigation_graph_message
from graphnav_builder.utils.message_buffer import MessageBuffer
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
import pytest
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from tf2_ros import TransformException


def test_message_buffer_drops_oldest_in_realtime_mode():
    """实时模式缓冲区满时必须保留最新同步消息对。"""
    buffer = MessageBuffer(max_size=1, wait_for_oldest=False)
    buffer.add_msg({'value': 1}, Time(sec=1))
    buffer.add_msg({'value': 2}, Time(sec=2))
    assert buffer.get_oldest()[0] == {'value': 2}


def test_message_buffer_reports_rejected_strict_order_input():
    """严格顺序缓冲必须显式统计被拒绝的新输入。"""
    buffer = MessageBuffer(max_size=1, wait_for_oldest=True)
    assert buffer.add_msg({'value': 1}, Time(sec=1))
    assert not buffer.add_msg({'value': 2}, Time(sec=2))
    assert buffer.rejected_full_count == 1
    assert buffer.dropped_overflow_count == 0


def test_explicit_qos_profiles_support_mixed_publishers():
    """里程计和 GridMap 订阅的 QoS 必须能独立配置。"""
    odom_qos = qos_profile_from_parameters(
        'best_effort',
        'volatile',
        20,
    )
    grid_qos = qos_profile_from_parameters(
        'reliable',
        'transient_local',
        5,
    )
    assert odom_qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert odom_qos.durability == DurabilityPolicy.VOLATILE
    assert odom_qos.depth == 20
    assert grid_qos.reliability == ReliabilityPolicy.RELIABLE
    assert grid_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert grid_qos.depth == 5


def test_navigation_graph_serializes_empty_and_nonempty_states():
    """图消息必须能清空下游状态，并保留显式边代价。"""
    empty = navigation_graph_message(
        nodes=[],
        edges=set(),
        edge_costs={},
        current_node_idx=0,
        traversability_class='default',
        frame_id='odom',
        stamp=Time(),
    )
    assert empty.nodes == []
    assert empty.edges == []
    assert empty.current_node_idx == 0

    grid = make_grid([[1.0] * 5 for _ in range(5)])
    nodes = [make_node(grid, (2, 1)), make_node(grid, (2, 3))]
    graph = navigation_graph_message(
        nodes=nodes,
        edges={(0, 1)},
        edge_costs={(0, 1): 7.5},
        current_node_idx=1,
        traversability_class='default',
        frame_id='odom',
        stamp=Time(sec=3),
    )
    assert graph.current_node_idx == 1
    edge_cost = graph.edges[0].traversability[0].traversability_cost
    assert edge_cost == pytest.approx(7.5)


def test_navigation_graph_fallback_edge_cost_respects_distance_mode():
    """未缓存边代价时，消息兜底距离必须遵循二维/三维开关。"""
    nodes = [
        make_node_at(0.0, 0.0, 0.0),
        make_node_at(3.0, 0.0, 4.0),
    ]

    graph_2d = navigation_graph_message(
        nodes=nodes,
        edges={(0, 1)},
        edge_costs={},
        current_node_idx=0,
        traversability_class='default',
        frame_id='odom',
        stamp=Time(),
        edge_distance_mode='2d',
    )
    edge_cost_2d = graph_2d.edges[0].traversability[0].traversability_cost
    assert edge_cost_2d == pytest.approx(3.0)

    graph_3d = navigation_graph_message(
        nodes=nodes,
        edges={(0, 1)},
        edge_costs={},
        current_node_idx=0,
        traversability_class='default',
        frame_id='odom',
        stamp=Time(),
        edge_distance_mode='3d',
    )
    edge_cost_3d = graph_3d.edges[0].traversability[0].traversability_cost
    assert edge_cost_3d == pytest.approx(5.0)


def test_listener_callback_only_buffers_synchronized_messages():
    """同步回调不得直接执行图构建，只能将消息入队。"""
    node = object.__new__(SparseGraphBuilderNode)
    node.clbk_cntr = 0
    node.msg_buffer = MessageBuffer(max_size=2)
    odom = Odometry()
    odom.header.frame_id = 'odom'
    grid = GridMap()
    grid.header.frame_id = 'base_link'
    grid.header.stamp = Time(sec=4)
    node.listener_callback(odom, grid)
    buffered = node.msg_buffer.get_oldest()[0]
    assert node.clbk_cntr == 1
    assert buffered == {
        'odom': odom,
        'traversability_map': grid,
    }


def test_tf_attempt_limit_drops_blocking_oldest_message():
    """永久缺失的历史 TF 不得无限阻塞处理队列。"""
    class Logger:
        """记录告警文本的极简日志替身。"""

        def __init__(self):
            """创建空的告警记录列表。"""
            self.warnings = []

        def warning(self, message):
            """保存节点本应发送的告警文本。"""
            self.warnings.append(message)

    node = object.__new__(SparseGraphBuilderNode)
    node.msg_buffer = MessageBuffer(max_size=2, wait_for_oldest=True)
    node.msg_buffer.add_msg(
        {
            'odom': Odometry(),
            'traversability_map': GridMap(),
        },
        Time(sec=1),
    )
    node.tf_message_max_age = 0.0
    node.tf_max_lookup_attempts = 1
    node.tf_timeout_drop_count = 0
    node.warning_timestamps = {}
    node.processed_message_count = 0
    logger = Logger()
    node.get_logger = lambda: logger
    node.resolve_transform = lambda *_: (
        (_ for _ in ()).throw(TransformException('missing TF'))
    )

    node.process_buffer()

    assert node.msg_buffer.get_oldest() is None
    assert node.tf_timeout_drop_count == 1
    assert logger.warnings


def test_node_does_not_override_rclpy_parameter_api():
    """辅助方法名不得遮蔽 Node.declare_parameters。"""
    assert SparseGraphBuilderNode.declare_parameters is Node.declare_parameters


def test_same_frame_resolution_does_not_require_tf():
    """已经位于 global_frame 的输入必须解析为恒等变换。"""
    node = object.__new__(SparseGraphBuilderNode)
    node.global_frame = 'odom'
    node.use_tf = False
    node.tf_buffer = None
    transform = node.resolve_transform('odom', Time())
    assert transform.x == 0.0
    assert transform.yaw == 0.0


def test_mismatched_frame_requires_tf_when_disabled():
    """禁用 TF 查询时，不同坐标系必须明确报错。"""
    node = object.__new__(SparseGraphBuilderNode)
    node.global_frame = 'odom'
    node.use_tf = False
    node.tf_buffer = None
    with pytest.raises(ValueError, match='use_tf is false'):
        node.resolve_transform('base_link', Time())


def test_strict_resolution_check_rejects_mismatch():
    """严格论文模式必须拒绝分辨率不匹配的地图。"""
    node = object.__new__(SparseGraphBuilderNode)
    node.expected_map_resolution = 0.1
    node.resolution_tolerance = 0.001
    node.strict_resolution_check = True
    with pytest.raises(ValueError, match='differs from expected'):
        node.validate_resolution(0.2)


def test_rectangular_grid_map_bounds_are_default_description():
    """默认构建器不得把局部地图错误假设为圆形。"""
    node = object.__new__(SparseGraphBuilderNode)
    node.crop_to_local_radius = False
    node.local_map_radius = 10.0
    assert node.local_bounds_description() == 'GridMap rectangle'


def test_grid_contract_strict_size_and_unknown_boundary_checks():
    """运行时诊断必须执行已配置的上游地图合同。"""
    class Logger:
        """同时记录 info 与 warning 的测试日志替身。"""

        def __init__(self):
            """创建两类空日志记录列表。"""
            self.infos = []
            self.warnings = []

        def info(self, message):
            """保存节点本应发送的信息日志。"""
            self.infos.append(message)

        def warning(self, message):
            """保存节点本应发送的告警日志。"""
            self.warnings.append(message)

    node = object.__new__(SparseGraphBuilderNode)
    node.log_grid_map_contract = True
    node.grid_contract_logged = False
    node.last_grid_geometry = None
    node.unknown_boundary_warning_emitted = False
    node.expected_map_length_x = 10.0
    node.expected_map_length_y = 10.0
    node.map_length_tolerance = 0.01
    node.strict_map_size_check = True
    node.warn_if_no_unknown_boundary = True
    node.require_unknown_map_boundary = False
    logger = Logger()
    node.get_logger = lambda: logger

    with pytest.raises(ValueError, match='size mismatch'):
        node.validate_grid_contract(make_grid([[1.0] * 5 for _ in range(5)]))
    assert logger.infos

    node.expected_map_length_x = 5.0
    node.expected_map_length_y = 5.0
    node.strict_map_size_check = False
    node.validate_grid_contract(make_grid([[1.0] * 5 for _ in range(5)]))
    assert any(
        'no Free/Unknown frontier' in warning
        for warning in logger.warnings
    )


def test_grid_contract_accepts_frontier_created_by_radial_crop():
    """裁切后的合同验证必须能看到可靠半径边界产生的前沿。"""
    class Logger:
        """记录合同验证输出的最小日志替身。"""

        def __init__(self):
            """创建两类空日志记录列表。"""
            self.infos = []
            self.warnings = []

        def info(self, message):
            """保存信息日志供断言读取。"""
            self.infos.append(message)

        def warning(self, message):
            """保存告警日志供断言读取。"""
            self.warnings.append(message)

    node = object.__new__(SparseGraphBuilderNode)
    node.log_grid_map_contract = True
    node.grid_contract_logged = False
    node.last_grid_geometry = None
    node.unknown_boundary_warning_emitted = False
    node.expected_map_length_x = 0.0
    node.expected_map_length_y = 0.0
    node.map_length_tolerance = 0.01
    node.strict_map_size_check = False
    node.warn_if_no_unknown_boundary = False
    node.require_unknown_map_boundary = False
    node.warn_if_no_frontier_candidates = True
    node.require_frontier_candidates = False
    node.frontier_connectivity = 4
    logger = Logger()
    node.get_logger = lambda: logger
    grid = make_grid([[1.0] * 21 for _ in range(21)])
    grid.apply_circular_mask((0.0, 0.0), 3.0)

    node.validate_grid_contract(grid)

    assert not logger.warnings
    assert 'frontier_candidates:' in logger.infos[0]


def test_builder_fixture_has_unique_node_uuids():
    """每个图节点 UUID 必须唯一。"""
    builder = make_builder()
    grid = make_grid([[1.0] * 5 for _ in range(5)])
    builder.nodes = [
        make_node(grid, (1, 1)),
        make_node(grid, (3, 3)),
    ]
    first = tuple(builder.nodes[0].uuid_msg.id)
    second = tuple(builder.nodes[1].uuid_msg.id)
    assert first != second
    assert math.isfinite(builder.nodes[0].free_radius)
