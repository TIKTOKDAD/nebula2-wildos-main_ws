"""单元测试共用的 GridMap 编码器、图构建器和节点夹具。"""

import math

from geometry_msgs.msg import Pose
from graphnav_builder.graph_builder import SparseGraphBuilder
from graphnav_builder.utils.graph_data import GraphNodeState, make_uuid
from graphnav_builder.utils.transforms import PlanarTransform
from graphnav_builder.utils.traversability_grid import TraversabilityGrid
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def encode_layer(
    logical_values,
    outer_start_index=0,
    inner_start_index=0,
    storage_order='column',
):
    """按 GridMap 循环缓冲区物理布局编码一个逻辑矩阵。

    该夹具故意设置非零 ``data_offset``，并支持列主序/行主序，以覆盖生产解码器
    最易出错的三个约束：布局标签、循环起点和数据偏移。
    """
    rows = len(logical_values)
    cols = len(logical_values[0])
    # 先按逻辑到物理的循环偏移写入，随后再按目标存储顺序展平。
    physical = [[math.nan for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            physical_row = (row + outer_start_index) % rows
            physical_col = (col + inner_start_index) % cols
            physical[physical_row][physical_col] = logical_values[row][col]

    layer = Float32MultiArray()
    if storage_order == 'column':
        # grid_map 常见布局：第一个维度为 column，数据按列连续。
        layer.layout.dim = [
            MultiArrayDimension(
                label='column_index',
                size=cols,
                stride=rows * cols,
            ),
            MultiArrayDimension(
                label='row_index',
                size=rows,
                stride=rows,
            ),
        ]
        values = [
            float(physical[row][col])
            for col in range(cols)
            for row in range(rows)
        ]
    else:
        # 兼容标准行主序布局，验证生产代码不会把标签顺序写死。
        layer.layout.dim = [
            MultiArrayDimension(
                label='row_index',
                size=rows,
                stride=rows * cols,
            ),
            MultiArrayDimension(
                label='column_index',
                size=cols,
                stride=cols,
            ),
        ]
        values = [
            float(physical[row][col])
            for row in range(rows)
            for col in range(cols)
        ]
    # 放入两个哨兵值，确认解码器没有忽略 data_offset。
    layer.layout.data_offset = 2
    layer.data = [123.0, 456.0] + values
    return layer


def make_grid(
    traversability,
    elevation=None,
    observed=None,
    cost=None,
    cost_min=0.0,
    cost_max=1.0,
    cost_higher_is_riskier=True,
    strict_cost_range=False,
    resolution=1.0,
    outer_start_index=0,
    inner_start_index=0,
    storage_order='column',
    semantics=TraversabilityGrid.HIGHER_IS_SAFER,
    safe_threshold=0.5,
    unknown_value_policy=TraversabilityGrid.UNKNOWN_NON_FINITE,
    frame_transform=None,
    center_x=0.0,
    center_y=0.0,
    frame_id='map',
    stamp=None,
    yaw=0.0,
):
    """构造并立即解码一张符合论文约束的测试 GridMap。

    参数覆盖生产输入中的可选高程、观测、成本层、循环缓冲区、坐标变换和地图
    朝向，使行为测试不需要 ROS 发布器或 TF 运行时。
    """
    rows = len(traversability)
    cols = len(traversability[0])
    msg = GridMap()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.info.resolution = resolution
    msg.info.length_x = rows * resolution
    msg.info.length_y = cols * resolution
    msg.info.pose.position.x = float(center_x)
    msg.info.pose.position.y = float(center_y)
    msg.info.pose.orientation.z = math.sin(0.5 * yaw)
    msg.info.pose.orientation.w = math.cos(0.5 * yaw)
    msg.outer_start_index = outer_start_index
    msg.inner_start_index = inner_start_index
    msg.layers = ['traversability']
    msg.data = [
        encode_layer(
            traversability,
            outer_start_index,
            inner_start_index,
            storage_order,
        )
    ]

    # 仅把提供的可选层写入消息，模拟实际发布端可能缺少某些图层的情况。
    optional_layers = (
        ('elevation', elevation),
        ('observed', observed),
        ('cost', cost),
    )
    for layer_name, values in optional_layers:
        if values is None:
            continue
        msg.layers.append(layer_name)
        msg.data.append(
            encode_layer(
                values,
                outer_start_index,
                inner_start_index,
                storage_order,
            )
        )

    return TraversabilityGrid(
        msg=msg,
        traversability_layer='traversability',
        safe_threshold=safe_threshold,
        elevation_layer='elevation' if elevation is not None else '',
        observed_layer='observed' if observed is not None else '',
        observed_threshold=0.5,
        traversability_semantics=semantics,
        unknown_value_policy=unknown_value_policy,
        traversability_cost_layer='cost' if cost is not None else '',
        cost_min=cost_min,
        cost_max=cost_max,
        cost_higher_is_riskier=cost_higher_is_riskier,
        strict_cost_range=strict_cost_range,
        frame_transform=frame_transform or PlanarTransform(),
        output_frame='odom',
    )


def make_builder(**overrides):
    """创建参数固定的、与 ROS 无关的图构建器测试实例。"""
    parameters = {
        'max_free_radius': 4.0,
        'traversable_radius': 0.1,
        'edge_radius': 8.0,
        'num_samples': 0,
        'random_seed': 7,
        'ensure_robot_anchor': False,
    }
    parameters.update(overrides)
    return SparseGraphBuilder(**parameters)


def make_node(grid, cell, free_radius=1.0, explored_radius=0.0):
    """在已解码 GridMap 的指定单元中心创建图节点。"""
    pose = Pose()
    pose.position.x, pose.position.y = grid.cell_to_xy(cell)
    pose.position.z = grid.elevation_at_cell(cell)
    pose.orientation.w = 1.0
    return GraphNodeState(
        pose=pose,
        uuid_msg=make_uuid(),
        free_radius=free_radius,
        explored_radius=explored_radius,
    )


def make_node_at(x, y, z=0.0, free_radius=1.0, explored_radius=0.0):
    """在明确给定的全局位置创建图节点，不依赖任何栅格。"""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return GraphNodeState(
        pose=pose,
        uuid_msg=make_uuid(),
        free_radius=free_radius,
        explored_radius=explored_radius,
    )
