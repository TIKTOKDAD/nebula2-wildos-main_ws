"""将局部分类与可选稀疏调试记忆渲染为 RViz 标记的辅助函数."""

import math
from typing import List, Tuple

from geometry_msgs.msg import Point
from graphnav_builder.utils.global_memory import GlobalTraversabilityMemory
from graphnav_builder.utils.traversability_grid import TraversabilityGrid
from visualization_msgs.msg import Marker, MarkerArray


def points_marker(
    frame_id: str,
    stamp,
    namespace: str,
    marker_id: int,
    points: List[Point],
    scale: float,
    color: Tuple[float, float, float, float],
) -> Marker:
    """构造一个带颜色的 ``CUBE_LIST`` RViz 标记。

    使用单个 marker 承载同类单元，避免为每个栅格发送一个独立 ROS 消息。
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(scale)
    marker.scale.y = float(scale)
    marker.scale.z = 0.025
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    marker.points = points
    return marker


def local_classification_marker_array(
    grid: TraversabilityGrid,
    frame_id: str,
    stamp,
    z_offset: float = 0.02,
) -> MarkerArray:
    """在单一平面上渲染当前 GridMap 的三态分类.

    点坐标保留在 GridMap 自身的局部地图轴中，再通过 marker pose 一次性施加
    地图中心平移和 yaw。这样旋转地图的方格仍与算法实际使用的单元严格对齐。
    每个方块始终使用原始分辨率，不做抽样或放大，避免调试图产生几何重叠。
    """
    markers = MarkerArray()
    delete_marker = Marker()
    delete_marker.header.frame_id = frame_id
    delete_marker.header.stamp = stamp
    delete_marker.ns = 'local_traversability_2d'
    delete_marker.action = Marker.DELETEALL
    markers.markers.append(delete_marker)

    def point_for_cell(cell) -> Point:
        """返回相对 GridMap 中心、位于固定显示平面的单元中心."""
        row, col = cell
        point = Point()
        point.x = (
            0.5 * grid.length_x - (row + 0.5) * grid.resolution
        )
        point.y = (
            0.5 * grid.length_y - (col + 0.5) * grid.resolution
        )
        point.z = float(z_offset)
        return point

    groups = (
        (
            'local_traversability_2d/free',
            1,
            grid.free_cells,
            (0.0, 0.75, 0.15, 0.45),
        ),
        (
            'local_traversability_2d/obstacle',
            2,
            grid.obstacle_cells,
            (1.0, 0.0, 0.0, 0.75),
        ),
        (
            'local_traversability_2d/unknown',
            3,
            grid.unknown_cells,
            (0.2, 0.45, 1.0, 0.22),
        ),
    )
    half_yaw = 0.5 * grid.map_yaw
    for namespace, marker_id, cells, color in groups:
        marker = points_marker(
            frame_id,
            stamp,
            namespace,
            marker_id,
            [point_for_cell(cell) for cell in cells],
            grid.resolution,
            color,
        )
        marker.pose.position.x = grid.center_x
        marker.pose.position.y = grid.center_y
        marker.pose.position.z = grid.center_z
        marker.pose.orientation.z = math.sin(half_yaw)
        marker.pose.orientation.w = math.cos(half_yaw)
        marker.scale.z = 0.01
        markers.markers.append(marker)
    return markers


def global_memory_marker_array(
    memory: GlobalTraversabilityMemory,
    frame_id: str,
    stamp,
    stride: int,
) -> MarkerArray:
    """从有界稀疏记忆构造自由区与障碍区的 RViz 标记数组。"""
    markers = MarkerArray()
    # 每次发布先清理旧标记，防止已不在稀疏记忆中的单元残留在 RViz。
    delete_marker = Marker()
    delete_marker.header.frame_id = frame_id
    delete_marker.header.stamp = stamp
    delete_marker.ns = 'global_traversability'
    delete_marker.action = Marker.DELETEALL
    markers.markers.append(delete_marker)

    free_points = []
    obstacle_points = []
    for (grid_x, grid_y), state in memory.cells.items():
        # 抽样只影响调试显示密度，绝不影响全局记忆或建图结果。
        if stride > 1 and ((grid_x + grid_y) % stride) != 0:
            continue
        point = Point()
        point.x = (grid_x + 0.5) * memory.resolution
        point.y = (grid_y + 0.5) * memory.resolution
        point.z = memory.elevations.get((grid_x, grid_y), 0.04)
        if state == TraversabilityGrid.FREE:
            free_points.append(point)
        elif state == TraversabilityGrid.OBSTACLE:
            obstacle_points.append(point)

    scale = memory.resolution * stride
    markers.markers.append(
        points_marker(
            frame_id,
            stamp,
            'global_traversability/free',
            1,
            free_points,
            scale,
            (0.0, 0.75, 0.15, 0.35),
        )
    )
    markers.markers.append(
        points_marker(
            frame_id,
            stamp,
            'global_traversability/obstacle',
            2,
            obstacle_points,
            scale,
            (1.0, 0.0, 0.0, 0.75),
        )
    )
    return markers
