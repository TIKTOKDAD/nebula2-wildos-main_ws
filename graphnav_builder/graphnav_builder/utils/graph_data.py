"""建图算法共享的数据模型与二维/三维几何辅助函数。

这里的对象刻意不依赖 rclpy：``SparseGraphBuilder`` 可以在没有 ROS
执行器的单元测试中运行，只在数据边界使用 ROS 消息类型，以便最终直接序列化。
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Set, Tuple
import uuid

from geometry_msgs.msg import Point, Pose
from graphnav_msgs.msg import UUID


# 栅格坐标统一采用 (row, column)，而世界平面坐标采用 (x, y)。
# 明确这两个别名可避免把矩阵索引与米制坐标混用。
Cell = Tuple[int, int]
XY = Tuple[float, float]


@dataclass
class GraphNodeState:
    """单个导航图节点的可变内部状态。

    ``free_radius`` 是节点周围同时避开障碍物和未知区域的安全半径；
    ``explored_radius`` 只记录已知未知边界的历史扩张范围，用来淘汰已经
    被探索覆盖的前沿点。前沿点属于节点，但节点自身始终位于自由空间。
    """

    pose: Pose
    uuid_msg: UUID
    free_radius: float = 0.0
    explored_radius: float = 0.0
    is_frontier: bool = False
    frontier_points: List[Point] = field(default_factory=list)


@dataclass
class GraphState:
    """算法 1--5 持久维护的稀疏无向导航图。

    边使用升序的 ``(from_idx, to_idx)`` 元组存储，因此同一无向边不会重复；
    ``edge_costs`` 使用完全相同的键，确保删除或重编号节点时能同步更新。
    """

    nodes: List[GraphNodeState] = field(default_factory=list)
    edges: Set[Tuple[int, int]] = field(default_factory=set)
    edge_costs: Dict[Tuple[int, int], float] = field(default_factory=dict)
    current_node_idx: int = 0


def make_uuid() -> UUID:
    """生成一个进程内唯一的导航节点 UUID ROS 消息。"""
    uuid_msg = UUID()
    uuid_msg.id = list(uuid.uuid4().bytes)
    return uuid_msg


def pose_xy(pose: Pose) -> XY:
    """提取 ``Pose`` 中用于平面建图的 ``(x, y)`` 坐标。"""
    return float(pose.position.x), float(pose.position.y)


def distance_xy(a: XY, b: XY) -> float:
    """计算两个世界平面位置之间的欧氏距离（单位：米）。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_xyz(a: Pose, b: Pose) -> float:
    """计算两个位姿位置的三维欧氏距离，用于最终边代价。"""
    return math.sqrt(
        (float(a.position.x) - float(b.position.x)) ** 2
        + (float(a.position.y) - float(b.position.y)) ** 2
        + (float(a.position.z) - float(b.position.z)) ** 2
    )
