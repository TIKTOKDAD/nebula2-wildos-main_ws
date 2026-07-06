"""把建图算法的内部状态转换为 ``graphnav_msgs`` ROS 消息。"""

import math
from typing import Dict, Sequence, Set, Tuple

from graphnav_builder.utils.graph_data import (
    distance_pose,
    GraphNodeState,
)
from graphnav_msgs.msg import (
    Edge,
    EdgeTraversability,
    NavigationGraph,
    Node,
    NodeTraversabilityProperties,
)


def graph_node_message(state: GraphNodeState) -> Node:
    """将内部节点状态转换为 ``graphnav_msgs/Node``。

    对半径做有限数检查，防止距离场边界产生的 ``inf`` 进入 ROS 消息并污染
    下游规划器；无限半径在消息层以 0 表示“无可用有限半径”。
    """
    node = Node()
    node.uuid = state.uuid_msg
    node.pose = state.pose
    properties = NodeTraversabilityProperties()
    properties.is_frontier = state.is_frontier
    properties.frontier_points = state.frontier_points
    properties.explored_radius = float(
        state.explored_radius
        if math.isfinite(state.explored_radius)
        else 0.0
    )
    properties.free_radius = float(
        state.free_radius if math.isfinite(state.free_radius) else 0.0
    )
    node.trav_properties = [properties]
    return node


def graph_edge_message(
    nodes: Sequence[GraphNodeState],
    from_idx: int,
    to_idx: int,
    edge_cost: float = math.nan,
    edge_distance_mode: str = '3d',
) -> Edge:
    """将一条内部无向边转换为 ``graphnav_msgs/Edge``。

    若调用方没有缓存边代价，则退化为端点的欧氏距离；最小值限制为
    ``1e-3``，避免零长度边使依赖正权重的图搜索算法失效。
    """
    edge = Edge()
    edge.from_idx = from_idx
    edge.to_idx = to_idx
    traversability = EdgeTraversability()
    if not math.isfinite(edge_cost):
        edge_cost = distance_pose(
            nodes[from_idx].pose,
            nodes[to_idx].pose,
            edge_distance_mode,
        )
    traversability.traversability_cost = float(max(edge_cost, 1e-3))
    edge.traversability = [traversability]
    return edge


def navigation_graph_message(
    nodes: Sequence[GraphNodeState],
    edges: Set[Tuple[int, int]],
    edge_costs: Dict[Tuple[int, int], float],
    current_node_idx: int,
    traversability_class: str,
    frame_id: str,
    stamp,
    edge_distance_mode: str = '3d',
) -> NavigationGraph:
    """构造完整 ``NavigationGraph``，并正确表达空图。

    ``current_node_idx`` 在发布前截断到合法范围；空图按消息约定发布索引 0，
    而不会引用不存在的节点。
    """
    msg = NavigationGraph()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.trav_classes = [traversability_class]
    msg.current_node_idx = (
        min(current_node_idx, len(nodes) - 1) if nodes else 0
    )
    msg.nodes = [graph_node_message(node) for node in nodes]
    # 排序使消息顺序稳定，便于日志比较、录包回放和确定性测试。
    msg.edges = [
        graph_edge_message(
            nodes,
            from_idx,
            to_idx,
            edge_costs.get((from_idx, to_idx), math.nan),
            edge_distance_mode,
        )
        for from_idx, to_idx in sorted(edges)
    ]
    return msg
