"""验证局部 GridMap 滑动时持久导航图的跨帧更新行为。"""

import math
from test.helpers import make_builder, make_grid, make_node_at

import pytest


def node_identity(node):
    """返回节点稳定 UUID 与全局 XY，便于检验滑图前后身份不变。"""
    return (
        tuple(node.uuid_msg.id),
        node.pose.position.x,
        node.pose.position.y,
    )


def node_index_at(builder, expected_xy):
    """查找期望全局 XY 处的节点索引，找不到则以测试错误结束。"""
    for idx, node in enumerate(builder.nodes):
        actual_xy = (node.pose.position.x, node.pose.position.y)
        if actual_xy == pytest.approx(expected_xy):
            return idx
    raise AssertionError(f'No graph node found at {expected_xy}')


def frontier_xy_set(builder):
    """汇总当前各节点归属的所有前沿点 XY 坐标。"""
    return {
        (point.x, point.y)
        for node in builder.nodes
        for point in node.frontier_points
    }


def test_map_shift_preserves_global_node_coordinates():
    """局部图移动不能平移持久全局图节点的世界坐标。"""
    grid_t0 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        center_x=0.0,
        center_y=0.0,
        frame_id='odom',
    )
    grid_t1 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        center_x=2.0,
        center_y=0.0,
        frame_id='odom',
    )
    builder = make_builder()
    builder.nodes = [make_node_at(0.0, 0.0)]

    builder.update_navigation_graph(grid_t0, (0.0, 0.0))
    identity_t0 = node_identity(builder.nodes[0])
    builder.update_navigation_graph(grid_t1, (2.0, 0.0))

    assert node_identity(builder.nodes[0]) == identity_t0
    actual_xy = (
        builder.nodes[0].pose.position.x,
        builder.nodes[0].pose.position.y,
    )
    assert actual_xy == pytest.approx((0.0, 0.0))


def test_map_shift_preserves_nodes_outside_current_map():
    """滑动后位于当前局部图外的节点和边必须作为历史保留。"""
    grid_t0 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        center_x=0.0,
        frame_id='odom',
    )
    grid_t1 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        center_x=2.0,
        frame_id='odom',
    )
    builder = make_builder(edge_radius=2.0)
    builder.nodes = [
        make_node_at(-4.5, 3.5),
        make_node_at(-3.5, 3.5),
    ]

    builder.update_navigation_graph(grid_t0, (0.0, 0.0))
    identities_t0 = [node_identity(node) for node in builder.nodes]
    assert builder.edges == {(0, 1)}
    builder.update_navigation_graph(grid_t1, (2.0, 0.0))

    assert [node_identity(node) for node in builder.nodes] == identities_t0
    assert all(
        grid_t1.xy_to_cell((node.pose.position.x, node.pose.position.y))
        is None
        for node in builder.nodes
    )
    assert builder.edges == {(0, 1)}


def test_map_shift_updates_overlap_nodes():
    """重叠区节点应更新半径/高程，刚变无效的节点必须消失。"""
    grid_t0 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        elevation=[[1.0] * 10 for _ in range(10)],
        center_x=0.0,
        frame_id='odom',
    )
    traversability_t1 = [[1.0] * 10 for _ in range(10)]
    traversability_t1[6][5] = 0.0
    grid_t1 = make_grid(
        traversability_t1,
        elevation=[[3.0] * 10 for _ in range(10)],
        center_x=2.0,
        frame_id='odom',
    )
    builder = make_builder()
    builder.nodes = [
        make_node_at(0.0, 0.0),
        make_node_at(1.0, 0.0),
    ]

    builder.update_navigation_graph(grid_t0, (0.0, 0.0))
    survivor_uuid = tuple(builder.nodes[0].uuid_msg.id)
    invalid_uuid = tuple(builder.nodes[1].uuid_msg.id)
    radius_t0 = builder.nodes[0].free_radius
    explored_t0 = builder.nodes[0].explored_radius
    builder.update_navigation_graph(grid_t1, (2.0, 0.0))

    remaining_uuids = {tuple(node.uuid_msg.id) for node in builder.nodes}
    assert survivor_uuid in remaining_uuids
    assert invalid_uuid not in remaining_uuids
    survivor = builder.nodes[0]
    assert survivor.pose.position.z == pytest.approx(3.0)
    assert survivor.free_radius < radius_t0
    assert survivor.explored_radius >= explored_t0


def test_map_shift_cleans_old_and_adds_new_frontiers():
    """滑图清理应删除陈旧前沿，并为新未知格分配新前沿。"""
    traversability_t0 = [[1.0] * 10 for _ in range(10)]
    traversability_t0[5][5] = math.nan
    traversability_t0[9][5] = math.nan
    grid_t0 = make_grid(
        traversability_t0,
        center_x=0.0,
        frame_id='odom',
    )
    builder = make_builder(frontier_connectivity=4)
    builder.nodes = [
        make_node_at(-1.5, -0.5),
        make_node_at(-3.5, -0.5),
        make_node_at(4.5, -0.5),
    ]
    builder.update_frontier_nodes(grid_t0)
    old_frontiers = {
        grid_t0.cell_to_xy((5, 5)),
        grid_t0.cell_to_xy((9, 5)),
    }
    assert old_frontiers <= frontier_xy_set(builder)

    traversability_t1 = [[1.0] * 10 for _ in range(10)]
    new_frontier_cell = (1, 5)
    traversability_t1[new_frontier_cell[0]][new_frontier_cell[1]] = math.nan
    grid_t1 = make_grid(
        traversability_t1,
        center_x=2.0,
        frame_id='odom',
    )
    builder.update_frontier_nodes(grid_t1)

    current_frontiers = frontier_xy_set(builder)
    new_frontier_xy = grid_t1.cell_to_xy(new_frontier_cell)
    assert old_frontiers.isdisjoint(current_frontiers)
    assert current_frontiers == {new_frontier_xy}
    owner = next(
        node
        for node in builder.nodes
        if any(
            (point.x, point.y) == pytest.approx(new_frontier_xy)
            for point in node.frontier_points
        )
    )
    owner_cell = grid_t1.xy_to_cell(
        (owner.pose.position.x, owner.pose.position.y)
    )
    assert owner_cell is not None
    assert grid_t1.is_free(owner_cell)
    assert grid_t1.state_at_cell(new_frontier_cell) == grid_t1.UNKNOWN
    assert grid_t1.collision_free_to_frontier(
        (owner.pose.position.x, owner.pose.position.y),
        new_frontier_cell,
    )


def test_map_shift_updates_historical_and_new_edges():
    """滑图后应保留安全历史边、删除受阻边并建立新边。"""
    grid_t0 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        center_x=0.0,
        frame_id='odom',
    )
    builder = make_builder(
        max_free_radius=0.4,
        edge_radius=2.1,
        num_samples=0,
    )
    historical_positions = [
        (-4.5, 3.5),
        (-3.5, 3.5),
        (-0.5, 3.5),
        (1.5, 3.5),
        (-0.5, -0.5),
        (1.5, -0.5),
    ]
    builder.nodes = [make_node_at(*xy) for xy in historical_positions]
    builder.update_navigation_graph(grid_t0, (0.0, 0.0))

    outside_edge = tuple(sorted((
        node_index_at(builder, (-4.5, 3.5)),
        node_index_at(builder, (-3.5, 3.5)),
    )))
    overlap_edge = tuple(sorted((
        node_index_at(builder, (-0.5, 3.5)),
        node_index_at(builder, (1.5, 3.5)),
    )))
    blocked_edge = tuple(sorted((
        node_index_at(builder, (-0.5, -0.5)),
        node_index_at(builder, (1.5, -0.5)),
    )))
    assert {outside_edge, overlap_edge, blocked_edge} <= builder.edges

    traversability_t1 = [[1.0] * 10 for _ in range(10)]
    traversability_t1[6][5] = 0.0
    grid_t1 = make_grid(
        traversability_t1,
        center_x=2.0,
        frame_id='odom',
    )

    class OrderedChoice:
        """返回新进入局部图区域的确定性自由格，控制测试采样。"""

        def __init__(self):
            """初始化只包含新可见区域单元的迭代器。"""
            self.cells = iter(((2, 8), (1, 8)))

        def choice(self, _):
            """模拟随机选择，稳定地返回下一预设单元。"""
            return next(self.cells)

    builder.num_samples = 2
    builder.random = OrderedChoice()
    builder.update_navigation_graph(grid_t1, (2.0, 0.0))

    outside_edge = tuple(sorted((
        node_index_at(builder, (-4.5, 3.5)),
        node_index_at(builder, (-3.5, 3.5)),
    )))
    overlap_edge = tuple(sorted((
        node_index_at(builder, (-0.5, 3.5)),
        node_index_at(builder, (1.5, 3.5)),
    )))
    blocked_edge = tuple(sorted((
        node_index_at(builder, (-0.5, -0.5)),
        node_index_at(builder, (1.5, -0.5)),
    )))
    new_edge = tuple(sorted((
        node_index_at(builder, (4.5, -3.5)),
        node_index_at(builder, (5.5, -3.5)),
    )))
    assert outside_edge in builder.edges
    assert overlap_edge in builder.edges
    assert blocked_edge not in builder.edges
    assert new_edge in builder.edges


def test_map_shift_handles_circular_buffer_indices():
    """地图移动与循环缓冲区偏移不能破坏逻辑单元数据。"""
    logical_t1 = [[1.0] * 10 for _ in range(10)]
    logical_t1[6][5] = 0.0
    canonical_grid = make_grid(
        logical_t1,
        center_x=2.0,
        frame_id='odom',
    )
    shifted_grid = make_grid(
        logical_t1,
        center_x=2.0,
        frame_id='odom',
        outer_start_index=3,
        inner_start_index=4,
    )
    builder = make_builder()
    builder.nodes = [make_node_at(0.0, 0.0)]
    identity_t0 = node_identity(builder.nodes[0])

    builder.update_navigation_graph(shifted_grid, (2.0, 0.0))

    assert shifted_grid.values == pytest.approx(canonical_grid.values)
    assert shifted_grid.center_x == pytest.approx(2.0)
    assert shifted_grid.xy_to_cell((0.0, 0.0)) == (7, 5)
    assert node_identity(builder.nodes[0]) == identity_t0
    assert builder.nodes[0].free_radius == pytest.approx(
        canonical_grid.clearance_field(canonical_grid.obstacle_cells)[
            canonical_grid.flat_index((7, 5))
        ]
    )


def test_map_shift_updates_current_node():
    """当前节点应随机器人切换，而持久节点坐标保持固定。"""
    grid_t0 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        center_x=0.0,
        frame_id='odom',
    )
    grid_t1 = make_grid(
        [[1.0] * 10 for _ in range(10)],
        center_x=2.0,
        frame_id='odom',
    )
    builder = make_builder()
    builder.nodes = [
        make_node_at(0.0, 0.0),
        make_node_at(2.0, 0.0),
    ]

    builder.update_navigation_graph(grid_t0, (0.0, 0.0))
    assert builder.current_node_idx == node_index_at(builder, (0.0, 0.0))
    builder.update_navigation_graph(grid_t1, (2.0, 0.0))

    assert builder.current_node_idx == node_index_at(builder, (2.0, 0.0))
