"""验证与 ROS 无关的论文算法 1--5 实现及其安全扩展。"""

import math
from test.helpers import make_builder, make_grid, make_node, make_node_at

from geometry_msgs.msg import Point
import pytest


def clearance_fields(grid):
    """返回未知边界与障碍物边界的保守净空场，供多项测试复用。"""
    return (
        grid.clearance_field(
            grid.unknown_cells,
            include_map_exterior=True,
        ),
        grid.clearance_field(grid.obstacle_cells),
    )


def test_algorithm2_removes_unknown_node_and_associated_edge():
    """算法 2：安全半径为零的节点及其关联边必须同时删除。"""
    values = [[1.0] * 5 for _ in range(5)]
    values[2][2] = math.nan
    grid = make_grid(values)
    builder = make_builder()
    builder.nodes = [
        make_node(grid, (2, 2)),
        make_node(grid, (1, 1)),
    ]
    builder.edges = {(0, 1)}
    builder.edge_costs = {(0, 1): 1.0}
    unknown, obstacle = clearance_fields(grid)
    builder.update_nodes(grid, unknown, obstacle)
    assert len(builder.nodes) == 1
    assert builder.edges == set()
    assert builder.edge_costs == {}


def test_algorithm2_explored_radius_never_decreases():
    """算法 2：持久探索半径只能扩张，不能因局部图变化而减小。"""
    grid = make_grid([[1.0] * 7 for _ in range(7)])
    builder = make_builder()
    builder.nodes = [
        make_node(grid, (3, 3), explored_radius=20.0)
    ]
    unknown, obstacle = clearance_fields(grid)
    builder.update_nodes(grid, unknown, obstacle)
    assert builder.nodes[0].explored_radius == pytest.approx(20.0)


def test_algorithm2_preserves_node_outside_reliable_circle():
    """算法 2：当前可靠圆外的节点属于历史，不能被当作未知节点删除。"""
    grid = make_grid([[1.0] * 21 for _ in range(21)])
    grid.apply_circular_mask((0.0, 0.0), 3.0)
    builder = make_builder()
    node = make_node_at(8.0, 0.0, free_radius=2.0)
    builder.nodes = [node]
    unknown, obstacle = clearance_fields(grid)
    builder.update_nodes(grid, unknown, obstacle)
    assert builder.nodes == [node]
    assert builder.nodes[0].free_radius == pytest.approx(2.0)


def test_algorithm3_strict_and_enhanced_sampling_modes():
    """算法 3：严格模式只比较 V_(t-1)，增强模式也排斥 V_new 重叠。"""
    assert make_builder().sample_against_new_nodes
    grid = make_grid([[1.0] * 9 for _ in range(9)])
    unknown, obstacle = clearance_fields(grid)
    sample_cells = [(4, 4), (4, 5)]

    class OrderedChoice:
        """按指定顺序返回单元，使随机采样测试具有确定性。"""

        def __init__(self):
            """初始化指向预设采样序列的游标。"""
            self.index = 0

        def choice(self, _):
            """模拟 Random.choice，按预设顺序返回下一个测试单元。"""
            cell = sample_cells[self.index]
            self.index += 1
            return cell

    strict = make_builder(sample_against_new_nodes=False)
    strict.random = OrderedChoice()
    strict.sample_new_nodes(
        grid,
        unknown,
        obstacle,
        sample_cells,
        2,
    )
    assert len(strict.nodes) == 2

    enhanced = make_builder(sample_against_new_nodes=True)
    enhanced.random = OrderedChoice()
    enhanced.sample_new_nodes(
        grid,
        unknown,
        obstacle,
        sample_cells,
        2,
    )
    assert len(enhanced.nodes) == 1


def test_algorithm4_frontiers_are_stable_and_unique():
    """算法 4：重复更新不得重复归属同一个前沿。"""
    values = [[math.nan] * 7 for _ in range(7)]
    for row in range(1, 6):
        for col in range(1, 6):
            values[row][col] = 1.0
    grid = make_grid(values)
    builder = make_builder()
    builder.nodes = [make_node(grid, (3, 3))]
    builder.update_frontier_nodes(grid)
    first_count = len(builder.nodes[0].frontier_points)
    builder.update_frontier_nodes(grid)
    assert first_count > 0
    assert len(builder.nodes[0].frontier_points) == first_count


def test_algorithm4_removes_frontier_that_becomes_known():
    """算法 4：历史前沿格被观测为已知后必须移除。"""
    unknown_grid = make_grid([[1.0, math.nan], [1.0, math.nan]])
    known_grid = make_grid([[1.0, 1.0], [1.0, 1.0]])
    builder = make_builder()
    node = make_node(unknown_grid, (0, 0))
    x, y = unknown_grid.cell_to_xy((0, 1))
    node.frontier_points = [Point(x=x, y=y, z=0.0)]
    node.is_frontier = True
    builder.nodes = [node]
    builder.update_frontier_nodes(known_grid)
    assert builder.nodes[0].frontier_points == []
    assert not builder.nodes[0].is_frontier


def test_algorithm4_removes_historical_frontier_inside_explored_radius():
    """算法 4：落入任一历史探索半径的前沿必须重新检查并移除。"""
    values = [[math.nan] * 5 for _ in range(5)]
    values[2][1] = 1.0
    values[2][2] = 1.0
    grid = make_grid(values)
    owner = make_node(grid, (2, 2), explored_radius=0.0)
    frontier_x, frontier_y = grid.cell_to_xy((2, 3))
    owner.frontier_points = [
        Point(x=frontier_x, y=frontier_y, z=0.0)
    ]
    owner.is_frontier = True
    covering_node = make_node(grid, (2, 1), explored_radius=3.0)
    builder = make_builder(frontier_connectivity=4)
    builder.nodes = [owner, covering_node]
    builder.update_frontier_nodes(grid)
    assert owner.frontier_points == []
    assert not owner.is_frontier


def test_algorithm4_owner_explored_radius_keeps_own_frontier():
    """算法 4：owner 自身 explored radius 不能清除自己的边界前沿。"""
    values = [[math.nan] * 7 for _ in range(7)]
    for row in range(1, 6):
        for col in range(1, 6):
            values[row][col] = 1.0
    grid = make_grid(values)
    owner = make_node(grid, (3, 3), explored_radius=10.0)
    frontier_x, frontier_y = grid.cell_to_xy((3, 6))
    owner.frontier_points = [
        Point(x=frontier_x, y=frontier_y, z=0.0)
    ]
    owner.is_frontier = True
    builder = make_builder(frontier_connectivity=4)
    builder.nodes = [owner]

    builder.update_frontier_nodes(grid)

    assert owner.frontier_points == [
        Point(x=frontier_x, y=frontier_y, z=0.0)
    ]
    assert owner.is_frontier


def test_algorithm4_rejects_frontier_through_narrow_clearance():
    """算法 4：表面自由但窄于 r_trav 的路径不能获得前沿。"""
    values = [[1.0] * 7 for _ in range(7)]
    values[3][6] = math.nan
    for row in (2, 4):
        for col in (3, 4):
            values[row][col] = 0.0

    grid = make_grid(values)
    builder = make_builder(
        traversable_radius=0.5,
        frontier_connectivity=4,
    )
    builder.nodes = [make_node(grid, (3, 1))]
    obstacle = grid.clearance_field(grid.obstacle_cells)

    assert obstacle[grid.flat_index((3, 1))] > 0.5
    assert obstacle[grid.flat_index((3, 3))] < 0.5
    builder.update_frontier_nodes(grid, obstacle)

    assert builder.nodes[0].frontier_points == []
    assert not builder.nodes[0].is_frontier


def test_algorithm4_skips_frontier_without_safe_free_side():
    """终端自由侧净空不足时不得逐个尝试所有 owner 节点。"""
    values = [[0.0] * 7 for _ in range(7)]
    values[3][3] = 1.0
    values[3][4] = math.nan
    grid = make_grid(values)
    builder = make_builder(traversable_radius=0.6)
    builder.nodes = [make_node(grid, (3, 3))]

    builder.update_frontier_nodes(grid)

    assert builder.frontier_candidate_count == 1
    assert builder.frontier_unsafe_approach_count == 1
    assert builder.frontier_path_check_count == 0
    assert not builder.nodes[0].is_frontier


def test_algorithm4_skips_owner_in_disconnected_safe_component():
    """障碍隔开的 owner 不可能 CollisionFree，不应执行逐线段检查。"""
    values = [[1.0] * 7 for _ in range(7)]
    for row in range(7):
        values[row][3] = 0.0
    values[3][6] = math.nan
    grid = make_grid(values)
    builder = make_builder(traversable_radius=0.1)
    builder.nodes = [make_node(grid, (3, 1))]

    builder.update_frontier_nodes(grid)

    assert builder.frontier_candidate_count == 1
    assert builder.frontier_component_reject_count == 1
    assert builder.frontier_path_check_count == 0
    assert not builder.nodes[0].is_frontier


def test_algorithm4_removes_frontier_when_path_becomes_too_narrow():
    """算法 4：地图变化后若路径变窄，历史前沿归属必须撤销。"""
    open_values = [[1.0] * 7 for _ in range(7)]
    open_values[3][6] = math.nan
    open_grid = make_grid(open_values)
    owner = make_node(open_grid, (3, 1))
    frontier_x, frontier_y = open_grid.cell_to_xy((3, 6))
    owner.frontier_points = [
        Point(x=frontier_x, y=frontier_y, z=0.0)
    ]
    owner.is_frontier = True

    narrow_values = [row[:] for row in open_values]
    for row in (2, 4):
        for col in (3, 4):
            narrow_values[row][col] = 0.0
    narrow_grid = make_grid(narrow_values)
    obstacle = narrow_grid.clearance_field(
        narrow_grid.obstacle_cells
    )
    builder = make_builder(
        traversable_radius=0.5,
        frontier_connectivity=4,
    )
    builder.nodes = [owner]

    builder.update_frontier_nodes(narrow_grid, obstacle)

    assert owner.frontier_points == []
    assert not owner.is_frontier


def test_algorithm4_accepts_frontier_through_wide_clearance():
    """算法 4：宽于 r_trav 的自由路径应保持可到达前沿。"""
    values = [[1.0] * 7 for _ in range(7)]
    values[3][6] = math.nan
    for row in (1, 5):
        for col in (3, 4):
            values[row][col] = 0.0

    grid = make_grid(values)
    builder = make_builder(
        traversable_radius=0.5,
        frontier_connectivity=4,
    )
    builder.nodes = [make_node(grid, (3, 1))]
    obstacle = grid.clearance_field(grid.obstacle_cells)

    assert min(
        obstacle[grid.flat_index((3, col))]
        for col in range(1, 6)
    ) > 0.5
    builder.update_frontier_nodes(grid, obstacle)

    frontier_xy = grid.cell_to_xy((3, 6))
    assert [
        (point.x, point.y)
        for point in builder.nodes[0].frontier_points
    ] == [pytest.approx(frontier_xy)]
    assert builder.nodes[0].is_frontier


def test_algorithm4_assigns_frontier_created_by_circular_mask():
    """算法 4：圆形可靠范围裁切产生的边界仍应成为可到达前沿。"""
    grid = make_grid([[1.0] * 21 for _ in range(21)])
    grid.apply_circular_mask((0.0, 0.0), 3.0)
    builder = make_builder(
        frontier_connectivity=4,
        traversable_radius=0.5,
    )
    builder.nodes = [make_node_at(0.0, 0.0)]

    builder.update_frontier_nodes(grid)

    assert any(node.is_frontier for node in builder.nodes)
    assert any(node.frontier_points for node in builder.nodes)


def test_algorithm5_builds_only_clear_collision_free_edges():
    """算法 5：空间近邻仅能穿过有足够净空的自由单元连边。"""
    grid = make_grid([[1.0] * 9 for _ in range(9)])
    builder = make_builder(edge_radius=4.0)
    builder.nodes = [
        make_node(grid, (4, 3)),
        make_node(grid, (4, 5)),
        make_node(grid, (0, 0)),
    ]
    unknown, obstacle = clearance_fields(grid)
    builder.build_edges(grid, unknown, obstacle)
    assert (0, 1) in builder.edges
    assert (0, 2) not in builder.edges


def test_historical_edge_visible_obstacle_is_removed():
    """安全扩展：跨边界旧边必须验证当前可见段并删除受阻边。"""
    values = [[1.0] * 5 for _ in range(5)]
    values[2][2] = 0.0
    grid = make_grid(values)
    builder = make_builder(edge_radius=20.0)
    builder.nodes = [
        make_node_at(2.0, 0.0),
        make_node_at(-10.0, 0.0),
    ]
    builder.edges = {(0, 1)}
    builder.edge_costs = {(0, 1): 12.0}
    obstacle = grid.clearance_field(grid.obstacle_cells)
    builder.validate_existing_edges(grid, obstacle)
    assert builder.edges == set()


def test_integrated_edge_cost_increases_with_risk():
    """风险积分模式只能增加或保持欧氏边代价，不能降低它。"""
    grid = make_grid(
        [[1.0] * 5 for _ in range(5)],
        cost=[[0.5] * 5 for _ in range(5)],
    )
    builder = make_builder(
        edge_cost_mode='integrated_traversability',
        traversability_cost_weight=2.0,
    )
    builder.nodes = [
        make_node(grid, (2, 1)),
        make_node(grid, (2, 3)),
    ]
    cost = builder.compute_edge_cost(grid, 0, 1)
    assert cost == pytest.approx(4.0)


def test_current_node_uses_nearest_global_position():
    """当前节点应选择离机器人最近的全局位置节点。"""
    builder = make_builder()
    builder.nodes = [
        make_node_at(0.0, 0.0),
        make_node_at(5.0, 0.0),
    ]
    builder.update_current_node((4.0, 0.0))
    assert builder.current_node_idx == 1


def test_current_node_rejects_closer_node_across_obstacle():
    """当前节点选择必须优先安全可达节点，而非隔障碍物的几何近点。"""
    values = [[1.0] * 9 for _ in range(9)]
    values[4][4] = 0.0
    grid = make_grid(values)
    robot_xy = grid.cell_to_xy((4, 3))
    builder = make_builder(edge_radius=8.0, traversable_radius=0.1)
    builder.nodes = [
        make_node(grid, (4, 5)),
        make_node(grid, (7, 3)),
    ]
    unknown, obstacle = clearance_fields(grid)

    builder.update_current_node(robot_xy, grid, unknown, obstacle)

    assert builder.current_node_idx == 1
    assert builder.current_node_is_safe


def test_robot_anchor_is_added_when_no_safe_node_is_reachable():
    """没有安全可达节点时，有效机器人位姿必须确定性地播种图连通分量。"""
    grid = make_grid([[1.0] * 21 for _ in range(21)])
    builder = make_builder(
        ensure_robot_anchor=True,
        traversable_radius=0.5,
        num_samples=0,
    )

    builder.update_navigation_graph(grid, (0.0, 0.0))

    assert len(builder.nodes) == 1
    assert builder.robot_anchor_added
    assert builder.current_node_is_safe
    assert builder.current_node_idx == 0
    assert (
        builder.nodes[0].pose.position.x,
        builder.nodes[0].pose.position.y,
    ) == pytest.approx((0.0, 0.0))


def test_connectivity_diagnostics_detect_unreachable_frontier():
    """诊断应报告位于机器人连通分量外的不可达前沿。"""
    grid = make_grid([[1.0] * 9 for _ in range(9)])
    builder = make_builder()
    builder.nodes = [
        make_node(grid, (4, 4)),
        make_node(grid, (4, 5)),
        make_node(grid, (1, 1)),
    ]
    builder.edges = {(0, 1)}
    builder.current_node_idx = 0
    builder.nodes[2].is_frontier = True
    builder.nodes[2].frontier_points = [Point(x=3.0, y=3.0)]

    diagnostics = builder.graph_diagnostics()

    assert diagnostics['components'] == 2
    assert diagnostics['frontier_nodes'] == 1
    assert diagnostics['reachable_frontier_nodes'] == 0
    assert diagnostics['unreachable_frontier_nodes'] == 1


def test_end_to_end_update_returns_persistent_graph_state():
    """算法 1 的端到端更新必须产生可序列化的持久图状态。"""
    values = [[math.nan] * 11 for _ in range(11)]
    for row in range(1, 10):
        for col in range(1, 10):
            values[row][col] = 1.0
    grid = make_grid(values)
    builder = make_builder(num_samples=40, traversable_radius=0.1)
    state = builder.update_navigation_graph(grid, (0.0, 0.0))
    assert state.nodes
    assert 0 <= state.current_node_idx < len(state.nodes)
    assert any(node.is_frontier for node in state.nodes)


def test_update_records_all_algorithm_stage_timings():
    """算法 1：每帧应暴露完整、非负且总耗时一致的阶段计时。"""
    grid = make_grid([[1.0] * 9 for _ in range(9)])
    builder = make_builder(num_samples=10)

    builder.update_navigation_graph(grid, (0.0, 0.0))

    expected = {
        'distance_fields',
        'update_nodes',
        'sample_nodes',
        'robot_anchor',
        'update_frontiers',
        'validate_edges',
        'build_edges',
        'current_node',
        'total',
    }
    assert set(builder.last_stage_durations) == expected
    assert all(
        duration >= 0.0
        for duration in builder.last_stage_durations.values()
    )
    stage_sum = sum(
        duration
        for name, duration in builder.last_stage_durations.items()
        if name != 'total'
    )
    assert builder.last_stage_durations['total'] >= stage_sum
