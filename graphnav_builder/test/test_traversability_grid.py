"""验证 GridMap 解码、坐标变换、距离场和线段几何规则。"""

import math
from test.helpers import make_grid

from geometry_msgs.msg import Quaternion, TransformStamped
from graphnav_builder.utils.transforms import planar_transform_from_tf
from graphnav_builder.utils.transforms import PlanarTransform
from graphnav_builder.utils.transforms import quaternion_to_planar_yaw
from graphnav_builder.utils.traversability_grid import TraversabilityGrid
import pytest


def test_circular_column_major_layer_is_unwrapped():
    """循环 GridMap 存储必须解码为逻辑行主序数值。"""
    logical = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    grid = make_grid(
        logical,
        resolution=0.5,
        outer_start_index=1,
        inner_start_index=2,
    )
    assert grid.values == pytest.approx([
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
    ])


def test_row_major_layer_is_supported():
    """两种文档化的 GridMap 布局标签顺序都必须被接受。"""
    grid = make_grid(
        [[0.1, 0.2], [0.3, 0.4]],
        storage_order='row',
    )
    assert grid.values == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_observed_elevation_and_lower_is_safer_semantics():
    """观测掩码和可配置通行性正负语义必须正确生效。"""
    grid = make_grid(
        traversability=[[0.1, 0.9]],
        elevation=[[2.5, 3.5]],
        observed=[[1.0, 0.0]],
        semantics=TraversabilityGrid.LOWER_IS_SAFER,
    )
    assert grid.state_at_cell((0, 0)) == TraversabilityGrid.FREE
    assert grid.state_at_cell((0, 1)) == TraversabilityGrid.UNKNOWN
    assert grid.elevation_at_cell((0, 0)) == pytest.approx(2.5)


def test_negative_unknown_value_policy():
    """负的有限哨兵值在配置后可以被分类为未知。"""
    grid = make_grid(
        [[-1.0, 1.0]],
        unknown_value_policy=TraversabilityGrid.UNKNOWN_NEGATIVE,
    )
    assert grid.state_at_cell((0, 0)) == TraversabilityGrid.UNKNOWN
    assert grid.state_at_cell((0, 1)) == TraversabilityGrid.FREE


def test_planar_frame_transform_and_rotated_map_round_trip():
    """经全局平移与 yaw 旋转后，栅格坐标转换必须可往返。"""
    grid = make_grid(
        [[1.0, 1.0], [1.0, 1.0]],
        frame_transform=PlanarTransform(x=10.0, y=-2.0, yaw=math.pi / 2),
    )
    for cell in ((0, 0), (0, 1), (1, 0), (1, 1)):
        assert grid.xy_to_cell(grid.cell_to_xy(cell)) == cell


def test_tf_conversion_accepts_yaw_and_rejects_roll():
    """只有平面 TF 旋转与 2.5D 图构建兼容。"""
    transform = TransformStamped()
    transform.transform.translation.x = 2.0
    transform.transform.rotation.z = math.sin(math.pi / 4)
    transform.transform.rotation.w = math.cos(math.pi / 4)
    planar = planar_transform_from_tf(transform)
    assert planar.x == pytest.approx(2.0)
    assert planar.yaw == pytest.approx(math.pi / 2)

    roll = Quaternion()
    roll.x = math.sin(0.1)
    roll.w = math.cos(0.1)
    with pytest.raises(ValueError, match='roll/pitch'):
        quaternion_to_planar_yaw(roll)


def test_local_radius_masks_cells_outside_reliable_region():
    """r_max 外的单元必须在建图前变为未知。"""
    grid = make_grid([[1.0] * 5 for _ in range(5)])
    grid.apply_circular_mask((0.0, 0.0), 1.1)
    assert grid.state_at_cell((2, 2)) == TraversabilityGrid.FREE
    assert grid.state_at_cell((0, 0)) == TraversabilityGrid.UNKNOWN
    assert grid.is_active((2, 2))
    assert not grid.is_active((0, 0))


def test_clearance_field_uses_cell_boundary_distance():
    """距离场净空必须是保守边界距离，而非中心到中心距离。"""
    values = [[1.0] * 3 for _ in range(3)]
    values[0][0] = 0.0
    grid = make_grid(values)
    field = grid.clearance_field(grid.obstacle_cells)
    center_idx = grid.flat_index((1, 1))
    assert field[center_idx] == pytest.approx(
        math.sqrt(2.0) - math.sqrt(0.5)
    )


def test_distance_field_without_targets_is_infinite():
    """无障碍目标时距离场快路径必须保持数学上的无穷净空。"""
    grid = make_grid([[1.0] * 4 for _ in range(3)])

    field = grid.distance_field([])

    assert len(field) == 12
    assert all(math.isinf(distance) for distance in field)


@pytest.mark.parametrize('include_map_exterior', [False, True])
def test_opencv_distance_field_matches_python_reference(
    include_map_exterior,
):
    """精确 EDT 必须逐格保持原 Python 实现的距离语义。"""
    grid = make_grid(
        [[1.0] * 9 for _ in range(7)],
        resolution=0.1,
    )
    targets = [(0, 0), (2, 6), (5, 3), (99, 99)]

    actual = grid.distance_field(
        targets,
        include_map_exterior=include_map_exterior,
    )
    reference = grid._distance_field_python_reference(
        targets,
        include_map_exterior=include_map_exterior,
    )

    assert actual == pytest.approx(reference, abs=1e-6)
    actual_clearance = grid.clearance_field(
        targets,
        include_map_exterior=include_map_exterior,
    )
    reference_clearance = [
        grid.distance_to_cell_boundary(distance)
        for distance in reference
    ]
    assert actual_clearance == pytest.approx(
        reference_clearance,
        abs=1e-6,
    )


def test_opencv_distance_field_with_only_map_exterior_matches_reference():
    """无内部目标时，地图外圈仍必须产生与旧实现一致的有限距离。"""
    grid = make_grid(
        [[1.0] * 6 for _ in range(4)],
        resolution=0.2,
    )

    actual = grid.distance_field([], include_map_exterior=True)
    reference = grid._distance_field_python_reference(
        [],
        include_map_exterior=True,
    )

    assert actual == pytest.approx(reference, abs=1e-6)


def test_out_of_bounds_targets_preserve_infinite_fast_path():
    """仅有越界目标时必须继续返回无穷，不能接受 OpenCV 哨兵距离。"""
    grid = make_grid([[1.0] * 4 for _ in range(3)])

    field = grid.distance_field([(-1, 0), (3, 4), (99, 99)])

    assert all(math.isinf(distance) for distance in field)


def test_safe_free_components_are_split_by_obstacle_wall():
    """安全自由空间标签必须把障碍墙两侧划分为不同四连通分量。"""
    values = [[1.0] * 7 for _ in range(7)]
    for row in range(7):
        values[row][3] = 0.0
    grid = make_grid(values)
    obstacle = grid.clearance_field(grid.obstacle_cells)

    labels = grid.safe_free_space_components(obstacle, clearance=0.1)

    left = labels[grid.flat_index((3, 1))]
    right = labels[grid.flat_index((3, 5))]
    assert left >= 0
    assert right >= 0
    assert left != right


def test_opencv_safe_components_match_python_reference_partition():
    """OpenCV 四连通域必须保持原 DFS 的安全格集合和连通分区。"""
    values = []
    for row in range(15):
        grid_row = []
        for col in range(17):
            selector = (row * 7 + col * 11) % 23
            if selector == 0:
                grid_row.append(math.nan)
            elif selector in (1, 2, 3):
                grid_row.append(0.0)
            else:
                grid_row.append(1.0)
        values.append(grid_row)
    grid = make_grid(values, resolution=0.1)
    grid.apply_circular_mask((0.0, 0.0), radius=0.7)
    obstacle = grid.clearance_field(grid.obstacle_cells)

    actual = grid.safe_free_space_components(obstacle, clearance=0.12)
    reference = grid._safe_free_space_components_python_reference(
        obstacle,
        clearance=0.12,
    )

    assert {
        index for index, label in enumerate(actual) if label < 0
    } == {
        index for index, label in enumerate(reference) if label < 0
    }

    def component_partition(labels):
        """按成员集合比较分量，避免依赖实现特定的标签编号。"""
        components = {}
        for index, label in enumerate(labels):
            if label >= 0:
                components.setdefault(label, set()).add(index)
        return {frozenset(indices) for indices in components.values()}

    assert component_partition(actual) == component_partition(reference)


def test_supercover_includes_corner_touching_cells():
    """对角线精确穿过角点时必须包含两个侧方单元。"""
    grid = make_grid([[1.0] * 2 for _ in range(2)])
    cells = grid.supercover_cells((0, 0), (1, 1))
    assert set(cells) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_frontier_connectivity_distinguishes_diagonal_contact():
    """四邻接必须排除仅对角接触的自由/未知前沿。"""
    grid = make_grid([
        [1.0, 0.0],
        [0.0, math.nan],
    ])
    frontiers4 = dict(
        grid.unknown_frontier_cells_next_to_free(connectivity=4)
    )
    frontiers8 = dict(
        grid.unknown_frontier_cells_next_to_free(connectivity=8)
    )
    assert (1, 1) not in frontiers4
    assert (1, 1) in frontiers8


def test_frontier_detection_scans_sparse_unknown_side_without_semantic_change():
    """unknown 更少时仍须返回完整的未知端点及四个自由侧邻居。"""
    values = [[1.0] * 5 for _ in range(5)]
    values[2][2] = math.nan
    grid = make_grid(values)

    frontiers = dict(
        grid.unknown_frontier_cells_next_to_free(connectivity=4)
    )

    assert set(frontiers) == {(2, 2)}
    assert set(frontiers[(2, 2)]) == {
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
    }


def test_default_frontier_detection_is_cached_by_connectivity():
    """默认全图检测应按邻接规则复用结果，显式子集调用不得污染缓存。"""
    values = [[1.0] * 7 for _ in range(7)]
    values[2][2] = math.nan
    values[4][4] = math.nan
    grid = make_grid(values)

    first4 = grid.unknown_frontier_cells_next_to_free(connectivity=4)
    second4 = grid.unknown_frontier_cells_next_to_free(connectivity=4)
    first8 = grid.unknown_frontier_cells_next_to_free(connectivity=8)
    second8 = grid.unknown_frontier_cells_next_to_free(connectivity=8)

    assert second4 is first4
    assert second8 is first8
    assert first8 is not first4

    explicit = grid.unknown_frontier_cells_next_to_free(
        free_cells=[(2, 1)],
        connectivity=4,
    )
    assert explicit is not first4
    assert grid.unknown_frontier_cells_next_to_free(
        connectivity=4,
    ) is first4


def test_frontier_cache_is_invalidated_when_grid_state_is_rebuilt():
    """圆形裁剪改变分类后必须重新检测，不能返回裁剪前缓存。"""
    grid = make_grid([[1.0] * 9 for _ in range(9)])
    before = grid.unknown_frontier_cells_next_to_free(connectivity=4)
    assert before == []

    grid.apply_circular_mask((0.0, 0.0), radius=2.5)
    after = grid.unknown_frontier_cells_next_to_free(connectivity=4)

    assert after is not before
    assert after
    assert grid.unknown_frontier_cells_next_to_free(
        connectivity=4,
    ) is after


def test_collision_free_to_frontier_allows_only_unknown_endpoint():
    """前沿终点允许未知，但中间路径单元必须自由。"""
    clear_grid = make_grid([[1.0, 1.0, math.nan]])
    start = clear_grid.cell_to_xy((0, 0))
    assert clear_grid.collision_free_to_frontier(start, (0, 2))
    assert clear_grid.frontier_approach_cell(start, (0, 2)) == (0, 1)

    blocked_grid = make_grid([[1.0, 0.0, math.nan]])
    blocked_start = blocked_grid.cell_to_xy((0, 0))
    assert not blocked_grid.collision_free_to_frontier(
        blocked_start,
        (0, 2),
    )


def test_frontier_path_enforces_obstacle_clearance_before_endpoint():
    """前沿路径的净空约束只施加在其自由部分。"""
    values = [[1.0] * 5 for _ in range(3)]
    values[1][4] = math.nan
    values[0][2] = 0.0
    values[2][2] = 0.0
    grid = make_grid(values)
    obstacle = grid.clearance_field(grid.obstacle_cells)
    start = grid.cell_to_xy((1, 0))

    assert grid.collision_free_to_frontier(start, (1, 4))
    assert grid.clearance_path_cells_to_frontier(
        start,
        (1, 4),
        obstacle,
        0.5,
    ) is None
    assert grid.clearance_approach_cell_to_frontier(
        start,
        (1, 4),
        obstacle,
        0.5,
    ) is None
    assert grid.clearance_path_cells_to_frontier(
        start,
        (1, 4),
        obstacle,
        0.2,
    ) is not None
    assert grid.clearance_approach_cell_to_frontier(
        start,
        (1, 4),
        obstacle,
        0.2,
    ) == (1, 3)


def test_clipped_line_checks_visible_part_of_boundary_edge():
    """一个端点在图外的线段仍必须保留其图内可见单元。"""
    values = [[1.0] * 5 for _ in range(5)]
    values[2][2] = 0.0
    grid = make_grid(values)
    obstacle_clearance = grid.clearance_field(grid.obstacle_cells)
    assert grid.contradicted_by_obstacle(
        (2.0, 0.0),
        (-10.0, 0.0),
        obstacle_clearance,
        0.1,
    )


def test_fully_external_segment_has_no_local_contradiction():
    """与当前局部图无交集的历史边必须保持未知而非受阻。"""
    grid = make_grid([[1.0] * 3 for _ in range(3)])
    obstacle_clearance = grid.clearance_field(grid.obstacle_cells)
    assert not grid.contradicted_by_obstacle(
        (10.0, 10.0),
        (20.0, 10.0),
        obstacle_clearance,
        0.1,
    )


def test_boundary_state_counts_describe_upstream_unknown_border():
    """边界诊断必须反映未知单元是否被上游地图表示。"""
    grid = make_grid([
        [math.nan, math.nan, math.nan],
        [math.nan, 1.0, math.nan],
        [math.nan, math.nan, math.nan],
    ])
    assert grid.boundary_state_counts() == {
        'free': 0,
        'obstacle': 0,
        'unknown': 8,
    }


def test_circular_mask_exposes_inactive_unknown_frontier_endpoints():
    """可靠半径边界必须仍可作为可用前沿。"""
    grid = make_grid([[1.0] * 21 for _ in range(21)])
    grid.apply_circular_mask((0.0, 0.0), 3.0)

    frontiers = grid.unknown_frontier_cells_next_to_free(connectivity=4)
    assert frontiers
    frontier_cell, free_side_cells = frontiers[0]
    assert not grid.is_active(frontier_cell)
    assert grid.is_frontier_endpoint_allowed(frontier_cell)
    assert free_side_cells


def test_rotated_grid_frontier_keys_are_unique_per_cell():
    """地图 yaw 旋转后，栅格单元身份键不得碰撞。"""
    grid = make_grid(
        [[math.nan] * 5 for _ in range(5)],
        yaw=math.pi / 4.0,
    )
    keys = {
        grid.frontier_key((row, col))
        for row in range(grid.height)
        for col in range(grid.width)
    }
    assert len(keys) == grid.height * grid.width


def test_cost_layer_uses_configured_normalization_range():
    """积分风险必须保留非单位区间成本层之间的差异。"""
    grid = make_grid(
        [[1.0, 1.0, 1.0]],
        cost=[[10.0, 50.0, 90.0]],
        cost_min=0.0,
        cost_max=100.0,
    )
    assert [
        grid.risk_at_cell((0, col))
        for col in range(3)
    ] == pytest.approx([0.1, 0.5, 0.9])


def test_strict_cost_range_rejects_out_of_contract_values():
    """严格风险模式必须拒绝超出配置范围的数值。"""
    with pytest.raises(ValueError, match='outside'):
        make_grid(
            [[1.0]],
            cost=[[101.0]],
            cost_min=0.0,
            cost_max=100.0,
            strict_cost_range=True,
        )
