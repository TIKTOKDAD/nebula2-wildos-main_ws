"""与 ROS 解耦的 WildOS 稀疏导航图算法 1--5 实现。

本文件只接收已解码的 ``TraversabilityGrid`` 和机器人全局位置，不处理订阅、
TF 或消息发布。这样算法状态可跨局部地图滚动持久存在，也能独立单元测试。
"""

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from geometry_msgs.msg import Point, Pose

from graphnav_builder.utils.graph_data import (
    Cell,
    distance_xy,
    distance_pose,
    GraphNodeState,
    GraphState,
    make_uuid,
    pose_xy,
    XY,
)
from graphnav_builder.utils.spatial_index import SpatialHashIndex
from graphnav_builder.utils.traversability_grid import TraversabilityGrid


class SparseGraphBuilder:
    """从局部几何地图持续维护一个全局稀疏无向导航图。

    节点、边与 UUID 都是持久状态；当前帧只更新其可见部分。算法顺序固定为：
    更新旧节点、随机补点、必要时添加机器人锚点、维护前沿、验证旧边、建立新边、
    最后选择当前节点。
    """

    def __init__(
        self,
        max_free_radius: float = 4.0,
        traversable_radius: float = 0.5,
        edge_radius: float = 8.0,
        num_samples: int = 1000,
        random_seed: int = 7,
        sample_against_new_nodes: bool = True,
        frontier_connectivity: int = 4,
        validate_frontier_paths: bool = True,
        keep_frontiers_outside_grid: bool = False,
        prune_historical_frontiers_by_explored_radius: bool = True,
        validate_historical_edges: bool = True,
        ensure_robot_anchor: bool = True,
        edge_cost_mode: str = 'euclidean',
        edge_distance_mode: str = '3d',
        traversability_cost_weight: float = 1.0,
        min_x: float = -math.inf,
        max_x: float = math.inf,
        min_y: float = -math.inf,
        max_y: float = math.inf,
    ):
        """以算法与部署安全参数初始化一个空的持久导航图。

        ``max_free_radius`` 控制节点覆盖范围，``traversable_radius`` 是机器人
        足迹所需净空，``edge_radius`` 限制候选连边长度；其余开关分别控制采样
        稀疏化、前沿邻接、历史前沿清理、历史边验证、机器人锚点和可选风险代价。
        """
        # 参数检查在构造期完成，避免在更新循环中因无效几何约束产生隐性错误。
        if max_free_radius <= 0.0:
            raise ValueError('max_free_radius must be positive')
        if traversable_radius < 0.0:
            raise ValueError('traversable_radius must be non-negative')
        if edge_radius <= 0.0:
            raise ValueError('edge_radius must be positive')
        if num_samples < 0:
            raise ValueError('num_samples must be non-negative')
        if frontier_connectivity not in (4, 8):
            raise ValueError('frontier_connectivity must be 4 or 8')
        if edge_cost_mode not in ('euclidean', 'integrated_traversability'):
            raise ValueError(
                'edge_cost_mode must be euclidean or '
                'integrated_traversability'
            )
        edge_distance_mode = str(edge_distance_mode).lower()
        if edge_distance_mode not in ('2d', '3d'):
            raise ValueError("edge_distance_mode must be '2d' or '3d'")

        # 将数值参数冻结为普通 float/int，避免 ROS 参数类型或 numpy 标量泄漏到热路径。
        self.max_free_radius = float(max_free_radius)
        self.traversable_radius = float(traversable_radius)
        self.edge_radius = float(edge_radius)
        self.num_samples = int(num_samples)
        self.sample_against_new_nodes = bool(sample_against_new_nodes)
        self.frontier_connectivity = int(frontier_connectivity)
        self.validate_frontier_paths = bool(validate_frontier_paths)
        self.keep_frontiers_outside_grid = bool(keep_frontiers_outside_grid)
        self.prune_historical_frontiers_by_explored_radius = bool(
            prune_historical_frontiers_by_explored_radius
        )
        self.validate_historical_edges = bool(validate_historical_edges)
        self.ensure_robot_anchor = bool(ensure_robot_anchor)
        self.edge_cost_mode = edge_cost_mode
        self.edge_distance_mode = edge_distance_mode
        self.traversability_cost_weight = float(
            traversability_cost_weight
        )
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        # 使用私有且可复现的随机源，避免影响调用进程的全局随机状态。
        self.random = random.Random(int(random_seed))
        # ``state`` 是跨帧唯一持久化的算法状态；每次更新都原地演化它。
        self.state = GraphState()
        # 这两个标志只用于诊断：分别说明当前索引是否安全、是否刚添加机器人锚点。
        self.current_node_is_safe = False
        self.robot_anchor_added = False
        # 最近一帧的细粒度耗时与 frontier 工作量仅用于诊断，不参与算法决策。
        self.last_stage_durations: Dict[str, float] = {}
        self.last_frontier_stage_durations: Dict[str, float] = {}
        self.frontier_candidate_count = 0
        self.frontier_path_check_count = 0
        self.historical_frontier_check_count = 0
        self.frontier_unsafe_approach_count = 0
        self.frontier_owner_node_count = 0
        self.frontier_component_reject_count = 0
        self.edge_candidate_count = 0
        self.edge_validation_count = 0
        self.historical_edge_check_count = 0

    @property
    def nodes(self) -> List[GraphNodeState]:
        """返回可变节点列表，供序列化、测试及算法内部访问。"""
        return self.state.nodes

    @nodes.setter
    def nodes(self, value: List[GraphNodeState]):
        """用新节点列表替换图状态中的节点存储。"""
        self.state.nodes = value

    @property
    def edges(self) -> set[Tuple[int, int]]:
        """返回可变无向边集合；每条边的端点索引始终升序存储。"""
        return self.state.edges

    @edges.setter
    def edges(self, value: set[Tuple[int, int]]):
        """用规范化的无向边集合替换图状态中的边存储。"""
        self.state.edges = value

    @property
    def edge_costs(self) -> Dict[Tuple[int, int], float]:
        """返回以边元组为键的已缓存通行代价。"""
        return self.state.edge_costs

    @edge_costs.setter
    def edge_costs(self, value: Dict[Tuple[int, int], float]):
        """用与当前边键对应的代价字典替换缓存。"""
        self.state.edge_costs = value

    @property
    def current_node_idx(self) -> int:
        """返回代表当前机器人位置的图节点索引。"""
        return self.state.current_node_idx

    @current_node_idx.setter
    def current_node_idx(self, value: int):
        """设置当前节点索引，并转换为普通整数以保持消息兼容性。"""
        self.state.current_node_idx = int(value)

    def update_navigation_graph(
        self,
        grid: TraversabilityGrid,
        robot_xy: XY,
    ) -> GraphState:
        """执行算法 1：用一帧局部地图增量更新导航图。

        未知净空将地图外也视为未知，防止节点/边贴近本地观测边界；障碍净空只
        由已观测障碍物计算。两个场共同约束采样、锚点和新边。本帧新节点只从
        机器人所在的四连通 Free 分量采样，但该限制不会删除其他分量中的历史节点。
        """
        update_start = time.perf_counter()
        stage_start = update_start
        timings: Dict[str, float] = {}
        self.edge_candidate_count = 0
        self.edge_validation_count = 0
        self.historical_edge_check_count = 0

        # 距离场的索引与 grid 的行主序单元一一对应，后续不能混用不同帧的距离场。
        # include_map_exterior=True 把地图矩形外当未知，避免在观测边缘留下安全假象。
        unknown_clearance = grid.clearance_field(
            grid.unknown_cells,
            include_map_exterior=True,
        )
        # 障碍净空只针对已知障碍；未知风险由上面的 unknown_clearance 单独约束。
        obstacle_clearance = grid.clearance_field(grid.obstacle_cells)
        timings['distance_fields'] = time.perf_counter() - stage_start

        # 以下顺序不能随意交换：后续步骤必须看到已清理、已重编号的旧图。
        # 算法 2：用当前观测修正旧节点，并清除已经失效的局部节点。
        stage_start = time.perf_counter()
        self.update_nodes(
            grid,
            unknown_clearance,
            obstacle_clearance,
        )
        timings['update_nodes'] = time.perf_counter() - stage_start
        # 算法 3：只在机器人当前可达的可靠自由区补充随机节点。机器人不在
        # Free 单元时返回空候选集，以安全地跳过本帧随机采样。
        stage_start = time.perf_counter()
        reachable_free_cells = grid.reachable_free_cells(robot_xy)
        self.sample_new_nodes(
            grid,
            unknown_clearance,
            obstacle_clearance,
            reachable_free_cells,
            self.num_samples,
        )
        timings['sample_nodes'] = time.perf_counter() - stage_start
        # 部署安全扩展：随机采样遗漏机器人附近时，尝试把机器人确定性接入图。
        stage_start = time.perf_counter()
        self.ensure_robot_anchor_node(
            grid,
            unknown_clearance,
            obstacle_clearance,
            robot_xy,
        )
        timings['robot_anchor'] = time.perf_counter() - stage_start
        # 算法 4：先清理历史前沿，再为本帧 Free/Unknown 边界分配新的归属节点。
        stage_start = time.perf_counter()
        self.update_frontier_nodes(grid, obstacle_clearance)
        timings['update_frontiers'] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        if self.validate_historical_edges:
            # 可选安全扩展：只凭当前可见障碍物撤销已经不可信的旧边。
            self.validate_existing_edges(grid, obstacle_clearance)
        timings['validate_edges'] = time.perf_counter() - stage_start
        # 算法 5：为当前全部节点（含历史节点）补充本帧可证明安全的新边。
        stage_start = time.perf_counter()
        self.build_edges(grid, unknown_clearance, obstacle_clearance)
        timings['build_edges'] = time.perf_counter() - stage_start
        # 发布前确定机器人当前所在/可达的图节点，供下游规划器作为起点。
        stage_start = time.perf_counter()
        self.update_current_node(
            robot_xy,
            grid,
            unknown_clearance,
            obstacle_clearance,
        )
        timings['current_node'] = time.perf_counter() - stage_start
        timings['total'] = time.perf_counter() - update_start
        self.last_stage_durations = timings
        return self.state

    def update_nodes(
        self,
        grid: TraversabilityGrid,
        unknown_clearance: Sequence[float],
        obstacle_clearance: Sequence[float],
    ):
        """执行算法 2：更新旧节点半径，并删除已被局部观测否定的节点。

        地图内的节点根据最新净空更新；地图外节点保留其历史状态。删除节点后
        必须重映射全部边和边代价，因为列表索引就是图的节点 ID。
        """
        # valid_nodes 会压缩删除后的列表；index_remap 保留“旧索引 -> 新索引”映射。
        valid_nodes: List[GraphNodeState] = []
        index_remap: Dict[int, int] = {}
        for old_idx, node in enumerate(self.nodes):
            # 可选工作空间限制适用于历史节点，越界节点与相关边一并淘汰。
            if not self.in_workspace_bounds(pose_xy(node.pose)):
                continue

            # ``None`` 表示节点不在本帧局部矩形中，不能仅因看不见而删除它。
            cell = grid.xy_to_cell(pose_xy(node.pose))
            if cell is not None and grid.is_active(cell):
                # 仅用当前可靠区域修正历史节点；地图外观测不能推翻持久节点。
                cell_idx = grid.flat_index(cell)
                # 真实可用自由半径受障碍、未知边界和部署上限三者中最小值限制。
                node.free_radius = min(
                    obstacle_clearance[cell_idx],
                    unknown_clearance[cell_idx],
                    self.max_free_radius,
                )
                # 探索范围代表“曾确认到的未知边界距离”，只能单调增长。
                node.explored_radius = max(
                    node.explored_radius,
                    unknown_clearance[cell_idx],
                )
                # 节点所在格已变障碍/未知，或正贴着边界时，没有可用自由圆。
                if node.free_radius <= 0.0:
                    continue
                node.pose.position.z = grid.elevation_at_cell(
                    cell,
                    node.pose.position.z,
                )

            # 在 append 前记录新位置；后面重建每条旧边会依赖这份映射。
            index_remap[old_idx] = len(valid_nodes)
            valid_nodes.append(node)

        # 节点列表压缩后，重建边集合以消除指向删除节点的引用。
        old_edges = self.edges
        old_costs = self.edge_costs
        self.nodes = valid_nodes
        self.edges = set()
        self.edge_costs = {}
        for edge in old_edges:
            from_idx, to_idx = edge
            # 任一端删除后，旧边及其代价都不再有合法含义。
            if from_idx not in index_remap or to_idx not in index_remap:
                continue
            remapped = tuple(
                sorted((index_remap[from_idx], index_remap[to_idx]))
            )
            # 防御性检查：无向图不允许重映射后形成自环。
            if remapped[0] == remapped[1]:
                continue
            self.edges.add(remapped)
            # 只迁移仍存在的缓存代价；之后算法 5 会重算本帧新增边的代价。
            if edge in old_costs:
                self.edge_costs[remapped] = old_costs[edge]

    def sample_new_nodes(
        self,
        grid: TraversabilityGrid,
        unknown_clearance: Sequence[float],
        obstacle_clearance: Sequence[float],
        reachable_free_cells: Sequence[Cell],
        num_samples: int,
    ):
        """执行算法 3：在机器人可达自由空间随机采样互不冗余的新节点。

        调用方提供机器人所在的四连通 Free 分量。候选节点还须有超过机器人
        半径的未知/障碍净空，且不得落在既有节点的自由覆盖圆内。可通过
        ``sample_against_new_nodes`` 选择是否也排斥本帧已接受的新节点。
        """
        # 空自由格或禁用采样时不产生副作用，历史图仍由算法 2 的结果保留。
        if not reachable_free_cells or num_samples <= 0:
            return

        # 前一帧节点数量区分论文严格模式与部署中的更稀疏增强模式。
        previous_node_count = len(self.nodes)
        # 同一个格被随机命中多次不能产生重复节点或浪费几何判定。
        accepted_cells = set()
        index = SpatialHashIndex(self.max_free_radius)
        index.rebuild(
            (idx, pose_xy(node.pose))
            for idx, node in enumerate(self.nodes)
        )

        for _ in range(num_samples):
            # choice 是有放回抽样；因此本次尝试并不保证一定产生一个新节点。
            cell = self.random.choice(reachable_free_cells)
            if cell in accepted_cells:
                continue
            cell_idx = grid.flat_index(cell)
            xy = grid.cell_to_xy(cell)
            # 节点需要同时避开未知和障碍，二者最小净空才是安全裕量。
            clearance = min(
                obstacle_clearance[cell_idx],
                unknown_clearance[cell_idx],
            )
            if (
                not self.in_workspace_bounds(xy)
                or clearance <= self.traversable_radius
            ):
                continue

            # 空间哈希只返回可能覆盖候选点的附近节点，避免全图逐一比较。
            candidates = index.radius_search(xy, self.max_free_radius)
            # 若候选位于任一已有节点的 free_radius 内，该节点已经覆盖这片自由空间。
            # candidate_idx 判断严格论文模式是否忽略“本帧刚加入”的 V_new。
            redundant = any(
                (
                    candidate_idx < previous_node_count
                    or self.sample_against_new_nodes
                )
                and distance_xy(
                    xy,
                    pose_xy(self.nodes[candidate_idx].pose),
                )
                <= self.nodes[candidate_idx].free_radius
                for candidate_idx in candidates
            )
            if redundant:
                continue

            # 新节点始终落在自由格中心；Z 使用同格高程以支持三维距离代价。
            pose = Pose()
            pose.position.x = xy[0]
            pose.position.y = xy[1]
            pose.position.z = grid.elevation_at_cell(
                cell,
                grid.center_z,
            )
            pose.orientation.w = 1.0
            new_idx = len(self.nodes)
            self.nodes.append(
                GraphNodeState(
                    pose=pose,
                    uuid_msg=make_uuid(),
                    free_radius=min(clearance, self.max_free_radius),
                    explored_radius=unknown_clearance[cell_idx],
                )
            )
            accepted_cells.add(cell)
            if self.sample_against_new_nodes:
                # 增强模式立即索引新节点，让后续样本也能与它去重。
                index.insert(new_idx, xy)

    def update_frontier_nodes(
        self,
        grid: TraversabilityGrid,
        obstacle_clearance: Optional[Sequence[float]] = None,
    ):
        """执行算法 4：清理、检测并安全地归属几何前沿点。

        前沿点位于“自由格相邻未知格”的未知格中心，却归属到可安全接近它的
        最近图节点。历史前沿在当前图内被证实变已知或失去安全路径时删除；
        是否因离开当前 GridMap 或落入探索覆盖半径而删除由独立参数控制。
        """
        frontier_update_start = time.perf_counter()
        timings = {
            'obstacle_clearance': 0.0,
            'node_index': 0.0,
            'safe_components': 0.0,
            'owner_nodes': 0.0,
            'historical_frontiers': 0.0,
            'candidate_detection': 0.0,
            'candidate_filter': 0.0,
            'owner_sort': 0.0,
            'new_path_checks': 0.0,
            'total': 0.0,
        }
        # 每次独立调用也重置计数，避免测试或调试读取到上一帧工作量。
        self.frontier_candidate_count = 0
        self.frontier_path_check_count = 0
        self.historical_frontier_check_count = 0
        self.frontier_unsafe_approach_count = 0
        self.frontier_owner_node_count = 0
        self.frontier_component_reject_count = 0
        # 没有图节点就没有前沿的安全归属对象，不能单独发布未知格作为节点。
        if not self.nodes:
            timings['total'] = time.perf_counter() - frontier_update_start
            self.last_frontier_stage_durations = timings
            return
        stage_start = time.perf_counter()
        if obstacle_clearance is None:
            # 允许独立调用本方法；完整更新路径会复用算法 1 已计算的距离场。
            obstacle_clearance = grid.clearance_field(
                grid.obstacle_cells
            )
        timings['obstacle_clearance'] = time.perf_counter() - stage_start

        # 桶尺寸兼顾前沿归属搜索半径、节点覆盖尺度和最小地图分辨率。
        stage_start = time.perf_counter()
        bucket_size = max(
            min(self.edge_radius, self.max_free_radius),
            grid.resolution,
        )
        node_index = SpatialHashIndex(bucket_size)
        node_index.rebuild(
            (idx, pose_xy(node.pose))
            for idx, node in enumerate(self.nodes)
        )
        timings['node_index'] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        safe_components = grid.safe_free_space_components(
            obstacle_clearance,
            self.traversable_radius,
        )
        timings['safe_components'] = time.perf_counter() - stage_start
        # 新 frontier 的 owner 必须位于当前安全自由空间，且与终端自由侧属于
        # 同一四连通分量；这是直线 CollisionFree 成立的必要条件。
        stage_start = time.perf_counter()
        owner_nodes_by_component: Dict[int, List[int]] = {}
        frontier_owner_positions: Dict[int, XY] = {}
        for node_idx, node in enumerate(self.nodes):
            node_xy = pose_xy(node.pose)
            node_cell = grid.xy_to_cell(node_xy)
            if node_cell is None:
                continue
            component_idx = safe_components[grid.flat_index(node_cell)]
            if component_idx < 0:
                continue
            owner_nodes_by_component.setdefault(
                component_idx,
                [],
            ).append(node_idx)
            frontier_owner_positions[node_idx] = node_xy
        self.frontier_owner_node_count = len(frontier_owner_positions)
        # 先用全局最大半径粗筛，再在 frontier_is_explored 内检查每个节点的实际半径。
        max_explored_radius = max(
            (node.explored_radius for node in self.nodes),
            default=0.0,
        )
        timings['owner_nodes'] = time.perf_counter() - stage_start

        explored_cache: Dict[Tuple[Cell, int], bool] = {}

        def frontier_is_explored(
            frontier_cell: Cell,
            frontier_xy: XY,
            excluded_node_idx: int = -1,
        ) -> bool:
            """判断前沿是否已落入任一节点累积的探索覆盖半径。"""
            cache_key = (frontier_cell, excluded_node_idx)
            cached = explored_cache.get(cache_key)
            if cached is not None:
                return cached
            # 半径搜索避免让远离前沿的历史节点参与逐个距离比较。
            candidates = node_index.radius_search(
                frontier_xy,
                max_explored_radius,
            )
            explored = any(
                node_idx != excluded_node_idx
                and distance_xy(
                    frontier_xy,
                    pose_xy(self.nodes[node_idx].pose),
                ) <= self.nodes[node_idx].explored_radius
                for node_idx in candidates
            )
            explored_cache[cache_key] = explored
            return explored

        # 同一未知格只能分配给一个节点，防止前沿重复发布和重复规划。
        assigned_frontiers = set()
        # 第一阶段：历史前沿不是永久记忆，必须在当前图上重新验证。
        stage_start = time.perf_counter()
        for owner_idx, node in enumerate(self.nodes):
            kept_frontier_points = []
            for point in node.frontier_points:
                self.historical_frontier_check_count += 1
                # 将历史点重新投回当前栅格，逐项验证仍满足前沿定义和安全接近。
                xy = (point.x, point.y)
                cell = grid.xy_to_cell(xy)
                if cell is None:
                    # 滚动局部图看不到该历史前沿时，保守模式会继续保留；严格
                    # 当前帧模式则将其删除，避免地图外前沿长期残留。
                    if self.keep_frontiers_outside_grid:
                        kept_frontier_points.append(point)
                    continue

                # 以下任一条件都破坏当前可见范围内的前沿语义：终点不允许、已知，
                # 或按配置被其他节点的 explored radius 覆盖。owner 自身半径恰好
                # 触及边界时不能把自己的前沿反复删掉重建。
                if (
                    not grid.is_frontier_endpoint_allowed(cell)
                    or grid.is_known(cell)
                    or (
                        self.prune_historical_frontiers_by_explored_radius
                        and frontier_is_explored(cell, xy, owner_idx)
                    )
                ):
                    continue
                # 安全模式重新证明历史 owner 的直线接近仍然成立；快速模式与
                # 新前沿采用同一开关，完全跳过前沿路径验证。
                if self.validate_frontier_paths:
                    self.frontier_path_check_count += 1
                    if grid.clearance_approach_cell_to_frontier(
                        pose_xy(node.pose),
                        cell,
                        obstacle_clearance,
                        self.traversable_radius,
                    ) is None:
                        continue
                frontier_key = grid.frontier_key(cell)
                if frontier_key in assigned_frontiers:
                    continue
                assigned_frontiers.add(frontier_key)
                kept_frontier_points.append(point)
            # 通过“替换列表”原子地丢弃失效点，并让 is_frontier 与列表保持一致。
            node.frontier_points = kept_frontier_points
            node.is_frontier = bool(kept_frontier_points)
        timings['historical_frontiers'] = (
            time.perf_counter() - stage_start
        )

        # 再从本帧 Free/Unknown 边界发现全新的未分配前沿。
        # 第二阶段：枚举新的 Free/Unknown 边界；grid 同时返回可从哪一侧自由格接近。
        stage_start = time.perf_counter()
        frontier_candidates = grid.unknown_frontier_cells_next_to_free(
            connectivity=self.frontier_connectivity,
        )
        self.frontier_candidate_count = len(frontier_candidates)
        timings['candidate_detection'] = time.perf_counter() - stage_start
        new_frontiers_start = time.perf_counter()
        owner_sort_duration = 0.0
        new_path_check_duration = 0.0
        for frontier_cell, free_side_cells in frontier_candidates:
            frontier_xy = grid.cell_to_xy(frontier_cell)
            frontier_key = grid.frontier_key(frontier_cell)
            # 已保留的历史点优先，避免同一地图格在同帧改变归属造成抖动。
            if frontier_key in assigned_frontiers:
                continue

            # 未知格虽仍在地图中，但已落入探索覆盖圈时不应重复驱动探索。
            if frontier_is_explored(frontier_cell, frontier_xy):
                continue

            # CollisionFree 路径的倒数第二格必然是具有足够障碍净空的自由侧格。
            # 若所有相邻自由侧都不安全，则任何节点都不可能成为合法 owner。
            safe_free_side_cells = {
                cell
                for cell in free_side_cells
                if obstacle_clearance[grid.flat_index(cell)]
                > self.traversable_radius
            }
            if not safe_free_side_cells:
                self.frontier_unsafe_approach_count += 1
                continue

            frontier_components = {
                safe_components[grid.flat_index(cell)]
                for cell in safe_free_side_cells
                if safe_components[grid.flat_index(cell)] >= 0
            }
            candidate_node_indices = {
                node_idx
                for component_idx in frontier_components
                for node_idx in owner_nodes_by_component.get(
                    component_idx,
                    (),
                )
            }
            if not candidate_node_indices:
                self.frontier_component_reject_count += 1
                continue

            approach_cells: Dict[int, Cell] = {}
            best_idx = None
            # Algorithm 4 要求在 collision-free 节点中取欧氏最近者；先按精确
            # 距离排序，再遇到首个合法路径时停止，结果与原 argmin 完全一致。
            sort_start = time.perf_counter()
            ordered_candidates = sorted(
                candidate_node_indices,
                key=lambda node_idx: distance_xy(
                    frontier_xy,
                    frontier_owner_positions[node_idx],
                ),
            )
            owner_sort_duration += time.perf_counter() - sort_start
            if self.validate_frontier_paths:
                path_checks_start = time.perf_counter()
                for node_idx in ordered_candidates:
                    self.frontier_path_check_count += 1
                    approach_cell = (
                        grid.clearance_approach_cell_to_frontier(
                            frontier_owner_positions[node_idx],
                            frontier_cell,
                            obstacle_clearance,
                            self.traversable_radius,
                        )
                    )
                    if approach_cell is None:
                        continue
                    if approach_cell not in safe_free_side_cells:
                        continue
                    # 记录路径末尾自由格；其高程是前沿点 Z 的可靠来源。
                    approach_cells[node_idx] = approach_cell
                    best_idx = node_idx
                    break
                new_path_check_duration += (
                    time.perf_counter() - path_checks_start
                )
            else:
                # 快速模式保留安全自由侧与四连通分量筛选，但不再证明 owner
                # 到 frontier 的直线安全；直接采用同分量内欧氏距离最近节点。
                best_idx = ordered_candidates[0]
                approach_cells[best_idx] = min(
                    safe_free_side_cells,
                    key=lambda cell: (
                        distance_xy(
                            frontier_owner_positions[best_idx],
                            grid.cell_to_xy(cell),
                        ),
                        cell[0],
                        cell[1],
                    ),
                )
            if best_idx is None:
                continue

            # 前沿 XY 位于未知格中心；Z 从相邻自由接近格取，避免采纳未知高程。
            point = Point()
            point.x = float(frontier_xy[0])
            point.y = float(frontier_xy[1])
            point.z = grid.elevation_at_cell(
                approach_cells[best_idx],
                self.nodes[best_idx].pose.position.z,
            )
            self.nodes[best_idx].frontier_points.append(point)
            self.nodes[best_idx].is_frontier = True
            assigned_frontiers.add(frontier_key)
        new_frontiers_duration = time.perf_counter() - new_frontiers_start
        timings['owner_sort'] = owner_sort_duration
        timings['new_path_checks'] = new_path_check_duration
        # 其余时间包括 explored 过滤、自由侧净空、连通分量筛选、集合构造和
        # 最终 Point 写入；用残差统计可避免在每个早退分支重复打点。
        timings['candidate_filter'] = max(
            0.0,
            new_frontiers_duration
            - owner_sort_duration
            - new_path_check_duration,
        )
        timings['total'] = time.perf_counter() - frontier_update_start
        self.last_frontier_stage_durations = timings

    def validate_existing_edges(
        self,
        grid: TraversabilityGrid,
        obstacle_clearance: Sequence[float],
    ):
        """安全扩展：移除被当前局部障碍物证伪的历史边。

        这不是论文算法 5 的一部分；它只检查当前地图可见段，避免滚动地图外
        不可见的历史区域导致边被过度删除。
        """
        # 不能原地删除 self.edges，否则遍历集合时会触发运行时错误。
        valid_edges = set()
        valid_costs = {}
        for edge in self.edges:
            self.historical_edge_check_count += 1
            if self.existing_edge_is_valid(
                grid,
                obstacle_clearance,
                edge[0],
                edge[1],
            ):
                valid_edges.add(edge)
                if edge in self.edge_costs:
                    valid_costs[edge] = self.edge_costs[edge]
        # 代价字典与边集合一同替换，保证不会留下悬空的 cost 键。
        self.edges = valid_edges
        self.edge_costs = valid_costs

    def build_edges(
        self,
        grid: TraversabilityGrid,
        unknown_clearance: Sequence[float],
        obstacle_clearance: Sequence[float],
    ):
        """执行算法 5：连接半径内且满足净空约束的空间邻居。"""
        # edge_radius 同时是哈希桶尺度和最终精确距离上限。
        node_index = SpatialHashIndex(self.edge_radius)
        node_index.rebuild(
            (idx, pose_xy(node.pose))
            for idx, node in enumerate(self.nodes)
        )
        for from_idx, node in enumerate(self.nodes):
            start_xy = pose_xy(node.pose)
            # 空间哈希先按圆半径粗筛；edge_is_valid 再验证整条线段的每个触及单元。
            for to_idx in node_index.radius_search(
                start_xy,
                self.edge_radius,
            ):
                # 只处理索引升序方向，保证每条无向边最多计算一次。
                if to_idx <= from_idx:
                    continue
                # 因 to_idx > from_idx，元组天然满足无向边的规范化存储约定。
                edge = (from_idx, to_idx)
                self.edge_candidate_count += 1
                # 已有边若通过算法 2 的索引重映射和可选历史验证，就无需在算法 5
                # 再栅格化同一线段。默认欧氏代价也不会随地图内容变化。
                if edge in self.edges:
                    if self.edge_cost_mode == 'integrated_traversability':
                        # 风险积分代价随局部地图变化；历史验证已开启时边刚刚通过
                        # 安全检查，可直接重算代价而不再栅格化一次。
                        edge_valid = self.validate_historical_edges
                        if not edge_valid:
                            self.edge_validation_count += 1
                            edge_valid = self.edge_is_valid(
                                grid,
                                unknown_clearance,
                                obstacle_clearance,
                                from_idx,
                                to_idx,
                                start_xy,
                            )
                        if edge_valid:
                            self.edge_costs[edge] = self.compute_edge_cost(
                                grid,
                                from_idx,
                                to_idx,
                            )
                    elif edge not in self.edge_costs:
                        self.edge_costs[edge] = self.compute_edge_cost(
                            grid,
                            from_idx,
                            to_idx,
                        )
                    continue
                self.edge_validation_count += 1
                if not self.edge_is_valid(
                    grid,
                    unknown_clearance,
                    obstacle_clearance,
                    from_idx,
                    to_idx,
                    start_xy,
                ):
                    continue
                # set.add 对历史已存在的同一边幂等，代价则按当前局部风险重新写入。
                self.edges.add(edge)
                self.edge_costs[edge] = self.compute_edge_cost(
                    grid,
                    from_idx,
                    to_idx,
                )

    def edge_is_valid(
        self,
        grid: TraversabilityGrid,
        unknown_clearance: Sequence[float],
        obstacle_clearance: Sequence[float],
        from_idx: int,
        to_idx: int,
        start_xy: Optional[XY] = None,
    ) -> bool:
        """检查一条候选新边是否满足算法 5 的距离和净空条件。"""
        # 首先过滤越界索引和自环，避免后续访问 nodes 时抛出异常。
        if not self.valid_node_pair(from_idx, to_idx):
            return False
        from_xy = start_xy or pose_xy(self.nodes[from_idx].pose)
        to_xy = pose_xy(self.nodes[to_idx].pose)
        # 新边必须既不超出图的局部连接尺度，也不能穿过未知、障碍或过窄的自由区。
        return (
            distance_xy(from_xy, to_xy) <= self.edge_radius
            and grid.clearance_collision_free(
                from_xy,
                to_xy,
                unknown_clearance,
                obstacle_clearance,
                self.traversable_radius,
            )
        )

    def existing_edge_is_valid(
        self,
        grid: TraversabilityGrid,
        obstacle_clearance: Sequence[float],
        from_idx: int,
        to_idx: int,
    ) -> bool:
        """检查历史边当前可见部分是否仍未被障碍物或狭窄区域否定。"""
        # 历史边也可能因算法 2 的节点删除而成为无效引用。
        if not self.valid_node_pair(from_idx, to_idx):
            return False
        from_xy = pose_xy(self.nodes[from_idx].pose)
        to_xy = pose_xy(self.nodes[to_idx].pose)
        # 超出当前连接半径的旧边不再符合图的尺度合同，直接移除。
        if distance_xy(from_xy, to_xy) > self.edge_radius:
            return False
        return not grid.contradicted_by_obstacle(
            from_xy,
            to_xy,
            obstacle_clearance,
            self.traversable_radius,
        )

    def compute_edge_cost(
        self,
        grid: TraversabilityGrid,
        from_idx: int,
        to_idx: int,
    ) -> float:
        """计算欧氏距离或可选的通行性积分边代价。

        集成模式并非改变可通行性判定，而是在已有效的边上以平均风险放大距离，
        供能处理风险权重的下游规划器选择。
        """
        # 下限保证权重严格为正；距离维度由 edge_distance_mode 控制。
        base_cost = max(
            distance_pose(
                self.nodes[from_idx].pose,
                self.nodes[to_idx].pose,
                self.edge_distance_mode,
            ),
            1e-3,
        )
        if self.edge_cost_mode == 'euclidean':
            # 论文兼容模式：只考虑几何长度。
            return base_cost
        risk = grid.mean_edge_risk(
            pose_xy(self.nodes[from_idx].pose),
            pose_xy(self.nodes[to_idx].pose),
        )
        # risk ∈ [0, 1]；非负权重时该模式不会把边变得比欧氏距离更便宜。
        return base_cost * (
            1.0 + self.traversability_cost_weight * risk
        )

    def nearest_safe_node(
        self,
        robot_xy: XY,
        grid: TraversabilityGrid,
        unknown_clearance: Sequence[float],
        obstacle_clearance: Sequence[float],
    ) -> Optional[int]:
        """返回从机器人位置可安全抵达的最近图节点。"""
        if not self.nodes:
            return None
        # 这里使用 edge_radius 而非节点自由半径：机器人到节点的可达性应与连边一致。
        index = SpatialHashIndex(max(self.edge_radius, 1e-6))
        index.rebuild(
            (idx, pose_xy(node.pose))
            for idx, node in enumerate(self.nodes)
        )

        def safely_reachable(node_idx: int) -> bool:
            """同时满足连边半径和未知/障碍净空的候选节点谓词。"""
            node_xy = pose_xy(self.nodes[node_idx].pose)
            # 机器人到候选节点的“虚拟边”采用与普通图边完全相同的净空定义。
            return (
                distance_xy(robot_xy, node_xy) <= self.edge_radius
                and grid.clearance_collision_free(
                    robot_xy,
                    node_xy,
                    unknown_clearance,
                    obstacle_clearance,
                    self.traversable_radius,
                )
            )

        return index.nearest(robot_xy, predicate=safely_reachable)

    def ensure_robot_anchor_node(
        self,
        grid: TraversabilityGrid,
        unknown_clearance: Sequence[float],
        obstacle_clearance: Sequence[float],
        robot_xy: XY,
    ):
        """当没有安全可达节点时，在机器人位置添加确定性锚点。

        随机采样可能恰好遗漏机器人附近，锚点保证当前机器人在足够净空的自由格
        中仍能接入图；若自身不在安全自由格，则不虚构节点。
        """
        # 每帧重置标志，避免上一次添加锚点的诊断结果泄漏到本帧。
        self.robot_anchor_added = False
        if not self.ensure_robot_anchor:
            return
        if self.nearest_safe_node(
            robot_xy,
            grid,
            unknown_clearance,
            obstacle_clearance,
        ) is not None:
            return

        # 锚点只能落在当前可靠地图内；地图外位置没有足够证据证明安全。
        cell = grid.xy_to_cell(robot_xy)
        if cell is None or not grid.is_free(cell):
            return
        cell_idx = grid.flat_index(cell)
        clearance = min(
            unknown_clearance[cell_idx],
            obstacle_clearance[cell_idx],
        )
        # 即使是自由格，贴近未知/障碍边界也不应创建虚假的安全锚点。
        if clearance <= self.traversable_radius:
            return

        pose = Pose()
        pose.position.x = float(robot_xy[0])
        pose.position.y = float(robot_xy[1])
        pose.position.z = grid.elevation_at_cell(cell, grid.center_z)
        pose.orientation.w = 1.0
        # 锚点复用普通节点结构，后续算法 4/5 会自然为它分配前沿和边。
        self.nodes.append(
            GraphNodeState(
                pose=pose,
                uuid_msg=make_uuid(),
                free_radius=min(clearance, self.max_free_radius),
                explored_radius=unknown_clearance[cell_idx],
            )
        )
        self.robot_anchor_added = True

    def update_current_node(
        self,
        robot_xy: XY,
        grid: Optional[TraversabilityGrid] = None,
        unknown_clearance: Optional[Sequence[float]] = None,
        obstacle_clearance: Optional[Sequence[float]] = None,
    ):
        """选择当前节点：优先安全可达节点，必要时回退到几何最近节点。"""
        # 空图没有可发布的有效索引；保持 ROS 消息层约定的 0 并标记为不安全。
        if not self.nodes:
            self.current_node_idx = 0
            self.current_node_is_safe = False
            return
        if (
            grid is not None
            and unknown_clearance is not None
            and obstacle_clearance is not None
        ):
            nearest_safe = self.nearest_safe_node(
                robot_xy,
                grid,
                unknown_clearance,
                obstacle_clearance,
            )
            if nearest_safe is not None:
                # 有地图且有安全候选时，绝不让隔障碍更近的节点抢占当前索引。
                self.current_node_idx = nearest_safe
                self.current_node_is_safe = True
                return

        # 缺少完整地图数据时仅能提供几何最近回退，并明确标记为不安全。
        index = SpatialHashIndex(max(self.edge_radius, 1e-6))
        index.rebuild(
            (idx, pose_xy(node.pose))
            for idx, node in enumerate(self.nodes)
        )
        nearest = index.nearest(robot_xy)
        self.current_node_idx = nearest if nearest is not None else 0
        self.current_node_is_safe = grid is None

    def graph_diagnostics(self) -> Dict[str, float]:
        """汇总图连通性、前沿可达性和当前节点安全状态。

        通过深度优先遍历计算连通分量及当前分量大小；诊断只读状态，不会修剪图，
        因此可安全用于运行时告警。
        """
        node_count = len(self.nodes)
        edge_count = len(self.edges)
        # 先从边集合构造无向邻接表；无效边索引被防御性忽略。
        adjacency = [set() for _ in range(node_count)]
        for from_idx, to_idx in self.edges:
            if self.valid_node_pair(from_idx, to_idx):
                adjacency[from_idx].add(to_idx)
                adjacency[to_idx].add(from_idx)

        components = 0
        visited = set()
        # 统计全部连通分量，孤立节点也算一个分量。
        for start_idx in range(node_count):
            if start_idx in visited:
                continue
            components += 1
            stack = [start_idx]
            visited.add(start_idx)
            while stack:
                node_idx = stack.pop()
                for neighbor in adjacency[node_idx]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        reachable = set()
        # 单独计算从当前节点出发可达的子图，用于判断前沿是否真的可探索。
        if 0 <= self.current_node_idx < node_count:
            stack = [self.current_node_idx]
            reachable.add(self.current_node_idx)
            while stack:
                node_idx = stack.pop()
                for neighbor in adjacency[node_idx]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        stack.append(neighbor)

        # 仅当节点标志和前沿列表同时存在时才算前沿节点，避免残留布尔状态误报。
        frontier_indices = {
            idx
            for idx, node in enumerate(self.nodes)
            if node.is_frontier and node.frontier_points
        }
        frontier_point_count = sum(
            len(node.frontier_points) for node in self.nodes
        )
        degrees = [len(neighbors) for neighbors in adjacency]
        # 返回纯标量/布尔值，便于 ROS 日志、监控系统与单元测试直接消费。
        return {
            'nodes': node_count,
            'edges': edge_count,
            'components': components,
            'current_component_nodes': len(reachable),
            'frontier_nodes': len(frontier_indices),
            'frontier_points': frontier_point_count,
            'frontier_candidates': self.frontier_candidate_count,
            'frontier_path_checks': self.frontier_path_check_count,
            'historical_frontier_checks': (
                self.historical_frontier_check_count
            ),
            'frontier_unsafe_approaches': (
                self.frontier_unsafe_approach_count
            ),
            'frontier_owner_nodes': self.frontier_owner_node_count,
            'frontier_component_rejects': (
                self.frontier_component_reject_count
            ),
            'edge_candidates': self.edge_candidate_count,
            'edge_validations': self.edge_validation_count,
            'historical_edge_checks': self.historical_edge_check_count,
            'reachable_frontier_nodes': len(frontier_indices & reachable),
            'unreachable_frontier_nodes': len(frontier_indices - reachable),
            'average_degree': (
                sum(degrees) / node_count if node_count else 0.0
            ),
            'max_degree': max(degrees, default=0),
            'current_node_is_safe': self.current_node_is_safe,
            'robot_anchor_added': self.robot_anchor_added,
        }

    def valid_node_pair(self, from_idx: int, to_idx: int) -> bool:
        """判断两个索引是否指向两个不同且仍存在的节点。"""
        # 这是一切索引访问前的共同前置条件，禁止自环也能避免无意义边代价。
        return (
            0 <= from_idx < len(self.nodes)
            and 0 <= to_idx < len(self.nodes)
            and from_idx != to_idx
        )

    def in_workspace_bounds(self, xy: XY) -> bool:
        """检查部署配置的可选工作空间边界，供采样和历史节点保留共用。"""
        # 边界为闭区间；inf 默认值自然表示不限制对应方向。
        return (
            self.min_x <= xy[0] <= self.max_x
            and self.min_y <= xy[1] <= self.max_y
        )
