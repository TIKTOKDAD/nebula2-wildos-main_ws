"""局部 ``GridMap`` 的解码、坐标变换、碰撞检测与距离场计算。

本模块是 ROS 消息与纯几何建图算法之间的边界：它负责处理 GridMap 的列主序
与循环缓冲区布局，并向上层暴露统一的行主序、全局坐标和保守安全判定。
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from graphnav_builder.utils.graph_data import Cell, XY
from graphnav_builder.utils.transforms import (
    PlanarTransform,
    quaternion_to_planar_yaw,
)
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray


class TraversabilityGrid:
    """在全局坐标系中解码一帧局部几何通行性地图。

单元状态只有三种：``UNKNOWN``（未知）、``FREE``（可通行）与 ``OBSTACLE``
（不可通行）。``active`` 与状态分离：圆形可靠半径外的单元会变成未知且不再
可采样/穿越，但仍可作为前沿路径的最后一个端点。
    """

    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    HIGHER_IS_SAFER = 'higher_is_safer'
    LOWER_IS_SAFER = 'lower_is_safer'
    UNKNOWN_NON_FINITE = 'non_finite'
    UNKNOWN_NEGATIVE = 'negative_or_non_finite'

    def __init__(
        self,
        msg: GridMap,
        traversability_layer: str,
        safe_threshold: float,
        elevation_layer: str = '',
        observed_layer: str = '',
        observed_threshold: float = 0.5,
        traversability_semantics: str = HIGHER_IS_SAFER,
        unknown_value_policy: str = UNKNOWN_NON_FINITE,
        traversability_cost_layer: str = '',
        cost_min: float = 0.0,
        cost_max: float = 1.0,
        cost_higher_is_riskier: bool = True,
        strict_cost_range: bool = False,
        frame_transform: Optional[PlanarTransform] = None,
        output_frame: str = '',
    ):
        """解码一帧 GridMap，并建立分类、坐标和几何查询缓存。

        ``frame_transform`` 把原始地图 frame 转到 ``output_frame``；可选图层分别
        提供高程、观测有效性和风险成本。构造成功后，所有公开查询都使用逻辑行
        主序单元和输出全局坐标系。
        """
        # 先验证消息合同，避免随后索引 data/layers 时出现难以定位的越界错误。
        self.validate_message(
            msg,
            traversability_layer,
            elevation_layer,
            observed_layer,
            traversability_cost_layer,
        )
        if traversability_semantics not in (
            self.HIGHER_IS_SAFER,
            self.LOWER_IS_SAFER,
        ):
            raise ValueError(
                'traversability_semantics must be higher_is_safer '
                'or lower_is_safer'
            )
        if unknown_value_policy not in (
            self.UNKNOWN_NON_FINITE,
            self.UNKNOWN_NEGATIVE,
        ):
            raise ValueError(
                'unknown_value_policy must be non_finite or '
                'negative_or_non_finite'
            )

        # 下游所有几何查询都使用输出（全局）坐标系，而不是 GridMap 原始 frame。
        self.msg = msg
        self.frame_id = output_frame or msg.header.frame_id
        self.layer_name = traversability_layer
        self.safe_threshold = float(safe_threshold)
        self.traversability_semantics = traversability_semantics
        self.unknown_value_policy = unknown_value_policy
        self.cost_min = float(cost_min)
        self.cost_max = float(cost_max)
        self.cost_higher_is_riskier = bool(cost_higher_is_riskier)
        self.strict_cost_range = bool(strict_cost_range)
        if self.cost_max <= self.cost_min:
            raise ValueError('cost_max must be greater than cost_min')
        self.resolution = float(msg.info.resolution)
        self.length_x = float(msg.info.length_x)
        self.length_y = float(msg.info.length_y)
        self.frame_transform = frame_transform or PlanarTransform()

        # 地图自身可能带 yaw，且其 frame 也可能经 TF 转到全局系；二者相加。
        map_pose = msg.info.pose
        map_pose_yaw = quaternion_to_planar_yaw(map_pose.orientation)
        self.map_yaw = self.frame_transform.yaw + map_pose_yaw
        self.center_x, self.center_y = self.frame_transform.apply_xy(
            (
                float(map_pose.position.x),
                float(map_pose.position.y),
            )
        )
        self.center_z = self.frame_transform.apply_z(
            float(map_pose.position.z)
        )
        self.origin_x = self.center_x - 0.5 * self.length_x
        self.origin_y = self.center_y - 0.5 * self.length_y

        layer_idx = msg.layers.index(traversability_layer)
        self.height, self.width = self.layer_shape(msg.data[layer_idx])
        self.cell_radius = 0.5 * math.sqrt(2.0) * self.resolution
        expected_height = int(round(self.length_x / self.resolution))
        expected_width = int(round(self.length_y / self.resolution))
        if self.height != expected_height or self.width != expected_width:
            raise ValueError(
                'GridMap layer dimensions do not match length_x/length_y '
                'and resolution'
            )

        # GridMap 的矩阵可作为循环缓冲区滚动，起点索引用于恢复逻辑矩阵顺序。
        outer_start_index = int(msg.outer_start_index) % self.height
        inner_start_index = int(msg.inner_start_index) % self.width
        self.values = self.decode_named_layer(
            traversability_layer,
            outer_start_index,
            inner_start_index,
        )
        self.elevation_values = (
            self.decode_named_layer(
                elevation_layer,
                outer_start_index,
                inner_start_index,
            )
            if elevation_layer else None
        )
        self.observed_values = (
            self.decode_named_layer(
                observed_layer,
                outer_start_index,
                inner_start_index,
            )
            if observed_layer else None
        )
        self.cost_values = (
            self.decode_named_layer(
                traversability_cost_layer,
                outer_start_index,
                inner_start_index,
            )
            if traversability_cost_layer else None
        )

        # 逻辑状态和“位于可靠区域内”分开保存，以支持圆形裁切边界生成前沿。
        self.state = [self.UNKNOWN] * (self.width * self.height)
        self.active = [True] * (self.width * self.height)
        self.frontier_endpoint_allowed = [
            False
        ] * (self.width * self.height)
        for row in range(self.height):
            for col in range(self.width):
                cell = (row, col)
                idx = self.flat_index(cell)
                value = self.values[idx]
                observed = self.is_observed(idx, observed_threshold)
                # 观测掩码优先级最高：未观测值即使数值看起来安全也只能算未知。
                if not observed or self.value_is_unknown(value):
                    self.state[idx] = self.UNKNOWN
                elif self.value_is_free(value):
                    self.state[idx] = self.FREE
                else:
                    self.state[idx] = self.OBSTACLE
        # 严格模式用于尽早发现上游成本层的量纲/归一化配置错误。
        if self.cost_values is not None and self.strict_cost_range:
            outside_range = [
                value
                for value in self.cost_values
                if (
                    math.isfinite(value)
                    and not self.cost_min <= value <= self.cost_max
                )
            ]
            if outside_range:
                raise ValueError(
                    'Traversability cost layer contains values outside '
                    f'[{self.cost_min}, {self.cost_max}]'
                )
        self.rebuild_cell_lists()

    @staticmethod
    def validate_message(
        msg: GridMap,
        traversability_layer: str,
        elevation_layer: str,
        observed_layer: str,
        traversability_cost_layer: str,
    ):
        """验证 GridMap 的基本几何信息和所需图层名称。

        这里只验证消息结构；层的矩阵维度与循环缓冲区布局由后续解码函数检查。
        """
        if not msg.data:
            raise ValueError('GridMap has no data layers')
        if len(msg.layers) != len(msg.data):
            raise ValueError(
                'GridMap layers and data arrays have different lengths'
        )
        if not traversability_layer:
            raise ValueError(
                'Parameter traversability_layer must not be empty'
            )
        for layer_name, description in (
            (traversability_layer, 'traversability'),
            (elevation_layer, 'elevation'),
            (observed_layer, 'observed'),
            (traversability_cost_layer, 'traversability cost'),
        ):
            if layer_name and layer_name not in msg.layers:
                raise ValueError(
                    f'GridMap does not contain {description} layer '
                    f"'{layer_name}'"
                )
        if msg.info.resolution <= 0.0:
            raise ValueError('GridMap resolution must be positive')

    def decode_named_layer(
        self,
        layer_name: str,
        outer_start_index: int,
        inner_start_index: int,
    ) -> List[float]:
        """按名称找到并解码一个 GridMap 图层为逻辑行主序数组。"""
        return self.decode_layer(
            self.msg.data[self.msg.layers.index(layer_name)],
            self.height,
            self.width,
            outer_start_index,
            inner_start_index,
        )

    def is_observed(self, idx: int, threshold: float) -> bool:
        """依据可选观测掩码判断一个单元是否可靠地被观测到。"""
        if self.observed_values is None:
            return True
        value = self.observed_values[idx]
        return math.isfinite(value) and value >= threshold

    def value_is_free(self, value: float) -> bool:
        """按配置的数值语义判断一个有限通行性值是否安全。"""
        if self.traversability_semantics == self.HIGHER_IS_SAFER:
            return value >= self.safe_threshold
        return value <= self.safe_threshold

    def value_is_unknown(self, value: float) -> bool:
        """按输入合同判断通行性值是否表示未知区域。"""
        return (
            not math.isfinite(value)
            or (
                self.unknown_value_policy == self.UNKNOWN_NEGATIVE
                and value < 0.0
            )
        )

    def rebuild_cell_lists(self):
        """根据当前状态重建自由、未知、障碍单元列表。

        圆形掩码会原地改变状态，因而每次掩码后都必须调用本函数，使采样和
        距离场使用同一份最新分类结果。
        """
        self.free_cells = []
        self.unknown_cells = []
        self.obstacle_cells = []
        for row in range(self.height):
            for col in range(self.width):
                cell = (row, col)
                state = self.state_at_cell(cell)
                if state == self.FREE:
                    self.free_cells.append(cell)
                elif state == self.OBSTACLE:
                    self.obstacle_cells.append(cell)
                else:
                    self.unknown_cells.append(cell)

    def apply_circular_mask(self, center: XY, radius: float):
        """将可靠局部半径外的单元标记为不可用的未知区。

        论文的可靠感知范围是圆形，而 GridMap 存储通常是矩形。被裁掉的单元
        不能用于节点或边，却保留“允许作为前沿终点”的标志，从而不会把圆形
        感知边界意外封死。
        """
        if not math.isfinite(radius):
            return
        if radius <= 0.0:
            raise ValueError('local_map_radius must be positive')
        radius_squared = radius * radius
        for row in range(self.height):
            for col in range(self.width):
                cell = (row, col)
                x, y = self.cell_to_xy(cell)
                if (
                    (x - center[0]) ** 2 + (y - center[1]) ** 2
                    > radius_squared
                ):
                    idx = self.flat_index(cell)
                    # 与真实未知区不同，这类单元只允许作为路径终点，不能穿越。
                    self.active[idx] = False
                    self.frontier_endpoint_allowed[idx] = True
                    self.state[idx] = self.UNKNOWN
        self.rebuild_cell_lists()

    def flat_index(self, cell: Cell) -> int:
        """把 ``(row, column)`` 矩阵坐标转换为内部行主序一维索引。"""
        return cell[0] * self.width + cell[1]

    @staticmethod
    def layer_shape(array_msg: Float32MultiArray) -> Tuple[int, int]:
        """从 GridMap 多维数组布局读取逻辑矩阵的行数和列数。

        ``grid_map`` 常用 column/row 标签，也兼容常规 row/column 标签；其余
        布局无法安全解释，故明确拒绝而不猜测数据顺序。
        """
        dims = array_msg.layout.dim
        if (
            len(dims) >= 2
            and dims[0].label == 'column_index'
            and dims[1].label == 'row_index'
        ):
            return max(1, dims[1].size), max(1, dims[0].size)
        if (
            len(dims) >= 2
            and dims[0].label == 'row_index'
            and dims[1].label == 'column_index'
        ):
            return max(1, dims[0].size), max(1, dims[1].size)
        raise ValueError(
            'GridMap layer layout must be column_index/row_index '
            'or row_index/column_index'
        )

    @staticmethod
    def decode_layer(
        array_msg: Float32MultiArray,
        rows: int,
        cols: int,
        outer_start_index: int,
        inner_start_index: int,
    ) -> List[float]:
        """解码并展开一个可能使用循环缓冲区的 ``Float32MultiArray`` 图层。

        返回值始终是逻辑行主序 ``values[row * cols + col]``，屏蔽上游的物理
        存储顺序、``data_offset`` 以及滚动起点差异。
        """
        data = list(array_msg.data)
        expected = rows * cols
        dims = array_msg.layout.dim
        data_offset = int(array_msg.layout.data_offset)
        if data_offset < 0 or data_offset + expected > len(data):
            raise ValueError(
                'GridMap layer data is shorter than its declared dimensions'
            )

        label0 = dims[0].label if len(dims) > 0 else ''
        label1 = dims[1].label if len(dims) > 1 else ''
        # 先定义“物理数组如何寻址”，再在下方统一做循环缓冲区反卷绕。
        if label0 == 'column_index' and label1 == 'row_index':

            def physical_value(row: int, col: int) -> float:
                """按列主序物理布局读取一个未反卷绕的单元值。"""
                return data[data_offset + col * rows + row]

        elif label0 == 'row_index' and label1 == 'column_index':

            def physical_value(row: int, col: int) -> float:
                """按行主序物理布局读取一个未反卷绕的单元值。"""
                return data[data_offset + row * cols + col]

        else:
            raise ValueError('Unsupported GridMap layer storage order')

        values = [float('nan')] * expected
        # logical = (0, 0) 对应当前窗口左上角，而非底层循环数组的索引 0。
        for row in range(rows):
            physical_row = (row + outer_start_index) % rows
            for col in range(cols):
                physical_col = (col + inner_start_index) % cols
                values[row * cols + col] = float(
                    physical_value(physical_row, physical_col)
                )
        return values

    def value_at_cell(self, cell: Cell) -> float:
        """返回一个单元未经分类的原始通行性数值。"""
        return self.values[self.flat_index(cell)]

    def state_at_cell(self, cell: Cell) -> int:
        """返回单元的 ``UNKNOWN``、``FREE`` 或 ``OBSTACLE`` 离散状态。"""
        return self.state[self.flat_index(cell)]

    def elevation_at_cell(self, cell: Cell, fallback: float = 0.0) -> float:
        """返回单元在全局坐标系的有限高程，缺失时使用给定回退值。"""
        if self.elevation_values is None or not self.in_bounds_cell(cell):
            return float(fallback)
        elevation = self.elevation_values[self.flat_index(cell)]
        if not math.isfinite(elevation):
            return float(fallback)
        return self.frame_transform.apply_z(elevation)

    def in_bounds_cell(self, cell: Cell) -> bool:
        """判断矩阵坐标是否位于局部地图矩形范围内。"""
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def world_to_map_axes(self, xy: XY) -> XY:
        """将全局点逆旋转到 GridMap 未旋转的局部地图轴。"""
        dx = xy[0] - self.center_x
        dy = xy[1] - self.center_y
        cos_yaw = math.cos(self.map_yaw)
        sin_yaw = math.sin(self.map_yaw)
        return (
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
        )

    def map_axes_to_world(self, xy: XY) -> XY:
        """将地图轴中的点旋转、平移到全局坐标系。"""
        cos_yaw = math.cos(self.map_yaw)
        sin_yaw = math.sin(self.map_yaw)
        return (
            self.center_x + cos_yaw * xy[0] - sin_yaw * xy[1],
            self.center_y + sin_yaw * xy[0] + cos_yaw * xy[1],
        )

    def xy_to_cell(self, xy: XY) -> Optional[Cell]:
        """把全局 XY 坐标转换为局部矩阵单元；矩形外返回 ``None``。

        GridMap 的行/列方向与常见图像坐标相反：行和列都从地图的正半轴向负半轴
        递增，因此必须先转到地图轴，不能直接用全局 X/Y 除以分辨率。
        """
        map_x, map_y = self.world_to_map_axes(xy)
        row = int(
            math.floor((0.5 * self.length_x - map_x) / self.resolution)
        )
        col = int(
            math.floor((0.5 * self.length_y - map_y) / self.resolution)
        )
        cell = (row, col)
        return cell if self.in_bounds_cell(cell) else None

    def cell_to_xy(self, cell: Cell) -> XY:
        """返回局部矩阵单元中心的全局 XY 坐标。"""
        map_xy = (
            0.5 * self.length_x - (cell[0] + 0.5) * self.resolution,
            0.5 * self.length_y - (cell[1] + 0.5) * self.resolution,
        )
        return self.map_axes_to_world(map_xy)

    def frontier_key(self, cell: Cell) -> Tuple[str, int, int]:
        """返回当前 GridMap 单元的前沿去重键。

        使用帧名和逻辑行列，而不对世界坐标量化；这在旋转或滚动 GridMap 中
        保证同一帧的前沿单元不会因浮点误差发生键碰撞。
        """
        return self.frame_id, int(cell[0]), int(cell[1])

    def is_free(self, cell: Cell) -> bool:
        """判断单元是否处于可靠区域内且被分类为可通行。"""
        return (
            self.is_active(cell)
            and self.state_at_cell(cell) == self.FREE
        )

    def is_active(self, cell: Cell) -> bool:
        """判断单元是否位于可靠的局部地图区域内。"""
        return (
            self.in_bounds_cell(cell)
            and self.active[self.flat_index(cell)]
        )

    def is_frontier_endpoint_allowed(self, cell: Cell) -> bool:
        """判断未知单元能否作为前沿路径的最终端点。

        可靠区域内的未知单元和圆形裁切产生的未知边界均允许；但路径的前置
        单元仍由 ``is_free`` 强制要求位于可靠自由区域。
        """
        return (
            self.in_bounds_cell(cell)
            and (
                self.is_active(cell)
                or self.frontier_endpoint_allowed[self.flat_index(cell)]
            )
        )

    def is_known(self, cell: Cell) -> bool:
        """判断可靠区域内的单元是否已经被观测为自由或障碍。"""
        return (
            self.is_active(cell)
            and self.state_at_cell(cell) in (self.FREE, self.OBSTACLE)
        )

    def reachable_free_cells(self, seed_xy: XY) -> List[Cell]:
        """返回从世界坐标种子可达的四连通可靠自由空间分量。

        机器人必须位于当前可靠区域的 Free 单元内；否则返回空列表，避免在
        无法证明与机器人连通的区域盲目采样。四邻域不会把仅在角点接触的两个
        自由区域误判为可通行，节点自身的净空要求仍由上层采样算法检查。
        """
        seed_cell = self.xy_to_cell(seed_xy)
        if seed_cell is None or not self.is_free(seed_cell):
            return []

        reachable = []
        visited = {seed_cell}
        stack = [seed_cell]
        while stack:
            cell = stack.pop()
            reachable.append(cell)
            for neighbor in self.neighbors4(cell):
                if neighbor in visited or not self.is_free(neighbor):
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        return reachable

    def unknown_frontier_cells_next_to_free(
        self,
        free_cells: Optional[Sequence[Cell]] = None,
        connectivity: int = 4,
    ) -> List[Tuple[Cell, List[Cell]]]:
        """找出与自由单元相邻的未知前沿单元及其自由侧单元。

        返回 ``(未知单元, [相邻自由单元...])``，后者供上层验证路径倒数第二格
        确实从自由侧接近前沿。4 邻接不把角点接触当成前沿，8 邻接则允许。
        """
        if connectivity not in (4, 8):
            raise ValueError('frontier connectivity must be 4 or 8')
        frontier_to_free_cells: Dict[Cell, List[Cell]] = {}
        candidates = free_cells if free_cells is not None else self.free_cells
        neighbor_function = (
            self.neighbors4 if connectivity == 4 else self.neighbors8
        )
        # 常见滚动高程图中 free cell 数以万计，而 unknown 主要集中在地图边界。
        # 未显式限制 free_cells 时，从数量较少的一侧扫描可保持完全相同的邻接
        # 定义，同时显著减少每帧 Python 邻居检查次数。
        if (
            free_cells is None
            and len(self.unknown_cells) < len(self.free_cells)
        ):
            for unknown_cell in self.unknown_cells:
                if not self.is_frontier_endpoint_allowed(unknown_cell):
                    continue
                adjacent_free_cells = [
                    neighbor
                    for neighbor in neighbor_function(unknown_cell)
                    if self.is_free(neighbor)
                ]
                if adjacent_free_cells:
                    frontier_to_free_cells[unknown_cell] = (
                        adjacent_free_cells
                    )
            return list(frontier_to_free_cells.items())

        # 同一个未知格可能邻接多个自由格，用字典聚合以避免重复前沿。
        for free_cell in candidates:
            for neighbor in neighbor_function(free_cell):
                if (
                    self.is_frontier_endpoint_allowed(neighbor)
                    and self.state_at_cell(neighbor) == self.UNKNOWN
                ):
                    frontier_to_free_cells.setdefault(neighbor, []).append(
                        free_cell
                    )
        return list(frontier_to_free_cells.items())

    @staticmethod
    def neighbors4(cell: Cell) -> Sequence[Cell]:
        """返回与单元共享边的四个矩阵邻居（不做边界裁剪）。"""
        row, col = cell
        return (
            (row - 1, col),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col),
        )

    @staticmethod
    def neighbors8(cell: Cell) -> Sequence[Cell]:
        """返回单元周围八个邻居（包含对角且不做边界裁剪）。"""
        row, col = cell
        return (
            (row - 1, col - 1),
            (row - 1, col),
            (row - 1, col + 1),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col - 1),
            (row + 1, col),
            (row + 1, col + 1),
        )

    def clip_segment_to_bounds(
        self,
        start: XY,
        end: XY,
    ) -> Optional[Tuple[XY, XY]]:
        """将全局线段裁剪到旋转后的 GridMap 矩形内部。

        先把端点转回地图轴，再使用 Liang--Barsky 参数裁剪。轻微内缩边界可
        避免恰好落在上边/右边时被 ``floor`` 映射到矩阵外。
        """
        start_local = self.world_to_map_axes(start)
        end_local = self.world_to_map_axes(end)
        dx = end_local[0] - start_local[0]
        dy = end_local[1] - start_local[1]
        inset = max(1e-9, self.resolution * 1e-9)
        x_min = -0.5 * self.length_x + inset
        x_max = 0.5 * self.length_x - inset
        y_min = -0.5 * self.length_y + inset
        y_max = 0.5 * self.length_y - inset
        lower = 0.0
        upper = 1.0

        # 对四个半平面逐一收紧线段参数区间 [lower, upper]。
        for p_value, q_value in (
            (-dx, start_local[0] - x_min),
            (dx, x_max - start_local[0]),
            (-dy, start_local[1] - y_min),
            (dy, y_max - start_local[1]),
        ):
            if abs(p_value) <= 1e-15:
                if q_value < 0.0:
                    return None
                continue
            ratio = q_value / p_value
            if p_value < 0.0:
                lower = max(lower, ratio)
            else:
                upper = min(upper, ratio)
            if lower > upper:
                return None

        clipped_start = (
            start_local[0] + lower * dx,
            start_local[1] + lower * dy,
        )
        clipped_end = (
            start_local[0] + upper * dx,
            start_local[1] + upper * dy,
        )
        return (
            self.map_axes_to_world(clipped_start),
            self.map_axes_to_world(clipped_end),
        )

    def line_cells(
        self,
        start: XY,
        end: XY,
        clip_to_bounds: bool = False,
    ) -> Optional[List[Cell]]:
        """以 supercover 语义将线段栅格化为所有触及的单元。

        ``clip_to_bounds`` 用于历史边：仅验证当前局部图可见的部分，地图外的
        未观测历史区域不应仅因为暂时不可见而删除边。
        """
        if clip_to_bounds:
            clipped = self.clip_segment_to_bounds(start, end)
            if clipped is None:
                return None
            start, end = clipped

        start_cell = self.xy_to_cell(start)
        end_cell = self.xy_to_cell(end)
        if start_cell is None or end_cell is None:
            return None
        return self.supercover_cells(start_cell, end_cell)

    def supercover_cells(self, start: Cell, end: Cell) -> List[Cell]:
        """返回中心到中心线段触及的全部单元（包含角点擦过的单元）。

        普通 Bresenham 在对角穿越格点时会漏掉两个侧邻单元；这里在决策值为零
        时显式追加它们，从而以保守方式避免边穿过角落障碍物。
        """
        return list(self.iter_supercover_cells(start, end))

    def iter_supercover_cells(self, start: Cell, end: Cell):
        """按线段顺序惰性生成 supercover 单元，允许碰撞检查提前停止。"""
        row, col = start
        end_row, end_col = end
        delta_col = end_col - col
        delta_row = end_row - row
        num_col = abs(delta_col)
        num_row = abs(delta_row)
        step_col = 1 if delta_col > 0 else -1
        step_row = 1 if delta_row > 0 else -1
        col_count = 0
        row_count = 0
        seen = set()
        initial = (row, col)
        if self.in_bounds_cell(initial):
            seen.add(initial)
            yield initial
        while col_count < num_col or row_count < num_row:
            decision = (
                (1 + 2 * col_count) * num_row
                - (1 + 2 * row_count) * num_col
            )
            if decision == 0:
                # 线段正好穿过格点：两个正交侧格都必须纳入碰撞检查。
                previous_row, previous_col = row, col
                col += step_col
                row += step_row
                col_count += 1
                row_count += 1
                step_cells = (
                    (previous_row, col),
                    (row, previous_col),
                )
            elif decision < 0:
                col += step_col
                col_count += 1
                step_cells = ()
            else:
                row += step_row
                row_count += 1
                step_cells = ()
            for cell in (*step_cells, (row, col)):
                if self.in_bounds_cell(cell) and cell not in seen:
                    seen.add(cell)
                    yield cell

    def collision_free(self, start: XY, end: XY) -> bool:
        """检查线段触及的每个单元是否都是可靠自由单元。"""
        cells = self.line_cells(start, end)
        return cells is not None and all(self.is_free(cell) for cell in cells)

    def path_cells_to_frontier(
        self,
        start: XY,
        frontier_cell: Cell,
    ) -> Optional[List[Cell]]:
        """返回到未知前沿的有效路径，且仅最后一格可以是未知。

        前沿点本身按定义位于未知区，不能套用普通 ``collision_free``；该专用
        规则确保机器人只在自由区行进，并停在感知边界前。
        """
        if (
            not self.is_frontier_endpoint_allowed(frontier_cell)
            or self.state_at_cell(frontier_cell) != self.UNKNOWN
        ):
            return None
        cells = self.line_cells(start, self.cell_to_xy(frontier_cell))
        if (
            not cells
            or cells[-1] != frontier_cell
            or len(cells) < 2
            or not all(self.is_free(cell) for cell in cells[:-1])
        ):
            return None
        return cells

    def collision_free_to_frontier(
        self,
        start: XY,
        frontier_cell: Cell,
    ) -> bool:
        """判断是否存在一条仅允许终点未知的自由前沿路径。"""
        return self.path_cells_to_frontier(start, frontier_cell) is not None

    def clearance_path_cells_to_frontier(
        self,
        start: XY,
        frontier_cell: Cell,
        obstacle_clearance: Sequence[float],
        clearance: float,
    ) -> Optional[List[Cell]]:
        """返回末端为未知前沿、且机器人足迹安全的路径。

        终点未知格不要求障碍物距离；其余自由格都必须严格大于机器人
        ``clearance``，从而给边界接近保留足够安全余量。
        """
        cells = self.path_cells_to_frontier(start, frontier_cell)
        if cells is None:
            return None

        # 最后一格是未知前沿端点；之前每一格都必须有足够的障碍物净空。
        for cell in cells[:-1]:
            if (
                obstacle_clearance[self.flat_index(cell)]
                <= clearance
            ):
                return None
        return cells

    def clearance_approach_cell_to_frontier(
        self,
        start: XY,
        frontier_cell: Cell,
        obstacle_clearance: Sequence[float],
        clearance: float,
    ) -> Optional[Cell]:
        """提前终止地验证前沿直线路径，并返回终点前的自由单元。

        语义与 ``clearance_path_cells_to_frontier`` 相同，但不会先分配完整路径；
        对大量不可达 frontier，遇到首个未知、障碍或净空不足单元即可返回。
        """
        if (
            not self.is_frontier_endpoint_allowed(frontier_cell)
            or self.state_at_cell(frontier_cell) != self.UNKNOWN
        ):
            return None
        start_cell = self.xy_to_cell(start)
        if start_cell is None:
            return None

        # 这是 Algorithm 4 最热的循环。直接在整数栅格上执行与
        # iter_supercover_cells 相同的遍历，避免为每条失败路径创建生成器、
        # seen 集合，并避免逐格重复调用 is_free/flat_index。
        row, col = start_cell
        end_row, end_col = frontier_cell
        delta_col = end_col - col
        delta_row = end_row - row
        num_col = abs(delta_col)
        num_row = abs(delta_row)
        step_col = 1 if delta_col > 0 else -1
        step_row = 1 if delta_row > 0 else -1
        col_count = 0
        row_count = 0
        previous_cell = None
        width = self.width
        state = self.state
        active = self.active
        free_state = self.FREE

        while True:
            cell_idx = row * width + col
            if (row, col) == frontier_cell:
                return previous_cell
            if (
                not active[cell_idx]
                or state[cell_idx] != free_state
                or obstacle_clearance[cell_idx] <= clearance
            ):
                return None
            previous_cell = (row, col)
            if col_count >= num_col and row_count >= num_row:
                return None

            decision = (
                (1 + 2 * col_count) * num_row
                - (1 + 2 * row_count) * num_col
            )
            if decision == 0:
                previous_row, previous_col = row, col
                col += step_col
                row += step_row
                col_count += 1
                row_count += 1
                # 正好穿过格点时，两个正交侧格都属于 supercover；任一侧不安全
                # 都必须拒绝，不能允许机器人从障碍物角点之间斜穿。
                for side_row, side_col in (
                    (previous_row, col),
                    (row, previous_col),
                ):
                    side_idx = side_row * width + side_col
                    if (
                        not active[side_idx]
                        or state[side_idx] != free_state
                        or obstacle_clearance[side_idx] <= clearance
                    ):
                        return None
                    previous_cell = (side_row, side_col)
            elif decision < 0:
                col += step_col
                col_count += 1
            else:
                row += step_row
                row_count += 1

    def frontier_approach_cell(
        self,
        start: XY,
        frontier_cell: Cell,
    ) -> Optional[Cell]:
        """返回未知前沿端点前的最后一个自由单元。"""
        cells = self.path_cells_to_frontier(start, frontier_cell)
        return cells[-2] if cells is not None else None

    def clearance_collision_free(
        self,
        start: XY,
        end: XY,
        unknown_clearance: Sequence[float],
        obstacle_clearance: Sequence[float],
        clearance: float,
    ) -> bool:
        """检查一条自由线段，并同时满足障碍与未知区域的净空约束。"""
        cells = self.line_cells(start, end)
        if cells is None:
            return False
        for cell in cells:
            if not self.is_free(cell):
                return False
            idx = self.flat_index(cell)
            # 节点/边都必须远离未知区和障碍物，因此取两种距离的较小值。
            if min(
                unknown_clearance[idx],
                obstacle_clearance[idx],
            ) <= clearance:
                return False
        return True

    def safe_free_space_components(
        self,
        obstacle_clearance: Sequence[float],
        clearance: float,
    ) -> List[int]:
        """标记满足机器人障碍净空的四连通自由空间分量。

        返回数组与逻辑栅格行主序一致，``-1`` 表示该格不是安全自由格。任何
        collision-free supercover 直线的起点和终点前自由格必然属于同一分量，
        因而该标签可作为精确路径检查之前的必要条件剪枝。
        """
        labels = [-1] * (self.height * self.width)
        component_idx = 0
        for start_cell in self.free_cells:
            start_idx = self.flat_index(start_cell)
            if (
                labels[start_idx] >= 0
                or obstacle_clearance[start_idx] <= clearance
            ):
                continue
            labels[start_idx] = component_idx
            stack = [start_cell]
            while stack:
                cell = stack.pop()
                for neighbor in self.neighbors4(cell):
                    if not self.is_free(neighbor):
                        continue
                    neighbor_idx = self.flat_index(neighbor)
                    if (
                        labels[neighbor_idx] >= 0
                        or obstacle_clearance[neighbor_idx] <= clearance
                    ):
                        continue
                    labels[neighbor_idx] = component_idx
                    stack.append(neighbor)
            component_idx += 1
        return labels

    def contradicted_by_obstacle(
        self,
        start: XY,
        end: XY,
        obstacle_clearance: Sequence[float],
        clearance: float,
    ) -> bool:
        """检查历史边在当前地图可见部分是否被新障碍物否定。

        对矩形裁剪后再检查：未知或矩形外部分保持历史信任，只有新观测到的
        障碍物或过窄的已知自由区才会让已有边失效。
        """
        cells = self.line_cells(start, end, clip_to_bounds=True)
        if cells is None:
            return False
        for cell in cells:
            idx = self.flat_index(cell)
            if self.state[idx] == self.OBSTACLE:
                return True
            if (
                self.state[idx] == self.FREE
                and obstacle_clearance[idx] <= clearance
            ):
                return True
        return False

    def distance_to_cell_boundary(self, center_distance: float) -> float:
        """把中心点距离场转换为到单元边界的保守净空距离。"""
        if not math.isfinite(center_distance):
            return math.inf
        return max(0.0, center_distance - self.cell_radius)

    def clearance_field(
        self,
        targets: Sequence[Cell],
        include_map_exterior: bool = False,
    ) -> List[float]:
        """计算到目标单元边界的保守距离场（单位：米）。"""
        return [
            self.distance_to_cell_boundary(distance)
            for distance in self.distance_field(
                targets,
                include_map_exterior=include_map_exterior,
            )
        ]

    def distance_field(
        self,
        targets: Sequence[Cell],
        include_map_exterior: bool = False,
    ) -> List[float]:
        """计算精确的中心到中心欧氏距离场（单位：米）。

        先在行、列两个维度分别应用一维平方距离变换，得到线性时间的二维 EDT。
        可选的一圈外部零点把“地图外未知”纳入到未知区域的距离计算。
        """
        # 没有目标且不把地图外视为目标时，结果严格为全无穷；无需再运行两遍
        # Python 一维 EDT。平坦、无障碍的局部地图会频繁命中这个安全快路径。
        if not targets and not include_map_exterior:
            return [math.inf] * (self.height * self.width)

        padding = 1 if include_map_exterior else 0
        rows = self.height + 2 * padding
        cols = self.width + 2 * padding
        squared = [math.inf] * (rows * cols)

        if include_map_exterior:
            # 填充一圈虚拟目标，表示矩形地图以外是未观测/不可靠区域。
            for row in range(rows):
                squared[row * cols] = 0.0
                squared[row * cols + cols - 1] = 0.0
            for col in range(cols):
                squared[col] = 0.0
                squared[(rows - 1) * cols + col] = 0.0

        for row, col in targets:
            if self.in_bounds_cell((row, col)):
                squared[(row + padding) * cols + col + padding] = 0.0

        # 可分离 EDT：先逐列变换，再逐行变换，与直接二维搜索等价但更高效。
        for col in range(cols):
            values = [squared[row * cols + col] for row in range(rows)]
            transformed = self.squared_distance_transform_1d(values)
            for row, value in enumerate(transformed):
                squared[row * cols + col] = value
        for row in range(rows):
            start = row * cols
            squared[start:start + cols] = self.squared_distance_transform_1d(
                squared[start:start + cols]
            )

        distances = [math.inf] * (self.height * self.width)
        for row in range(self.height):
            for col in range(self.width):
                value = squared[(row + padding) * cols + col + padding]
                distances[self.flat_index((row, col))] = (
                    math.sqrt(value) * self.resolution
                    if math.isfinite(value)
                    else math.inf
                )
        return distances

    @staticmethod
    def squared_distance_transform_1d(
        values: Sequence[float],
    ) -> List[float]:
        """返回一维精确平方欧氏距离变换。

        维护所有抛物线 ``(q-site)^2 + values[site]`` 的下包络；``boundaries``
        是相邻最优 site 的切换位置，因此每个维度可在线性时间完成。
        """
        finite_indices = [
            index for index, value in enumerate(values) if math.isfinite(value)
        ]
        if not finite_indices:
            return [math.inf] * len(values)

        sites = [finite_indices[0]]
        boundaries = [-math.inf, math.inf]
        for query in finite_indices[1:]:
            # 新 site 若在前一段开始前就更优，前一 site 永远不会成为最优点。
            while True:
                site = sites[-1]
                intersection = (
                    (values[query] + query * query)
                    - (values[site] + site * site)
                ) / (2.0 * (query - site))
                if intersection > boundaries[-2]:
                    break
                sites.pop()
                boundaries.pop(-2)
            sites.append(query)
            boundaries.insert(-1, intersection)

        transformed = [math.inf] * len(values)
        site_index = 0
        for query in range(len(values)):
            while boundaries[site_index + 1] < query:
                site_index += 1
            site = sites[site_index]
            transformed[query] = (query - site) ** 2 + values[site]
        return transformed

    def risk_at_cell(self, cell: Cell) -> float:
        """返回单元归一化风险值，用于可选的通行性积分边代价。

        未知或非有限成本以最大风险处理。若未配置专用成本层，则从通行性层
        推导风险，并依据“值越大越安全/危险”的语义决定是否反向。
        """
        idx = self.flat_index(cell)
        value = (
            self.cost_values[idx]
            if self.cost_values is not None
            else self.values[idx]
        )
        if not math.isfinite(value):
            return 1.0
        if self.cost_values is not None:
            normalized = (
                (float(value) - self.cost_min)
                / (self.cost_max - self.cost_min)
            )
            normalized = max(0.0, min(1.0, normalized))
            return (
                normalized
                if self.cost_higher_is_riskier
                else 1.0 - normalized
            )

        normalized = max(0.0, min(1.0, float(value)))
        if self.traversability_semantics == self.HIGHER_IS_SAFER:
            return 1.0 - normalized
        return normalized

    def mean_edge_risk(self, start: XY, end: XY) -> float:
        """返回局部边覆盖单元的平均归一化风险。"""
        cells = self.line_cells(start, end)
        if not cells:
            return 0.0
        return sum(self.risk_at_cell(cell) for cell in cells) / len(cells)

    def boundary_cells(self) -> List[Cell]:
        """返回 GridMap 矩形外边界上不重复的全部单元。"""
        cells = set()
        for row in range(self.height):
            cells.add((row, 0))
            cells.add((row, self.width - 1))
        for col in range(self.width):
            cells.add((0, col))
            cells.add((self.height - 1, col))
        return sorted(cells)

    def boundary_state_counts(self) -> Dict[str, int]:
        """统计矩形边界上的自由、障碍、未知单元数量，用于输入合同诊断。"""
        counts = {'free': 0, 'obstacle': 0, 'unknown': 0}
        for cell in self.boundary_cells():
            state = self.state_at_cell(cell)
            if state == self.FREE:
                counts['free'] += 1
            elif state == self.OBSTACLE:
                counts['obstacle'] += 1
            else:
                counts['unknown'] += 1
        return counts
