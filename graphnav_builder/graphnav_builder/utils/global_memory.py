"""用于调试可视化的、有容量上限的全局稀疏通行性记忆。

它不参与论文中的建图算法，也不会构造昂贵的稠密全局栅格；仅为 RViz
保留已经观测到的单元格，避免调试功能反过来改变导航行为。
"""

import math
from typing import Dict, Tuple

from graphnav_builder.utils.traversability_grid import TraversabilityGrid


class GlobalTraversabilityMemory:
    """维护按世界坐标对齐的稀疏调试视图，并严格限制内存占用。"""

    def __init__(self, resolution: float, max_cells: int):
        """以固定分辨率和硬容量上限创建空的稀疏调试记忆。"""
        # 与输入图分辨率对齐，保证相同世界位置总会落到相同的调试单元。
        if resolution <= 0.0:
            raise ValueError('Global memory resolution must be positive')
        if max_cells <= 0:
            raise ValueError('global_memory_max_cells must be positive')
        self.resolution = float(resolution)
        self.max_cells = int(max_cells)
        self.cells: Dict[Tuple[int, int], int] = {}
        self.elevations: Dict[Tuple[int, int], float] = {}
        self.limit_reached = False

    def integrate(self, local_grid: TraversabilityGrid):
        """合并局部已观测单元，而不实例化稠密全局地图。

        未知单元故意跳过：它们没有可靠语义，也会随着局部地图滑动而大量波动。
        当达到上限后，已有单元仍可被新观测覆盖，只有新单元被忽略。
        """
        for cell in local_grid.free_cells + local_grid.obstacle_cells:
            x, y = local_grid.cell_to_xy(cell)
            global_cell = self.xy_to_global_cell(x, y)
            if (
                global_cell not in self.cells
                and len(self.cells) >= self.max_cells
            ):
                self.limit_reached = True
                continue
            self.cells[global_cell] = local_grid.state_at_cell(cell)
            elevation = local_grid.elevation_at_cell(cell, math.nan)
            if math.isfinite(elevation):
                self.elevations[global_cell] = elevation

    def xy_to_global_cell(self, x: float, y: float) -> Tuple[int, int]:
        """把世界坐标映射为固定调试栅格的整数索引。"""
        return (
            int(math.floor(x / self.resolution)),
            int(math.floor(y / self.resolution)),
        )
