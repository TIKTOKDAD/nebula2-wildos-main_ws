"""供采样、前沿归属和连边使用的二维空间哈希索引。

该结构在需要“附近候选”时避免对全部图节点做线性扫描；最近邻查询使用
桶矩形到查询点的下界进行最佳优先扩展，保证谓词筛选后仍能得到精确最近点。
"""

import heapq
import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from graphnav_builder.utils.graph_data import distance_xy, XY


class SpatialHashIndex:
    """将平面点按固定边长分桶索引。"""

    def __init__(self, bucket_size: float):
        """以给定米制桶边长创建一个空索引。"""
        # 桶边长由调用方的搜索半径选定；必须为正才能定义唯一分桶。
        if bucket_size <= 0.0:
            raise ValueError('SpatialHashIndex bucket_size must be positive')
        self.bucket_size = float(bucket_size)
        self.positions: Dict[int, XY] = {}
        self.buckets: Dict[Tuple[int, int], List[int]] = {}

    def rebuild(self, items: Iterable[Tuple[int, XY]]):
        """清空并用 ``(索引, 坐标)`` 可迭代对象重建索引。"""
        self.positions.clear()
        self.buckets.clear()
        for item_idx, position in items:
            self.insert(item_idx, position)

    def insert(self, item_idx: int, position: XY):
        """插入一个带外部索引的平面点。

        本类按“只追加”使用；同一 ``item_idx`` 不应重复插入，否则旧桶中的
        索引不会自动移除。
        """
        xy = (float(position[0]), float(position[1]))
        self.positions[item_idx] = xy
        bucket = self.bucket_for(xy)
        self.buckets.setdefault(bucket, []).append(item_idx)

    def bucket_for(self, point: XY) -> Tuple[int, int]:
        """返回包含给定点的整数桶坐标，负坐标也用 floor 正确处理。"""
        return (
            int(math.floor(point[0] / self.bucket_size)),
            int(math.floor(point[1] / self.bucket_size)),
        )

    def radius_search(self, point: XY, radius: float) -> List[int]:
        """返回精确欧氏距离不超过 ``radius`` 的所有索引。

        先以查询圆的外包矩形定位候选桶，再做一次真实距离过滤；因此桶边界
        不会带来漏检或把矩形角落的远点错误纳入结果。
        """
        if radius < 0.0 or not self.positions:
            return []
        min_bucket = self.bucket_for((point[0] - radius, point[1] - radius))
        max_bucket = self.bucket_for((point[0] + radius, point[1] + radius))
        result = []
        # 扫描外包矩形涉及的桶，随后用精确圆形距离去除假阳性。
        for bucket_x in range(min_bucket[0], max_bucket[0] + 1):
            for bucket_y in range(min_bucket[1], max_bucket[1] + 1):
                for item_idx in self.buckets.get((bucket_x, bucket_y), []):
                    if distance_xy(point, self.positions[item_idx]) <= radius:
                        result.append(item_idx)
        return result

    def nearest(
        self,
        point: XY,
        predicate: Optional[Callable[[int], bool]] = None,
    ) -> Optional[int]:
        """返回满足可选谓词的精确最近索引。

        堆中的键是“查询点到桶矩形”的最小可能距离；一旦该下界大于当前最佳
        点距离，剩余桶不可能改善结果，因此可安全终止。
        """
        if not self.positions:
            return None

        # predicate 通常是昂贵的栅格碰撞检查。直接按点的真实距离建立最小堆，
        # 第一个满足谓词的点就是严格的全局最近解；这也避免先访问下界较小的桶
        # 时，对桶内实际很远的节点过早执行碰撞检查。
        heap = [
            (distance_xy(point, position), item_idx)
            for item_idx, position in self.positions.items()
        ]
        heapq.heapify(heap)
        while heap:
            _, item_idx = heapq.heappop(heap)
            if predicate is None or predicate(item_idx):
                return item_idx
        return None

    def bucket_distance(self, point: XY, bucket: Tuple[int, int]) -> float:
        """计算查询点到一个半开桶矩形的最小欧氏距离下界。"""
        min_x = bucket[0] * self.bucket_size
        max_x = min_x + self.bucket_size
        min_y = bucket[1] * self.bucket_size
        max_y = min_y + self.bucket_size
        dx = max(min_x - point[0], 0.0, point[0] - max_x)
        dy = max(min_y - point[1], 0.0, point[1] - max_y)
        return math.hypot(dx, dy)
