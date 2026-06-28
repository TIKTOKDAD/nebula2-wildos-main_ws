"""验证空间哈希索引的精确查询和昂贵谓词剪枝。"""

from graphnav_builder.utils.spatial_index import SpatialHashIndex


def test_nearest_checks_same_bucket_candidates_from_near_to_far():
    """最近有效点确定后，不应再调用同桶更远点的昂贵谓词。"""
    index = SpatialHashIndex(bucket_size=10.0)
    # 故意先插入远点，确保优化不依赖插入顺序。
    index.insert(0, (9.0, 0.0))
    index.insert(1, (1.0, 0.0))
    predicate_calls = []

    nearest = index.nearest(
        (0.0, 0.0),
        predicate=lambda item_idx: (
            predicate_calls.append(item_idx) or True
        ),
    )

    assert nearest == 1
    assert predicate_calls == [1]


def test_nearest_continues_when_closer_candidate_fails_predicate():
    """近点无效时仍须继续检查并返回精确的次近有效点。"""
    index = SpatialHashIndex(bucket_size=10.0)
    index.insert(0, (1.0, 0.0))
    index.insert(1, (3.0, 0.0))
    predicate_calls = []

    def accepts_only_farther(item_idx):
        """记录调用顺序，并只接受索引 1。"""
        predicate_calls.append(item_idx)
        return item_idx == 1

    nearest = index.nearest(
        (0.0, 0.0),
        predicate=accepts_only_farther,
    )

    assert nearest == 1
    assert predicate_calls == [0, 1]


def test_nearest_does_not_expand_buckets_in_sparse_global_graph():
    """最近邻应按真实点距离求解，不再扩展稀疏历史图的桶平面。"""

    class CountingSpatialHashIndex(SpatialHashIndex):
        """记录最近邻查询计算过多少个桶下界。"""

        def __init__(self, bucket_size):
            """创建带桶距离计数器的测试索引。"""
            super().__init__(bucket_size)
            self.bucket_distance_calls = 0

        def bucket_distance(self, point, bucket):
            """计数后复用生产实现的精确桶距离。"""
            self.bucket_distance_calls += 1
            return super().bucket_distance(point, bucket)

    index = CountingSpatialHashIndex(bucket_size=4.0)
    index.insert(0, (-900.0, -900.0))
    index.insert(1, (1000.0, 1000.0))
    index.insert(2, (1004.0, 1000.0))

    nearest = index.nearest((0.0, 0.0))

    assert nearest == 0
    assert index.bucket_distance_calls == 0
