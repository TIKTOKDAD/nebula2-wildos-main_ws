"""在 ROS 回调与定时处理之间传递消息的有界时间戳缓冲区。"""

from dataclasses import dataclass
import time
from typing import Optional

from builtin_interfaces.msg import Time


@dataclass
class BufferedMessage:
    """保存一对已同步输入及其等待历史 TF 的状态。

    ``enqueue_monotonic`` 只用于本进程内计算等待时长，不能用 ROS 时间，
    因为仿真时间可能暂停或回跳。
    """

    msg: dict
    stamp: Time
    time_flt: float
    enqueue_monotonic: float
    tf_attempts: int = 0
    last_tf_error: str = ''

    def __iter__(self):
        """保留旧适配器使用的三元组解包接口。"""
        return iter((self.msg, self.stamp, self.time_flt))

    def __getitem__(self, index):
        """兼容旧代码和测试对三元组下标访问的依赖。"""
        return (self.msg, self.stamp, self.time_flt)[index]

    def wait_age(self, now: Optional[float] = None) -> float:
        """返回该消息在处理缓冲区等待的秒数，且不返回负数。"""
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.enqueue_monotonic)


class MessageBuffer:
    """在等待时间对齐的 TF 时保存已同步消息。

    实时模式会丢弃最旧项以接收最新观测；严格顺序模式则拒绝新项，确保尚在
    等待 TF 的最旧消息不会被越过。两种策略都显式计数以支持运行时诊断。
    """

    def __init__(self, max_size: int, wait_for_oldest: bool = False):
        """创建有界 FIFO；``wait_for_oldest`` 决定满载时拒绝还是覆盖旧项。"""
        if max_size <= 0:
            raise ValueError('MessageBuffer max_size must be positive')
        self.max_size = int(max_size)
        self.wait_for_oldest = bool(wait_for_oldest)
        self.buffer = []
        self.accepted_count = 0
        self.dropped_overflow_count = 0
        self.rejected_full_count = 0

    def add_msg(
        self,
        msg: dict,
        stamp: Time,
        time_flt: Optional[float] = None,
    ) -> bool:
        """加入一条消息，并返回本次是否被缓冲区接收。"""
        if time_flt is None:
            time_flt = stamp.sec + stamp.nanosec * 1e-9
        if len(self.buffer) >= self.max_size:
            if self.wait_for_oldest:
                # 严格模式：阻塞语义由调用方实现，此处不能静默打乱时序。
                self.rejected_full_count += 1
                return False
            # 实时模式：宁可损失旧观测，也要避免延迟无限积累。
            self.buffer.pop(0)
            self.dropped_overflow_count += 1
        self.buffer.append(
            BufferedMessage(
                msg=msg,
                stamp=stamp,
                time_flt=float(time_flt),
                enqueue_monotonic=time.monotonic(),
            )
        )
        self.accepted_count += 1
        return True

    def get_oldest(self) -> Optional[BufferedMessage]:
        """查看最旧缓冲项但不移除，便于 TF 查询失败后继续重试。"""
        return self.buffer[0] if self.buffer else None

    def pop_oldest(self) -> Optional[BufferedMessage]:
        """移除并返回最旧缓冲项。"""
        return self.buffer.pop(0) if self.buffer else None

    def clear(self):
        """丢弃所有尚未处理的消息。"""
        self.buffer.clear()
