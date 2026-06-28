"""供时间对齐 GridMap 处理使用的二维刚体变换辅助函数。"""

from dataclasses import dataclass
import math
from typing import Tuple

from geometry_msgs.msg import Pose, Quaternion, TransformStamped

from graphnav_builder.utils.graph_data import XY


@dataclass(frozen=True)
class PlanarTransform:
    """表示 ``目标坐标系 <- 源坐标系`` 的平移加偏航角变换。

    图构建只支持 2.5D 地图，故仅保留 XY 平移、Z 偏移和绕 Z 轴的 yaw；
    含 roll/pitch 的 TF 会在转换时被明确拒绝。
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0

    def apply_xy(self, point: XY) -> XY:
        """将源坐标系平面点旋转、平移到目标坐标系。"""
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        return (
            self.x + cos_yaw * point[0] - sin_yaw * point[1],
            self.y + sin_yaw * point[0] + cos_yaw * point[1],
        )

    def apply_z(self, value: float) -> float:
        """把高程加上坐标系间 Z 平移（平面模型不旋转 Z）。"""
        return self.z + float(value)


def quaternion_to_planar_yaw(
    quaternion: Quaternion,
    tolerance: float = 1e-4,
) -> float:
    """从四元数提取 yaw，并拒绝与 2.5D 地图不兼容的 roll/pitch。

    先归一化四元数以容忍上游的数值误差；若倾斜超过容差，继续投影会让
    栅格的“水平距离”失去物理含义，因此直接报错而非悄悄近似。
    """
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError('Quaternion has zero norm')
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    if abs(roll) > tolerance or abs(pitch) > tolerance:
        raise ValueError(
            'Graph construction only supports planar TF roll/pitch'
        )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def planar_transform_from_tf(transform: TransformStamped) -> PlanarTransform:
    """将 ROS ``TransformStamped`` 解析为受限的平面变换。"""
    translation = transform.transform.translation
    return PlanarTransform(
        x=float(translation.x),
        y=float(translation.y),
        z=float(translation.z),
        yaw=quaternion_to_planar_yaw(transform.transform.rotation),
    )


def transform_pose_position(
    pose: Pose,
    transform: PlanarTransform,
) -> Tuple[float, float, float]:
    """只变换位姿的位置，不变换朝向（建图只使用位置）。"""
    x, y = transform.apply_xy(
        (float(pose.position.x), float(pose.position.y))
    )
    return x, y, transform.apply_z(float(pose.position.z))
