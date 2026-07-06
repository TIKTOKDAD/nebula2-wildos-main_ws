"""
Odometry twist 坐标转换的纯函数.

``nav_msgs/msg/Odometry`` 的约定是：

* ``pose`` 表达在 ``header.frame_id`` 中；
* ``twist`` 表达在 ``child_frame_id`` 中。

当前 DLIO 消息的 pose 满足第一条，但实测 twist 仍沿 odom 世界坐标轴表达。
本模块利用 pose 四元数表示的 ``world_from_body`` 旋转，把 twist 变换为
``body_from_world``，同时对 6x6 twist 协方差执行同一个基变换。
"""

from copy import deepcopy
import math
from typing import Sequence, Tuple

from nav_msgs.msg import Odometry


Matrix3 = Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]


def world_from_body_rotation(orientation) -> Matrix3:
    """
    把归一化四元数转换为 ``world_from_body`` 三维旋转矩阵.

    Odometry pose 的四元数描述车体坐标系在世界坐标系中的姿态。因此该矩阵
    直接作用于车体系向量会得到世界系向量；速度修正时使用它的转置执行逆旋转。
    """
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("Odometry pose orientation has a zero-norm quaternion")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def transpose(matrix: Matrix3) -> Matrix3:
    """返回 3x3 矩阵的转置，旋转矩阵的转置也就是其逆矩阵."""
    return tuple(
        tuple(matrix[column][row] for column in range(3))
        for row in range(3)
    )


def rotate_xyz(vector, rotation: Matrix3) -> Tuple[float, float, float]:
    """使用给定旋转矩阵变换 geometry_msgs/Vector3."""
    values = (float(vector.x), float(vector.y), float(vector.z))
    return tuple(
        sum(rotation[row][column] * values[column] for column in range(3))
        for row in range(3)
    )


def rotate_twist_covariance(
    covariance: Sequence[float],
    body_from_world: Matrix3,
) -> list[float]:
    """
    将 6x6 twist 协方差从世界基旋转到车体基.

    ROS TwistWithCovariance 的变量顺序为 ``[vx, vy, vz, wx, wy, wz]``。
    因而变换矩阵是两个 ``body_from_world`` 的块对角矩阵：
    ``T = diag(R, R)``，新协方差为 ``T * C * T^T``。
    """
    if len(covariance) != 36:
        raise ValueError("Twist covariance must contain exactly 36 elements")

    transform = [[0.0 for _ in range(6)] for _ in range(6)]
    for block_start in (0, 3):
        for row in range(3):
            for column in range(3):
                transform[block_start + row][block_start + column] = (
                    body_from_world[row][column]
                )

    source = [
        [float(covariance[row * 6 + column]) for column in range(6)]
        for row in range(6)
    ]
    intermediate = [
        [
            sum(transform[row][index] * source[index][column] for index in range(6))
            for column in range(6)
        ]
        for row in range(6)
    ]
    rotated = [
        [
            sum(intermediate[row][index] * transform[column][index] for index in range(6))
            for column in range(6)
        ]
        for row in range(6)
    ]
    return [rotated[row][column] for row in range(6) for column in range(6)]


def odometry_with_body_twist(
    message: Odometry,
    output_child_frame_id: str = "base_link",
) -> Odometry:
    """
    复制 Odometry，并把 twist 从世界坐标旋转到车体坐标.

    pose、pose covariance、header 和时间戳原样保留。线速度与角速度都作为三维
    向量执行逆旋转；在平面运动中 ``angular.z`` 在绕 Z 轴旋转前后保持不变。
    """
    output = deepcopy(message)
    world_from_body = world_from_body_rotation(message.pose.pose.orientation)
    body_from_world = transpose(world_from_body)

    linear = rotate_xyz(message.twist.twist.linear, body_from_world)
    angular = rotate_xyz(message.twist.twist.angular, body_from_world)
    output.twist.twist.linear.x = linear[0]
    output.twist.twist.linear.y = linear[1]
    output.twist.twist.linear.z = linear[2]
    output.twist.twist.angular.x = angular[0]
    output.twist.twist.angular.y = angular[1]
    output.twist.twist.angular.z = angular[2]
    output.twist.covariance = rotate_twist_covariance(
        message.twist.covariance,
        body_from_world,
    )

    if output_child_frame_id:
        output.child_frame_id = output_child_frame_id
    return output
