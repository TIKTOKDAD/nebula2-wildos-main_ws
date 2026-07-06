import math

from dlio_odom_twist_adapter.twist_transform import odometry_with_body_twist
from nav_msgs.msg import Odometry


def make_odom(yaw: float, vx: float, vy: float) -> Odometry:
    message = Odometry()
    message.header.frame_id = "odom"
    message.child_frame_id = "base_link"
    message.pose.pose.position.x = 12.0
    message.pose.pose.position.y = -3.0
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    message.twist.twist.linear.x = vx
    message.twist.twist.linear.y = vy
    return message


def test_identity_orientation_keeps_twist():
    source = make_odom(0.0, 1.2, -0.3)
    output = odometry_with_body_twist(source)

    assert math.isclose(output.twist.twist.linear.x, 1.2, abs_tol=1e-9)
    assert math.isclose(output.twist.twist.linear.y, -0.3, abs_tol=1e-9)


def test_ninety_degree_yaw_moves_world_y_into_body_x():
    source = make_odom(math.pi / 2.0, 0.0, 2.0)
    output = odometry_with_body_twist(source)

    assert math.isclose(output.twist.twist.linear.x, 2.0, abs_tol=1e-9)
    assert math.isclose(output.twist.twist.linear.y, 0.0, abs_tol=1e-9)


def test_observed_dlio_sample_becomes_forward_body_speed():
    source = make_odom(math.radians(-101.0), -0.0349, -0.1668)
    output = odometry_with_body_twist(source)

    assert output.twist.twist.linear.x > 0.16
    assert abs(output.twist.twist.linear.y) < 0.01


def test_covariance_axes_rotate_with_twist():
    source = make_odom(math.pi / 2.0, 0.0, 1.0)
    source.twist.covariance[0] = 1.0
    source.twist.covariance[7] = 4.0
    output = odometry_with_body_twist(source)

    assert math.isclose(output.twist.covariance[0], 4.0, abs_tol=1e-9)
    assert math.isclose(output.twist.covariance[7], 1.0, abs_tol=1e-9)


def test_pose_header_and_source_message_are_not_modified():
    source = make_odom(math.pi / 2.0, 0.0, 1.0)
    output = odometry_with_body_twist(source, "robot_base")

    assert output.header.frame_id == "odom"
    assert output.pose.pose.position.x == 12.0
    assert output.pose.pose.position.y == -3.0
    assert output.child_frame_id == "robot_base"
    assert source.child_frame_id == "base_link"
    assert source.twist.twist.linear.x == 0.0
    assert source.twist.twist.linear.y == 1.0


def test_zero_norm_orientation_is_rejected():
    source = make_odom(0.0, 1.0, 0.0)
    source.pose.pose.orientation.w = 0.0

    try:
        odometry_with_body_twist(source)
    except ValueError as exc:
        assert "zero-norm quaternion" in str(exc)
    else:
        raise AssertionError("Expected an invalid quaternion to be rejected")
