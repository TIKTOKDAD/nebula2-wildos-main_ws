#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace graphnav_planner
{

class PathFollowerNode : public rclcpp::Node
{
public:
  PathFollowerNode(const rclcpp::NodeOptions& options)
    : Node("path_follower_node", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_, this)
  {
    this->declare_parameter("wp_lookahead_dist", 2.0);
    wp_lookahead_dist_ = this->get_parameter("wp_lookahead_dist").as_double();
    this->declare_parameter("path_timeout", 1.0);
    path_timeout_ = this->get_parameter("path_timeout").as_double();
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "~/path", 10, std::bind(&PathFollowerNode::path_callback, this, std::placeholders::_1));
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("~/goal_pose", 10);
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "~/odom", 10, std::bind(&PathFollowerNode::odom_callback, this, std::placeholders::_1));
  }

private:
  void path_callback(const nav_msgs::msg::Path::ConstSharedPtr msg)
  {
    path_ = msg;
  }

  void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    if (!path_ || (rclcpp::Time(msg->header.stamp) - rclcpp::Time(path_->header.stamp)).seconds() > path_timeout_)
    {
      return;
    }
    if (path_->poses.empty())
    {
      return;
    }
    if (path_->poses.size() < 2)
    {
      pose_pub_->publish(path_->poses[0]);
      return;
    }
    geometry_msgs::msg::PoseStamped odom_pose;
    odom_pose.header = msg->header;
    odom_pose.pose = msg->pose.pose;
    geometry_msgs::msg::PoseStamped odom_pose_in_path_frame;
    try
    {
      odom_pose_in_path_frame = tf_buffer_.transform(odom_pose, path_->header.frame_id, tf2::durationFromSec(0.1));
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN(this->get_logger(), "Could not transform odometry pose to path frame: %s", ex.what());
      return;
    }
    Eigen::Vector3d robot_pos(odom_pose_in_path_frame.pose.position.x, odom_pose_in_path_frame.pose.position.y,
                              odom_pose_in_path_frame.pose.position.z);
    std::vector<Eigen::Vector3d> path_points;
    std::vector<double> path_robot_distances;
    for (const auto& pose : path_->poses)
    {
      path_points.push_back(Eigen::Vector3d(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z));
      path_robot_distances.push_back((path_points.back() - robot_pos).norm());
    }
    size_t closest_index =
        std::min_element(path_robot_distances.begin(), path_robot_distances.end()) - path_robot_distances.begin();
    // if last waypoint is closest, publish it
    if (closest_index == path_->poses.size() - 1)
    {
      auto last_pose = path_->poses.back();
      last_pose.header.frame_id = path_->header.frame_id;
      last_pose.header.stamp = msg->header.stamp;
      pose_pub_->publish(last_pose);
      return;
    }
    // find next waypoint past lookahead distance
    size_t wp_index = closest_index + 1;
    while (wp_index + 1 < path_->poses.size() && path_robot_distances[wp_index] < wp_lookahead_dist_)
    {
      // only increment as long as the direction is roughly the same
      Eigen::Vector3d current_dir = (path_points[wp_index] - path_points[wp_index - 1]).normalized();
      Eigen::Vector3d next_dir = (path_points[wp_index + 1] - path_points[wp_index]).normalized();
      if (current_dir.dot(next_dir) < 0.5)  // more than 60 degree turn
      {
        break;
      }
      wp_index++;
    }
    // interpolate between wp_index - 1 and wp_index  to find where distance equals wp_lookahead_dist_
    double a = path_robot_distances[wp_index - 1] - wp_lookahead_dist_;
    double b = path_robot_distances[wp_index] - wp_lookahead_dist_;
    // a + wp_fraction * (b - a) = 0
    double wp_fraction = -a / (b - a);
    wp_fraction = std::clamp(wp_fraction, 0.0, 1.0);
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = path_->header.frame_id;
    pose.header.stamp = msg->header.stamp;
    pose.pose.position.x = path_->poses[wp_index].pose.position.x * wp_fraction +
                           path_->poses[wp_index - 1].pose.position.x * (1 - wp_fraction);
    pose.pose.position.y = path_->poses[wp_index].pose.position.y * wp_fraction +
                           path_->poses[wp_index - 1].pose.position.y * (1 - wp_fraction);
    pose.pose.position.z = path_->poses[wp_index].pose.position.z * wp_fraction +
                           path_->poses[wp_index - 1].pose.position.z * (1 - wp_fraction);
    if (wp_fraction > 0.95)
    {
      pose.pose.orientation = path_->poses[wp_index].pose.orientation;
    }
    else
    {
      pose.pose.orientation = path_->poses[wp_index - 1].pose.orientation;
    }
    pose_pub_->publish(pose);
  }

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  double wp_lookahead_dist_ = 2.0;
  double path_timeout_ = 1.0;

  nav_msgs::msg::Path::ConstSharedPtr path_;
};

}  // namespace graphnav_planner

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(graphnav_planner::PathFollowerNode)
