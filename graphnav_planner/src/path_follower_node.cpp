// 路径前视目标生成节点：接收 PlannerNode 发布的完整 nav_msgs::msg::Path，
// 根据每帧里程计寻找机器人附近的路径点，并发布一个较近的 PoseStamped 目标。
// 本节点不输出速度、不判断最终任务是否完成，也不直接调用 Nav2 action。
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

// 数据流：
//   ~/path + ~/odom -> 坐标变换 -> 最近路径点 -> 前视点插值 -> ~/goal_pose
//
// 前视目标由里程计回调驱动，所以在里程计持续更新且路径有效时，会以接近里程计
// 频率发布。下游若使用 Nav2 action，通常需要额外的限频/去抖桥接节点。
class PathFollowerNode : public rclcpp::Node
{
public:
  PathFollowerNode(const rclcpp::NodeOptions& options)
    : Node("path_follower_node", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_, this)
  {
    // 机器人到目标前视点的期望三维直线距离，单位为米。这里不是路径累计弧长。
    this->declare_parameter("wp_lookahead_dist", 2.0);
    wp_lookahead_dist_ = this->get_parameter("wp_lookahead_dist").as_double();

    // 可选的路径折角保护：启用后，搜索前视点时不会跨过夹角过大的相邻路径段。
    this->declare_parameter("enable_lookahead_turn_limit", false);
    enable_lookahead_turn_limit_ = this->get_parameter("enable_lookahead_turn_limit").as_bool();
    this->declare_parameter("max_lookahead_turn_angle_deg", 120.0);

    // 将用户输入限制到 [0°, 180°]，并预先转成点积阈值，避免在每次里程计
    // 回调中反复计算 cos。单位向量点积小于 cos(阈值) 即表示转角超过阈值。
    const double max_lookahead_turn_angle_deg =
        std::clamp(this->get_parameter("max_lookahead_turn_angle_deg").as_double(), 0.0, 180.0);
    max_lookahead_turn_dot_ = std::cos(max_lookahead_turn_angle_deg * std::acos(-1.0) / 180.0);

    // 路径时间戳最多允许落后当前里程计时间戳的秒数；超时路径不再产生目标。
    this->declare_parameter("path_timeout", 1.0);
    path_timeout_ = this->get_parameter("path_timeout").as_double();

    // 私有话题 ~/path、~/goal_pose、~/odom 会包含节点名，并可在 launch 中 remap。
    // 所有通信对象使用深度为 10 的默认 QoS。
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        "~/path", 10, std::bind(&PathFollowerNode::path_callback, this, std::placeholders::_1));
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("~/goal_pose", 10);
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "~/odom", 10, std::bind(&PathFollowerNode::odom_callback, this, std::placeholders::_1));
  }

private:
  // 只保存最新路径。前视点并不在路径回调中计算，因为还需要最新机器人位置；
  // 真正的计算由下一帧 odom_callback() 触发。
  void path_callback(const nav_msgs::msg::Path::ConstSharedPtr msg)
  {
    path_ = msg;
  }

  // 每收到一帧里程计，尝试从当前有效路径生成一个局部前视目标。
  void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    // 路径尚未到达，或“里程计时间 - 路径时间”超过 path_timeout_ 时不发布。
    // 若两路消息存在时钟不一致，差值可能为负数，而负数不会被这里判为超时；
    // 因此 launch 中必须保证两个节点使用相同的系统时间或仿真时间。
    if (!path_ || (rclcpp::Time(msg->header.stamp) - rclcpp::Time(path_->header.stamp)).seconds() > path_timeout_)
    {
      return;
    }

    // 空路径代表规划失败或路径失效，不应继续发布上一目标。
    if (path_->poses.empty())
    {
      return;
    }

    // 只有一个点时没有可供插值的线段，直接原样转发该 PoseStamped。
    if (path_->poses.size() < 2)
    {
      pose_pub_->publish(path_->poses[0]);
      return;
    }

    // Odometry.pose.pose 表达在 msg->header.frame_id 下。先封装成 PoseStamped，
    // 再转换到 Path.header.frame_id，确保后面的距离计算位于同一坐标系。
    geometry_msgs::msg::PoseStamped odom_pose;
    odom_pose.header = msg->header;
    odom_pose.pose = msg->pose.pose;
    geometry_msgs::msg::PoseStamped odom_pose_in_path_frame;
    try
    {
      // 最多等待 0.1 秒获取 TF；失败时放弃本帧，避免混用不同坐标系的位置。
      odom_pose_in_path_frame = tf_buffer_.transform(odom_pose, path_->header.frame_id, tf2::durationFromSec(0.1));
    }
    catch (const tf2::TransformException& ex)
    {
      RCLCPP_WARN(this->get_logger(), "Could not transform odometry pose to path frame: %s", ex.what());
      return;
    }

    Eigen::Vector3d robot_pos(odom_pose_in_path_frame.pose.position.x, odom_pose_in_path_frame.pose.position.y,
                              odom_pose_in_path_frame.pose.position.z);

    // 缓存路径点以及“机器人到各离散路径点”的三维欧氏距离。
    // 注意这些距离不是相邻路径段的累计长度，也没有把机器人投影到路径线段上。
    std::vector<Eigen::Vector3d> path_points;
    std::vector<double> path_robot_distances;
    for (const auto& pose : path_->poses)
    {
      path_points.push_back(Eigen::Vector3d(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z));
      path_robot_distances.push_back((path_points.back() - robot_pos).norm());
    }

    // 以离散路径点为候选，选择距离机器人最近的点作为向前搜索的起点。
    size_t closest_index =
        std::min_element(path_robot_distances.begin(), path_robot_distances.end()) - path_robot_distances.begin();

    // 若末点已经是最近的离散路径点，说明机器人位于路径末端附近或已经越过其余
    // 点。直接发布末点，并用当前里程计时间戳更新 Header，避免被下游视为旧目标。
    if (closest_index == path_->poses.size() - 1)
    {
      auto last_pose = path_->poses.back();
      last_pose.header.frame_id = path_->header.frame_id;
      last_pose.header.stamp = msg->header.stamp;
      pose_pub_->publish(last_pose);
      return;
    }

    // 从最近点的后一个点开始向路径末端搜索，寻找第一个机器人距离达到前视阈值
    // 的路径点。循环条件保留至少一个有效的 [wp_index-1, wp_index] 插值区间。
    size_t wp_index = closest_index + 1;
    while (wp_index + 1 < path_->poses.size() && path_robot_distances[wp_index] < wp_lookahead_dist_)
    {
      if (enable_lookahead_turn_limit_)
      {
        // 比较进入 wp_index 和离开 wp_index 的两个单位方向。只有转角不超过
        // 配置阈值时才允许把前视点继续推进到下一段；阈值大于 90° 时才允许
        // 跨过普通直角弯，较小阈值会更保守地停在转角前。
        Eigen::Vector3d current_dir = (path_points[wp_index] - path_points[wp_index - 1]).normalized();
        Eigen::Vector3d next_dir = (path_points[wp_index + 1] - path_points[wp_index]).normalized();
        if (current_dir.dot(next_dir) < max_lookahead_turn_dot_)
        {
          break;
        }
      }
      wp_index++;
    }

    // 在相邻的两个候选点间做线性插值，使其端点距离近似跨过
    // wp_lookahead_dist_。这里线性插值的是“端点到机器人的距离”，并非求线段与
    // 以机器人为圆心的球面的精确交点，因此弯曲/长线段上是一个轻量近似。
    double a = path_robot_distances[wp_index - 1] - wp_lookahead_dist_;
    double b = path_robot_distances[wp_index] - wp_lookahead_dist_;
    // 解 a + wp_fraction * (b - a) = 0；再限制到线段参数 [0, 1]。
    // 急弯提前终止或路径末端不足前视距离时，clamp 会使目标落在某个端点。
    double wp_fraction = -a / (b - a);
    wp_fraction = std::clamp(wp_fraction, 0.0, 1.0);
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = path_->header.frame_id;
    // 输出使用本帧里程计时间，使下游拿到的是与当前机器人状态对应的局部目标。
    pose.header.stamp = msg->header.stamp;
    pose.pose.position.x = path_->poses[wp_index].pose.position.x * wp_fraction +
                           path_->poses[wp_index - 1].pose.position.x * (1 - wp_fraction);
    pose.pose.position.y = path_->poses[wp_index].pose.position.y * wp_fraction +
                           path_->poses[wp_index - 1].pose.position.y * (1 - wp_fraction);
    pose.pose.position.z = path_->poses[wp_index].pose.position.z * wp_fraction +
                           path_->poses[wp_index - 1].pose.position.z * (1 - wp_fraction);

    // 朝向不做四元数插值：目标非常接近后一端点时采用后一端点朝向，否则沿用
    // 前一端点朝向。这样可避免简单线性插值四元数造成非单位旋转。
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

  // ROS 通信对象。
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

  // TF Buffer 与节点使用同一时钟；仿真中需同时设置 use_sim_time。
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // 构造时读取的参数缓存。本节点未提供动态参数更新回调。
  double wp_lookahead_dist_ = 2.0;
  bool enable_lookahead_turn_limit_ = false;
  double max_lookahead_turn_dot_ = 0.5;
  double path_timeout_ = 1.0;

  // 最近一次收到的完整路径；shared_ptr 可避免在回调间复制整条 Path。
  nav_msgs::msg::Path::ConstSharedPtr path_;
};

}  // namespace graphnav_planner

// 注册为 rclcpp component；组件名为 graphnav_planner::PathFollowerNode。
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(graphnav_planner::PathFollowerNode)
