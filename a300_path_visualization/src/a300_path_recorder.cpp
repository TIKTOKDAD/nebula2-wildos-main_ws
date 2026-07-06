#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

/**
 * @brief 把 A300 的里程计位姿累计为一条实际行驶路径。
 *
 * 数据流非常简单：
 *
 *   /dlio/odom_node/odom (nav_msgs/Odometry)
 *                     -> 本节点
 *                     -> /a300_0000/driven_path (nav_msgs/Path)
 *
 * 这里记录的是车辆“已经走过”的实际轨迹，不是 graphnav/Nav2 规划出来的
 * 未来路径。输出使用 ROS 标准 nav_msgs/Path，因此 RViz 可以直接显示。
 */
class A300PathRecorder : public rclcpp::Node
{
public:
  A300PathRecorder()
  : Node("a300_path_recorder")
  {
    // 输入里程计话题。当前工程使用 DLIO 输出作为 A300 的连续位姿来源。
    odom_topic_ = this->declare_parameter<std::string>(
      "odom_topic", "/dlio/odom_node/odom");

    // 输出标准 Path 话题；RViz 的 Path Display 直接订阅此话题即可。
    path_topic_ = this->declare_parameter<std::string>(
      "path_topic", "/a300_0000/driven_path");

    // 车辆在水平面内至少移动该距离后才增加新点。
    // 这样可过滤静止时的里程计抖动，并控制 Path 消息的大小。
    min_sample_distance_ = this->declare_parameter<double>(
      "min_sample_distance", 0.05);

    // 最多保留多少个路径点。达到上限后丢弃最老的点，防止长时间运行时
    // 内存和 Path 消息无限增长；设为 0 表示不限制，不推荐长期运行时使用。
    max_path_points_ = this->declare_parameter<int64_t>(
      "max_path_points", 20000);

    // Path 发布频率独立于高频里程计。轨迹仍按距离采样，但不会在每条
    // Odometry 到达时都重复发送整条历史路径，从而减少 DDS 带宽占用。
    publish_rate_ = this->declare_parameter<double>("publish_rate", 10.0);

    // 对参数做基本保护，避免负距离、负容量或无效频率导致异常行为。
    if (min_sample_distance_ < 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "min_sample_distance 不能为负数，已从 %.3f 修正为 0.0",
        min_sample_distance_);
      min_sample_distance_ = 0.0;
    }
    if (max_path_points_ < 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "max_path_points 不能为负数，已从 %ld 修正为 0（不限制）",
        static_cast<long>(max_path_points_));
      max_path_points_ = 0;
    }
    if (!std::isfinite(publish_rate_) || publish_rate_ <= 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "publish_rate 必须大于 0，已从 %.3f 修正为 10.0 Hz",
        publish_rate_);
      publish_rate_ = 10.0;
    }

    // Path 使用 transient_local（类似 ROS 1 latched topic）：
    // 即使 RViz 晚于本节点启动，也能立即收到最近一次发布的完整轨迹。
    auto path_qos = rclcpp::QoS(rclcpp::KeepLast(1));
    path_qos.reliable();
    path_qos.transient_local();
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(path_topic_, path_qos);

    // 里程计属于高频传感器数据，SensorDataQoS 能兼容常见的 best-effort
    // 发布端，避免因 QoS 不匹配而收不到 DLIO 数据。
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&A300PathRecorder::odom_callback, this, std::placeholders::_1));

    // 使用墙钟定时发布。Gazebo 暂停时不会增加新点；由于 dirty 标志为
    // false，定时器也不会反复发送完全相同的大消息。
    const auto publish_period = std::chrono::duration<double>(1.0 / publish_rate_);
    publish_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(publish_period),
      std::bind(&A300PathRecorder::publish_path, this));

    RCLCPP_INFO(
      this->get_logger(),
      "A300 实际轨迹记录已启动：%s -> %s，采样距离 %.3f m，最多 %ld 点，发布 %.1f Hz",
      odom_topic_.c_str(), path_topic_.c_str(), min_sample_distance_,
      static_cast<long>(max_path_points_), publish_rate_);
  }

private:
  /**
   * @brief 接收一条里程计，并在车辆移动足够距离后保存为轨迹点。
   */
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;

    // NaN/Inf 坐标无法在 RViz 中正常显示，也会污染后续距离计算。
    if (!std::isfinite(x) || !std::isfinite(y)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "收到非有限里程计坐标，已跳过该轨迹点");
      return;
    }

    std::lock_guard<std::mutex> lock(path_mutex_);

    // 一条 Path 不能混合多个坐标系。如果上游 frame_id 在运行中变化，
    // 清空旧点并从新坐标系重新记录，避免画出一条错误的跨坐标系连线。
    if (!path_frame_id_.empty() && msg->header.frame_id != path_frame_id_) {
      RCLCPP_WARN(
        this->get_logger(),
        "里程计坐标系从 '%s' 变为 '%s'，已清空旧轨迹",
        path_frame_id_.c_str(), msg->header.frame_id.c_str());
      path_points_.clear();
      has_last_point_ = false;
    }
    path_frame_id_ = msg->header.frame_id;

    // 地面车辆只按 XY 平面距离采样。路面轻微起伏不会导致无意义的密集点。
    if (has_last_point_) {
      const double distance = std::hypot(x - last_x_, y - last_y_);
      if (distance < min_sample_distance_) {
        return;
      }
    }

    geometry_msgs::msg::PoseStamped path_point;
    path_point.header = msg->header;
    path_point.pose = msg->pose.pose;
    path_points_.push_back(std::move(path_point));

    last_x_ = x;
    last_y_ = y;
    has_last_point_ = true;

    // deque 从头删除是 O(1)，适合实现固定长度的滑动轨迹窗口。
    if (max_path_points_ > 0) {
      while (path_points_.size() > static_cast<std::size_t>(max_path_points_)) {
        path_points_.pop_front();
      }
    }

    latest_stamp_ = msg->header.stamp;
    path_dirty_ = true;
  }

  /**
   * @brief 按配置频率发布最新完整轨迹。
   */
  void publish_path()
  {
    nav_msgs::msg::Path path_msg;

    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      if (!path_dirty_ || path_points_.empty()) {
        return;
      }

      path_msg.header.frame_id = path_frame_id_;
      path_msg.header.stamp = latest_stamp_;
      path_msg.poses.assign(path_points_.begin(), path_points_.end());
      path_dirty_ = false;
    }

    // 在锁外发布，避免 DDS 发送较大 Path 消息时阻塞里程计回调。
    path_pub_->publish(path_msg);
  }

  std::string odom_topic_;
  std::string path_topic_;
  double min_sample_distance_{0.05};
  int64_t max_path_points_{20000};
  double publish_rate_{10.0};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  std::mutex path_mutex_;
  std::deque<geometry_msgs::msg::PoseStamped> path_points_;
  std::string path_frame_id_;
  builtin_interfaces::msg::Time latest_stamp_;
  bool has_last_point_{false};
  bool path_dirty_{false};
  double last_x_{0.0};
  double last_y_{0.0};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<A300PathRecorder>());
  rclcpp::shutdown();
  return 0;
}
