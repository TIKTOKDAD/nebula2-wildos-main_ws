// ROS 2 节点封装：负责接收导航图、最终目标和里程计，调用 Planner 完成图规划，
// 再把算法层返回的三维点序列封装成 nav_msgs::msg::Path。
//
// 触发关系：
//   新导航图 ─┐
//              ├─> plan_to_goal() ─> 完整图路径及调试信息
//   新最终目标 ┘
//   新里程计 ─────> 只缓存，用于下一次规划时判断是否到达最终目标
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <graphnav_msgs/msg/navigation_graph.hpp>
#include <std_msgs/msg/header.hpp>
#include <optional>
#include "graphnav_planner/planner.hpp"

namespace graphnav_planner
{

// PlannerNode 只处理 ROS 通信、参数和坐标系转换；具体图搜索算法位于 Planner。
// 继承 rclcpp::Node 并在文件末注册为组件，因此既可由 component container 加载，
// 也可通过 CMake 生成的 planner_node 可执行程序独立启动。
class PlannerNode : public rclcpp::Node
{
public:
  PlannerNode(const rclcpp::NodeOptions& options)
    : Node("planner_node", options)
    , tf_buffer_(this->get_clock())
    , tf_listener_(tf_buffer_, this)
    , planner_(this->get_logger())
  {
    // ------------------------- 规划代价参数 -------------------------
    // 参数只在构造阶段读取一次；本节点没有注册动态参数回调，运行中修改参数不会
    // 自动同步到 planner_。各参数的具体代价公式参见 Planner::plan_to_goal()。
    this->declare_parameter("graph_path_cost_factor", 1.0);
    this->declare_parameter("virtual_edge_cost_factor", 1.0);
    this->declare_parameter("frontier_dist_cost_factor", 2.0);
    this->declare_parameter("goal_dist_cost_factor", 1.0);
    this->declare_parameter("frontier_score_factor", 10.0);
    this->declare_parameter("min_local_frontier_score", 0.4);
    this->declare_parameter("local_frontier_radius", 7.0);
    this->declare_parameter("path_smoothness_period", 10.0);
    this->declare_parameter("path_cost_breakdown_text_size", 0.8);
    this->declare_parameter("candidate_cost_ranking_text_size", 0.7);

    auto get_nonnegative_parameter = [this](const char* name) {
      double value = this->get_parameter(name).as_double();
      if (value < 0.0)
      {
        RCLCPP_WARN(this->get_logger(), "Parameter %s must be non-negative; using 0.0 instead of %f", name, value);
        return 0.0;
      }
      return value;
    };

    planner_.graph_path_cost_factor_ = get_nonnegative_parameter("graph_path_cost_factor");
    planner_.virtual_edge_cost_factor_ = get_nonnegative_parameter("virtual_edge_cost_factor");
    planner_.frontier_dist_cost_factor_ = this->get_parameter("frontier_dist_cost_factor").as_double();
    planner_.goal_dist_cost_factor_ = this->get_parameter("goal_dist_cost_factor").as_double();
    planner_.frontier_score_factor_ = this->get_parameter("frontier_score_factor").as_double();
    planner_.min_local_frontier_score_ = this->get_parameter("min_local_frontier_score").as_double();
    planner_.local_frontier_radius_ = this->get_parameter("local_frontier_radius").as_double();
    planner_.path_smoothness_period_ = this->get_parameter("path_smoothness_period").as_double();
    planner_.path_cost_breakdown_text_size_ = get_nonnegative_parameter("path_cost_breakdown_text_size");
    planner_.candidate_cost_ranking_text_size_ = get_nonnegative_parameter("candidate_cost_ranking_text_size");

    // 选择 NavigationGraph.trav_classes 中要使用的通行性类别；该类别决定边权、
    // explored_radius、is_frontier 和 frontier_points 等数据的索引。
    this->declare_parameter("trav_class", "default");
    planner_.set_trav_class(this->get_parameter("trav_class").as_string());

    // 图节点与最终目标建立连接、以及判定机器人到达目标时共用的三维距离阈值。
    this->declare_parameter("goal_radius", 3.0);
    goal_radius_ = this->get_parameter("goal_radius").as_double();

    // ------------------------- 输入订阅 -------------------------
    // "~/..." 是节点私有话题。以默认节点名启动时，~/nav_graph 会展开为
    // /graphnav_planner/nav_graph，再由 launch 文件 remap 到 scored_nav_graph。
    // 新图到达后重建内部 graaf 图，并立即尝试使用当前保存的目标重新规划。
    graph_sub_ = this->create_subscription<graphnav_msgs::msg::NavigationGraph>(
        "~/nav_graph", 10, [this](const graphnav_msgs::msg::NavigationGraph::ConstSharedPtr msg) {
          this->planner_.update_graph(msg);
          // 保存图的坐标系和时间戳，后续 Path 也沿用这个 Header。
          this->latest_graph_header_ = msg->header;
          this->plan_to_goal();
        });

    // 收到的是最终任务目标，而不是 path_follower 发布的局部前视目标。
    // 保存最新目标后立即规划；若此时尚未收到导航图，plan_to_goal() 会直接等待。
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "~/goal_pose", 10, [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
          this->goal_pose_ = msg;
          this->plan_to_goal();
        });

    // 里程计回调只更新缓存，不触发图规划。缓存会在下一次图/目标更新所触发的规划
    // 结束后用于到达判定，因此“刚进入 goal_radius”不会单独清除目标。
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "~/odom", 10, [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { this->odom_ = msg; });

    // ------------------------- 输出发布器 -------------------------
    // 主输出为完整图路径；其他输出仅用于观察未知空间距离场、前沿代价和已选
    // 路径总代价拆分。
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("~/path", 10);
    grid_map_debug_pub_ = this->create_publisher<grid_map_msgs::msg::GridMap>("~/unexplored_space_map", 10);
    scores_debug_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/frontier_scores", 10);
    path_cost_debug_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("~/path_cost_breakdown", 10);
  }

private:
  // 当“最终目标”和“导航图 Header”同时存在时完成一次规划与发布。
  // 任何 TF 转换失败都会放弃本轮，不会发布部分结果或复用旧路径。
  void plan_to_goal()
  {
    if (goal_pose_ && latest_graph_header_)
    {
      // 使用副本，避免改写订阅消息。将目标时间戳对齐到最新导航图时间戳，保证
      // TF 查询和图所描述的环境状态处于同一时刻；目标原始 frame_id 保持不变。
      geometry_msgs::msg::PoseStamped goal = *goal_pose_;
      goal.header.stamp = latest_graph_header_->stamp;
      geometry_msgs::msg::PoseStamped goal_in_graph_frame;
      try
      {
        // 最多等待 0.1 秒，把最终目标转换到导航图坐标系。
        goal_in_graph_frame = tf_buffer_.transform(goal, latest_graph_header_->frame_id, tf2::durationFromSec(0.1));
      }
      catch (const tf2::TransformException& ex)
      {
        RCLCPP_WARN(this->get_logger(), "Could not transform goal pose to graph frame: %s", ex.what());
        return;
      }

      // Planner 算法层只接收位置向量；目标朝向会在构造 Path 末点时恢复。
      Eigen::Vector3d goal_vec(goal_in_graph_frame.pose.position.x, goal_in_graph_frame.pose.position.y,
                               goal_in_graph_frame.pose.position.z);
      auto path = planner_.plan_to_goal(goal_vec, goal_radius_, this->get_clock()->now());

      // 即使算法没有找到路径，仍发布同一 Header、poses 为空的 Path，便于下游
      // 明确获知当前路径已经失效，而不是继续永久使用上一条路径。
      nav_msgs::msg::Path path_msg;
      path_msg.header = *latest_graph_header_;
      path_msg.poses.resize(path.size());
      for (size_t i = 0; i < path.size(); i++)
      {
        // Planner 返回的所有点已位于导航图坐标系。
        path_msg.poses[i].pose.position.x = path[i].x();
        path_msg.poses[i].pose.position.y = path[i].y();
        path_msg.poses[i].pose.position.z = path[i].z();
        if (i < path.size() - 1)
        {
          // 中间路点朝向下一路点：固定 Z 轴向上，用相邻点方向构造正交旋转矩阵。
          // 因而主要表达平面航向；最后一个点不使用该计算。
          Eigen::Vector3d d = (path[i + 1] - path[i]).normalized();
          Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
          m.col(1) = Eigen::Vector3d::UnitZ().cross(d).normalized();
          m.col(0) = m.col(1).cross(m.col(2)).normalized();
          Eigen::Quaterniond q(m);
          path_msg.poses[i].pose.orientation.x = q.x();
          path_msg.poses[i].pose.orientation.y = q.y();
          path_msg.poses[i].pose.orientation.z = q.z();
          path_msg.poses[i].pose.orientation.w = q.w();
        }
        else
        {
          // 末点保留调用方给出的任务目标朝向，使最终姿态约束不被路径切线覆盖。
          path_msg.poses[i].pose.orientation = goal_in_graph_frame.pose.orientation;
        }
      }
      path_pub_->publish(path_msg);

      // 调试消息构造可能遍历整张栅格或全部前沿，因此仅在确有订阅者时执行，
      // 避免正常无人订阅运行时支付额外的计算和序列化开销。
      if (grid_map_debug_pub_->get_subscription_count() > 0)
      {
        grid_map_msgs::msg::GridMap grid_map_msg = planner_.get_unexplored_debug_map();
        grid_map_msg.header = *latest_graph_header_;
        grid_map_debug_pub_->publish(grid_map_msg);
      }
      if (scores_debug_pub_->get_subscription_count() > 0)
      {
        visualization_msgs::msg::MarkerArray marker_array = planner_.get_score_visualization(
          this->get_clock()->now(), latest_graph_header_->frame_id, true);
        scores_debug_pub_->publish(marker_array);
      }
      if (path_cost_debug_pub_->get_subscription_count() > 0)
      {
        visualization_msgs::msg::MarkerArray marker_array = planner_.get_path_cost_breakdown_visualization(
          this->get_clock()->now(), latest_graph_header_->frame_id);
        path_cost_debug_pub_->publish(marker_array);
      }

      // ------------------------- 最终目标到达判定 -------------------------
      // 只有已经收到里程计时才检查。将同一个目标副本转换到里程计父坐标系，
      // 再与 Odometry.pose.pose（也表达在 header.frame_id 中）进行三维距离比较。
      if (odom_)
      {
        try
        {
          geometry_msgs::msg::PoseStamped goal_in_odom_frame;
          goal_in_odom_frame = tf_buffer_.transform(goal, odom_->header.frame_id, tf2::durationFromSec(0.1));
          Eigen::Vector3d goal_vec(goal_in_odom_frame.pose.position.x, goal_in_odom_frame.pose.position.y,
                                   goal_in_odom_frame.pose.position.z);
          Eigen::Vector3d odom_vec(odom_->pose.pose.position.x, odom_->pose.pose.position.y,
                                   odom_->pose.pose.position.z);
          if ((goal_vec - odom_vec).norm() < goal_radius_)
          {
            // 清除内部最终目标，之后只有收到新的目标才会继续规划。
            // 本轮已经生成的 Path 不会因 reset 而撤回。
            goal_pose_.reset();
          }
        }
        catch (const tf2::TransformException& ex)
        {
          RCLCPP_WARN(this->get_logger(), "Could not transform goal pose to odom frame: %s", ex.what());
        }
      }
    }
  }

  // ROS 通信对象。订阅与发布均采用深度为 10 的默认 QoS。
  rclcpp::Subscription<graphnav_msgs::msg::NavigationGraph>::SharedPtr graph_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scores_debug_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr path_cost_debug_pub_;

  // TF Buffer 使用节点时钟；use_sim_time=true 时即使用 /clock。
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // 最近一次输入缓存。optional Header 用来区分“尚未收到图”和合法的空 Header。
  geometry_msgs::msg::PoseStamped::ConstSharedPtr goal_pose_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  std::optional<std_msgs::msg::Header> latest_graph_header_;
  double goal_radius_;

  Planner planner_;
};

}  // namespace graphnav_planner

// 注册为 rclcpp component；组件名为 graphnav_planner::PlannerNode。
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(graphnav_planner::PlannerNode)
