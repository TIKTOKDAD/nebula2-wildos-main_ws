#pragma once

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <map>
#include <sstream>
#include <iomanip>
#include <graaflib/graph.h>

#include <std_msgs/msg/color_rgba.hpp>
#include "graphnav_msgs/msg/navigation_graph.hpp"
#include "graphnav_msgs/msg/uuid.hpp"
#include <grid_map_msgs/msg/grid_map.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace graphnav_planner
{

// Utility function to convert UUID to string
inline std::string uuid_to_string(const graphnav_msgs::msg::UUID& uuid)
{
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i)
  {
    if (i == 4 || i == 6 || i == 8 || i == 10)
    {
      oss << '-';
    }
    oss << std::setw(2) << static_cast<int>(uuid.id[i]);
  }
  return oss.str();
}

inline std_msgs::msg::ColorRGBA colormapJet(double value) 
{
    std_msgs::msg::ColorRGBA color;
    color.a = 1.0;

    double r = std::min(1.0, std::max(0.0, 1.5 - std::fabs(4.0 * value - 3.0)));
    double g = std::min(1.0, std::max(0.0, 1.5 - std::fabs(4.0 * value - 2.0)));
    double b = std::min(1.0, std::max(0.0, 1.5 - std::fabs(4.0 * value - 1.0)));

    color.r = r;
    color.g = g;
    color.b = b;
    return color;
}

class UnexploredSpaceMap
{
public:
  UnexploredSpaceMap(double min_x, double max_x, double min_y, double max_y, double margin, double resolution)
    : resolution_(resolution)
  {
    origin_x_ = min_x - margin;
    origin_y_ = min_y - margin;
    size_x_ = static_cast<int>(std::ceil((max_x - min_x + 2 * margin) / resolution_));
    size_y_ = static_cast<int>(std::ceil((max_y - min_y + 2 * margin) / resolution_));
    map_ = Eigen::MatrixXi::Ones(size_x_, size_y_);  // initialize as unexplored
  }

  void mark_explored(double x, double y, double radius)
  {
    int ix = static_cast<int>((x - origin_x_) / resolution_);
    int iy = static_cast<int>((y - origin_y_) / resolution_);
    int ir = static_cast<int>(std::ceil(radius / resolution_));
    for (int dx = -ir; dx <= ir; ++dx)
    {
      for (int dy = -ir; dy <= ir; ++dy)
      {
        if (std::hypot(dx * resolution_, dy * resolution_) <= radius)
        {
          int nx = ix + dx;
          int ny = iy + dy;
          if (nx >= 0 && nx < map_.rows() && ny >= 0 && ny < map_.cols())
          {
            map_(nx, ny) = 0;  // mark as explored
          }
        }
      }
    }
  }

  void compute_distance_from(double x, double y)
  {
    int ix = static_cast<int>((x - origin_x_) / resolution_);
    int iy = static_cast<int>((y - origin_y_) / resolution_);
    Eigen::Vector2i goal(ix, iy);
    dist_map_ = Eigen::MatrixXf::Constant(size_x_, size_y_, std::numeric_limits<float>::infinity());
    using Cell = std::pair<int, int>;

    std::priority_queue<std::pair<float, Cell>, std::vector<std::pair<float, Cell>>, std::greater<>> pq;

    if (in_bounds(ix, iy))
    {
      dist_map_(ix, iy) = 0.0f;
      pq.emplace(0.0f, Cell(ix, iy));
    }
    else
    {
      // iterate around border and add to queue
      for (int x = 0; x < static_cast<int>(size_x_); ++x)
      {
        Eigen::Vector2i p(x, 0);
        p.y() = (goal.y() < 0) ? 0 : size_y_ - 1;
        dist_map_(p.x(), p.y()) = (p - goal).norm() * resolution_;
        pq.emplace(dist_map_(p.x(), p.y()), Cell(p.x(), p.y()));
      }
      for (int y = 0; y < static_cast<int>(size_y_); ++y)
      {
        Eigen::Vector2i p(0, y);
        p.x() = (goal.x() < 0) ? 0 : size_x_ - 1;
        dist_map_(p.x(), p.y()) = (p - goal).norm() * resolution_;
        pq.emplace(dist_map_(p.x(), p.y()), Cell(p.x(), p.y()));
      }
    }

    const int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    const int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const float cost[8] = { 1, std::sqrt(2), 1, std::sqrt(2), 1, std::sqrt(2), 1, std::sqrt(2) };

    while (!pq.empty())
    {
      auto [cur_dist, cell] = pq.top();
      pq.pop();
      int x = cell.first, y = cell.second;
      if (cur_dist > dist_map_(x, y))
        continue;
      for (int dir = 0; dir < 8; ++dir)
      {
        int nx = x + dx[dir];
        int ny = y + dy[dir];
        if (!in_bounds(nx, ny))
          continue;
        if (map_(nx, ny) == 0)
          continue;  // skip explored
        float new_dist = dist_map_(x, y) + cost[dir] * resolution_;
        if (new_dist < dist_map_(nx, ny))
        {
          dist_map_(nx, ny) = new_dist;
          pq.emplace(new_dist, Cell(nx, ny));
        }
      }
    }
  }

  double query_distance_to(double x, double y, int radius = 1)
  {
    int ix = static_cast<int>((x - origin_x_) / resolution_);
    int iy = static_cast<int>((y - origin_y_) / resolution_);
    float min_dist = std::numeric_limits<float>::infinity();
    for (int dx = -radius; dx <= radius; ++dx)
    {
      for (int dy = -radius; dy <= radius; ++dy)
      {
        int nx = ix + dx;
        int ny = iy + dy;
        if (in_bounds(nx, ny))
        {
          min_dist = std::min(min_dist, dist_map_(nx, ny));
        }
      }
    }
    return min_dist;
  }

  grid_map_msgs::msg::GridMap get_gridmap()
  {
    grid_map_msgs::msg::GridMap map_msg;
    map_msg.info.resolution = resolution_;
    map_msg.info.pose.position.x = origin_x_ + (size_x_ * resolution_) / 2.0;
    map_msg.info.pose.position.y = origin_y_ + (size_y_ * resolution_) / 2.0;
    map_msg.info.length_x = size_x_ * resolution_;
    map_msg.info.length_y = size_y_ * resolution_;
    map_msg.layers.push_back("unexplored");
    map_msg.layers.push_back("distance");
    map_msg.data.resize(2);
    map_msg.data[0].layout.data_offset = 0;
    map_msg.data[0].layout.dim.resize(2);
    map_msg.data[0].layout.dim[0].label = "column_index";
    map_msg.data[0].layout.dim[0].size = size_y_;
    map_msg.data[0].layout.dim[0].stride = size_y_ * size_x_;
    map_msg.data[0].layout.dim[1].label = "row_index";
    map_msg.data[0].layout.dim[1].size = size_x_;
    map_msg.data[0].layout.dim[1].stride = size_x_;
    map_msg.data[1].layout = map_msg.data[0].layout;
    map_msg.data[0].data.resize(size_x_ * size_y_);
    map_msg.data[1].data.resize(size_x_ * size_y_);
    for (size_t y = 0; y < size_y_; ++y)
    {
      for (size_t x = 0; x < size_x_; ++x)
      {
        int idx = (size_y_ - 1 - y) * size_x_ + (size_x_ - 1 - x);
        map_msg.data[0].data[idx] = map_(x, y);
        map_msg.data[1].data[idx] = dist_map_(x, y);
      }
    }
    return map_msg;
  }

private:
  bool in_bounds(int x, int y)
  {
    return x >= 0 && x < static_cast<int>(size_x_) && y >= 0 && y < static_cast<int>(size_y_);
  }

  size_t size_x_;
  size_t size_y_;
  double origin_x_;
  double origin_y_;
  double resolution_;
  Eigen::MatrixXi map_;  // 1 = unexplored, 0 = explored
  Eigen::MatrixXf dist_map_;
};

class Planner
{
public:
  using Polygon = std::vector<Eigen::Vector3f>;

  Planner(rclcpp::Logger logger);
  void set_trav_class(std::string trav_class);

  // void set_keep_in_polygons(std::vector<Polygon> &keep_in_polygons);
  // void set_keep_out_polygons(std::vector<Polygon> &keep_out_polygons);
  // void set_dynamic_obstacles(std::vector<Polygon> &obstacles);
  void update_graph(graphnav_msgs::msg::NavigationGraph::ConstSharedPtr graph);
  std::vector<Eigen::Vector3d> plan_to_goal(Eigen::Vector3d& goal, double goal_radius, rclcpp::Time current_time);
  // std::vector<Eigen::Vector3d> plan_to_goal(Polygon &goal_area);

  grid_map_msgs::msg::GridMap get_unexplored_debug_map()
  {
    grid_map_msgs::msg::GridMap map_msg;
    if (!unexplored_space_map_)
    {
      return map_msg;
    }
    return unexplored_space_map_->get_gridmap();
  }

private:
  UnexploredSpaceMap compute_unexplored_space_map();

  rclcpp::Logger logger_;
  std::string trav_class_;

  graaf::undirected_graph<graphnav_msgs::msg::Node, double> graph_;
  graaf::vertex_id_t current_node_idx_;
  size_t trav_class_idx_;
  std::optional<UnexploredSpaceMap> unexplored_space_map_;
  std::optional<Eigen::Vector3d> latest_frontier_;
  std::optional<rclcpp::Time> latest_frontier_time_;

  std::unordered_map<graaf::vertex_id_t, std::pair<graphnav_msgs::msg::Node, std::pair<double, double>>> frontier_scores_;

public:
  double frontier_dist_cost_factor_ = 2.0;
  double goal_dist_cost_factor_ = 1.0;
  double frontier_score_factor_ = 10.0;
  double min_local_frontier_score_ = 0.4;
  double local_frontier_radius_ = 7.0;
  double path_smoothness_period_ = 10.0; // seconds

  visualization_msgs::msg::MarkerArray get_score_visualization(const rclcpp::Time& stamp, std::string frame_id, bool with_id_text = false) const;

};

}  // namespace graphnav_planner
