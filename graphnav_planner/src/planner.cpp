// GraphNav 规划算法实现。
//
// 核心思路：
//   1. 把 NavigationGraph 转换为 graaf 无向加权图；
//   2. 根据各节点的 explored_radius 构建二维“未知空间”栅格；
//   3. 临时加入一个虚拟目标节点，把它连接到目标附近节点和所有前沿节点；
//   4. 使用 Dijkstra 从机器人当前图节点搜索到虚拟目标；
//   5. 删除临时节点，并把图节点路径转换成三维路点序列。
//
// 前沿连接代价用于估计“从已知图边走到某个前沿后，还要穿过多少未知空间才能
// 接近目标”；frontier_scores 则进一步对不同朝向的前沿进行偏好加权。
#include <rclcpp/rclcpp.hpp>
#include <graaflib/graph.h>
#include <graaflib/edge.h>
#include <graaflib/algorithm/shortest_path/dijkstra_shortest_path.h>
#include <graaflib/algorithm/shortest_path/dijkstra_shortest_paths.h>
#include <unordered_map>
#include <map>
#include <iterator>
#include <algorithm>
#include <cmath>

#include "graphnav_planner/planner.hpp"

namespace graphnav_planner
{

Planner::Planner(rclcpp::Logger logger) : logger_(logger)
{
}

// 设置后续解析 NavigationGraph 时使用的通行性类别名称。
// 真正的类别索引要等 update_graph() 收到图内 trav_classes 列表后才能确定。
void Planner::set_trav_class(std::string trav_class)
{
  trav_class_ = trav_class;
}

// 使用最新 NavigationGraph 完整重建内部图结构。
// 顶点 ID 直接采用 nodes 数组下标，保证 edge.from_idx/to_idx 和
// current_node_idx 可以原样用于 graaf 图。
void Planner::update_graph(graphnav_msgs::msg::NavigationGraph::ConstSharedPtr graph)
{
  // 每次更新都丢弃旧 graaf 图，避免动态图中已删除的节点或边残留。
  graph_ = graaf::undirected_graph<graphnav_msgs::msg::Node, double>();
  for (size_t i = 0; i < graph->nodes.size(); i++)
  {
    graphnav_msgs::msg::Node node = graph->nodes[i];
    graph_.add_vertex(node, i);
  }

  // 在消息声明的通行性类别中查找配置名称。找不到时仅保留刚加入的孤立顶点，
  // 不添加边，也不重建未知空间地图；调用方会在日志中看到警告。
  auto trav_class_it = std::find(graph->trav_classes.begin(), graph->trav_classes.end(), trav_class_);
  if (trav_class_it == graph->trav_classes.end())
  {
    RCLCPP_WARN(logger_, "Traversability class %s not found in graph", trav_class_.c_str());
    trav_class_idx_ = 0;
    return;
  }
  trav_class_idx_ = std::distance(graph->trav_classes.begin(), trav_class_it);

  // NavigationGraph 的一条边可以为多个通行性类别保存不同代价。本规划器只读取
  // 当前类别的 traversability_cost；缺少该类别数据的边直接忽略。
  for (auto edge : graph->edges)
  {
    if (trav_class_idx_ < edge.traversability.size())
    {
      double weight = edge.traversability[trav_class_idx_].traversability_cost;
      graph_.add_edge(edge.from_idx, edge.to_idx, weight);
    }
  }

  // current_node_idx 是图构建器认为机器人当前所在的节点，也是 Dijkstra 起点。
  current_node_idx_ = graph->current_node_idx;

  // 图节点或 explored_radius 改变后，未知/已探索区域也必须同步重建。
  unexplored_space_map_ = compute_unexplored_space_map();
}

// 根据图节点覆盖范围建立二维未知空间地图。
// 地图值 1 表示尚未探索，0 表示已探索；该地图不是障碍栅格，而是专门用于估算
// 前沿到最终目标之间需要穿过的未知空间距离。
UnexploredSpaceMap Planner::compute_unexplored_space_map()
{
  // 先计算所有图节点在 XY 平面的包围盒。调用方应保证导航图至少含一个节点。
  double min_x = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  for (auto& [id, node] : graph_.get_vertices())
  {
    const auto& pos = node.pose.position;
    min_x = std::min(min_x, pos.x);
    max_x = std::max(max_x, pos.x);
    min_y = std::min(min_y, pos.y);
    max_y = std::max(max_y, pos.y);
  }

  // 当前实现固定使用 1 m 分辨率，并在图包围盒四周增加 10 m 余量，以容纳
  // 位于图边缘之外的前沿和目标距离传播。
  double resolution = 1.0;
  double margin = 10.0;
  UnexploredSpaceMap unexplored_map(min_x, max_x, min_y, max_y, margin, resolution);

  // 每个节点以自身位置为圆心，将当前通行性类别给出的 explored_radius 圆盘
  // 标为已探索。多个圆盘自然合并成整张已探索走廊。
  for (auto& [id, node] : graph_.get_vertices())
  {
    if (trav_class_idx_ < node.trav_properties.size())
    {
      double explored_radius = node.trav_properties[trav_class_idx_].explored_radius;
      if (explored_radius > 0)
      {
        const auto& pos = node.pose.position;
        unexplored_map.mark_explored(pos.x, pos.y, explored_radius);
      }
    }
  }
  return unexplored_map;
}

// 从 current_node_idx_ 规划到最终目标。
// goal 和图节点必须已处于同一坐标系；current_time 只用于前沿选择滞回计时。
std::vector<Eigen::Vector3d> Planner::plan_to_goal(Eigen::Vector3d& goal, double goal_radius, rclcpp::Time current_time)
{
  // 尚无有效导航图/未知空间地图时无法规划。
  if (!unexplored_space_map_)
  {
    return {};
  }

  // 从最终目标位置在“未知单元”中做八邻域距离传播。之后查询任一前沿点，
  // 即可得到该前沿沿未知区域接近目标的估计距离。
  unexplored_space_map_->compute_distance_from(goal.x(), goal.y());

  // 最终目标通常不恰好是已有图节点，因此本轮搜索会在临时 search_graph 中加入
  // 一个虚拟顶点。基础 graph_ 保持原始 NavigationGraph 边权，便于调试显示。
  graphnav_msgs::msg::Node virtual_goal_node;
  virtual_goal_node.pose.position.x = goal.x();
  virtual_goal_node.pose.position.y = goal.y();
  virtual_goal_node.pose.position.z = goal.z();

  // 保存本轮每个前沿的 {原始节点, {方向分数, 最终连接代价}}，既供搜索连接
  // 使用，也供 RViz 调试可视化使用。
  frontier_scores_.clear();
  frontier_cost_debug_.clear();
  last_path_cost_breakdown_ = PathCostBreakdown{};

  // local_scored_frontiers 用于路径滞回：在一段时间内优先保留上次所选前沿
  // 附近的候选，减少导航图高频更新时左右分支反复切换。
  std::vector<graaf::vertex_id_t> local_scored_frontiers;
  // 目标半径内节点也会连接到虚拟目标。先缓存下来，等基础图最短路距离算完后
  // 再加边，避免虚拟目标把多个候选节点互相短接，污染候选排行里的 graph_path_cost。
  std::vector<std::tuple<graaf::vertex_id_t, double, double>> goal_radius_edges;
  // 一旦扫描到缺少 frontier_scores 的前沿，后续候选使用无分数回退代价。
  bool is_scored_graph = true;

  // 遍历所有基础图顶点，计算两类通往虚拟目标的连接：
  //   A. 前沿 -> 虚拟目标：代价由未知空间距离和可选方向分数决定；
  //   B. 目标半径内节点 -> 虚拟目标：代价由节点到目标直线距离决定。
  for (const auto& [id, node] : graph_.get_vertices())
  {
    Eigen::Vector3d node_pos;
    node_pos.x() = node.pose.position.x;
    node_pos.y() = node.pose.position.y;
    node_pos.z() = node.pose.position.z;

    if (trav_class_idx_ < node.trav_properties.size() && node.trav_properties[trav_class_idx_].is_frontier)
    {
      // max() 表示当前尚未找到可查询的前沿点；若 frontier_points 为空或距离场
      // 不可达，该值会保持极大值，从而使候选几乎不可能被 Dijkstra 选中。
      double frontier_path_distance = std::numeric_limits<double>::max();
      double frontier_score = -1.0;
      double frontier_cost = std::numeric_limits<double>::max();

      // 一个前沿节点可能包含多个边界采样点，选其中未知空间距离最短者代表该节点。
      for (const auto& frontier_pt : node.trav_properties[trav_class_idx_].frontier_points)
      {
        double d = unexplored_space_map_->query_distance_to(frontier_pt.x, frontier_pt.y);
        frontier_path_distance = std::min(frontier_path_distance, d);
      }

      // 计算“前沿节点指向最终目标”的三维单位方向；方向分箱时实际只使用 XY 航向。
      Eigen::Vector3d heading = (goal - node_pos).normalized();
      bool has_frontier_scores = false;
      double cur_frontier_dist_cost_factor = std::numeric_limits<double>::max();
      for (const auto& kv: node.properties)
      {
        if (kv.key == "frontier_scores")
        {
          // frontier_scores 将 360° 等分为 num_bins 个方向分箱。选择与目标航向
          // 最接近的分箱，并读取该方向从此前沿继续通行的预测分数。
          has_frontier_scores = true;
          int num_bins = kv.value.size();
          double angle_per_bin = 2 * M_PI / num_bins;
          double heading_angle = std::atan2(heading.y(), heading.x());
          if (heading_angle < 0)
            heading_angle += 2 * M_PI;
          int best_bin = static_cast<int>(std::round(heading_angle / angle_per_bin)) % num_bins;
          frontier_score = kv.value[best_bin];

          // 有分数时的前沿代价：
          //   cost = unknown_distance * (1 - frontier_score_factor * ln(score))
          // 对常见的 0 < score <= 1，分数越低，-ln(score) 越大，惩罚越强。
          cur_frontier_dist_cost_factor = 1.0 - frontier_score_factor_ * std::log(frontier_score);

          // 若已有上一轮选中的前沿，则收集其半径内且分数严格高于阈值的候选，
          // 供后面的路径保持时间窗使用。
          if (latest_frontier_){
            double distance_to_latest_frontier = (node_pos - *latest_frontier_).norm();
            if (distance_to_latest_frontier < local_frontier_radius_ && frontier_score > min_local_frontier_score_){
              local_scored_frontiers.push_back(id);
            }
          }
          
          // 保留的诊断代码：需要排查分箱或无穷代价时可临时启用。
          // if (frontier_score>0 && frontier_path_distance == std::numeric_limits<double>::max()){
          //   RCLCPP_INFO(logger_, "Node %s frontier score in best bin %d is %f, setting frontier_dist_cost_factor to %f",
          //              uuid_to_string(node.uuid).c_str(), best_bin, frontier_score, cur_frontier_dist_cost_factor);
          // }
        }
      }

      // 图或当前前沿没有有效方向分数时，退回固定距离系数：
      //   cost = unknown_distance * frontier_dist_cost_factor_
      if (!is_scored_graph || !has_frontier_scores)
      {
        is_scored_graph = false;
        frontier_cost = frontier_path_distance * frontier_dist_cost_factor_;
        cur_frontier_dist_cost_factor = frontier_dist_cost_factor_;
        // RCLCPP_WARN(logger_, "Node %ld is a frontier but has no frontier_scores property", id);
      
        // 无分数模式下，局部候选只检查与上一前沿的三维距离，不应用最低分阈值。
        if (latest_frontier_){
            double distance_to_latest_frontier = (node_pos - *latest_frontier_).norm();
            if (distance_to_latest_frontier < local_frontier_radius_){
              local_scored_frontiers.push_back(id);
            }
          }
      }
      else
      {
        frontier_cost = frontier_path_distance * cur_frontier_dist_cost_factor;
      }

      // 记录结果，稍后按“局部保持”或“全局搜索”策略选择哪些前沿连接虚拟目标。
      frontier_scores_[id] = std::make_pair(node, std::make_pair(frontier_score, frontier_cost));
      FrontierCostDebug cost_debug;
      cost_debug.frontier_path_distance = frontier_path_distance;
      cost_debug.frontier_score = frontier_score;
      cost_debug.cost_multiplier = cur_frontier_dist_cost_factor;
      cost_debug.frontier_cost = frontier_cost;
      cost_debug.used_frontier_scores = is_scored_graph && has_frontier_scores;
      frontier_cost_debug_[id] = cost_debug;
    }

    // 位于目标半径内的普通图节点可以直接连接虚拟目标：
    //   goal_cost = goal_dist_cost_factor_ * 三维直线距离
    // 使用严格小于，恰好等于 goal_radius 的节点不会建立连接。
    double node_goal_dist = (node_pos - goal).norm();
    if (node_goal_dist < goal_radius)
    {
      double goal_cost = goal_dist_cost_factor_ * node_goal_dist;
      goal_radius_edges.emplace_back(id, goal_cost, node_goal_dist);
    }
  }

  // 候选排行需要 current_node 到各真实节点的图路径代价。必须在任何虚拟目标边
  // 加入之前计算，否则无向图中的 virtual_goal 会让候选节点之间出现虚假捷径。
  const auto base_shortest_paths = graaf::algorithm::dijkstra_shortest_paths(graph_, current_node_idx_);

  auto search_graph = graph_;
  for (const auto& [edge_id, edge] : search_graph.get_edges())
  {
    search_graph.get_edge(edge_id) = graaf::get_weight(edge) * graph_path_cost_factor_;
  }
  graaf::vertex_id_t virtual_goal = search_graph.add_vertex(virtual_goal_node);
  std::unordered_map<graaf::vertex_id_t, double> virtual_edge_costs;
  std::unordered_map<graaf::vertex_id_t, bool> virtual_edge_is_frontier;
  std::unordered_map<graaf::vertex_id_t, double> virtual_goal_distances;

  auto record_candidate_cost = [this, &base_shortest_paths](graaf::vertex_id_t id,
                                                            double virtual_edge_cost,
                                                            bool is_frontier,
                                                            double goal_node_distance = std::numeric_limits<double>::quiet_NaN()) {
    auto path_it = base_shortest_paths.find(id);
    if (path_it == base_shortest_paths.end())
    {
      return;
    }
    CandidateCostDebug candidate;
    candidate.node = id;
    candidate.is_frontier = is_frontier;
    candidate.graph_path_cost = path_it->second.total_weight;
    candidate.virtual_edge_cost = virtual_edge_cost;
    candidate.total_cost =
        candidate.graph_path_cost * graph_path_cost_factor_ + candidate.virtual_edge_cost * virtual_edge_cost_factor_;
    candidate.goal_node_distance = goal_node_distance;
    if (is_frontier)
    {
      auto debug_it = frontier_cost_debug_.find(id);
      if (debug_it != frontier_cost_debug_.end())
      {
        candidate.frontier = debug_it->second;
      }
    }
    last_path_cost_breakdown_.candidate_costs.push_back(candidate);
  };

  auto add_virtual_edge = [&](graaf::vertex_id_t id,
                              double virtual_edge_cost,
                              bool is_frontier,
                              double goal_node_distance = std::numeric_limits<double>::quiet_NaN()) {
    if (search_graph.has_edge(virtual_goal, id))
    {
      return false;
    }
    search_graph.add_edge(virtual_goal, id, virtual_edge_cost * virtual_edge_cost_factor_);
    virtual_edge_costs[id] = virtual_edge_cost;
    virtual_edge_is_frontier[id] = is_frontier;
    virtual_goal_distances[id] = goal_node_distance;
    record_candidate_cost(id, virtual_edge_cost, is_frontier, goal_node_distance);
    return true;
  };

  for (const auto& [id, goal_cost, node_goal_dist] : goal_radius_edges)
  {
    add_virtual_edge(id, goal_cost, false, node_goal_dist);
  }
  

  // ------------------------- 前沿选择滞回 -------------------------
  // 若仍处于 path_smoothness_period_ 时间窗，且上次前沿附近存在候选，则只连接
  // 这些局部前沿。这样 Dijkstra 仍能在局部候选间优化，但不会突然跳到远处分支。
  bool use_local_frontiers = false;
  if (!local_scored_frontiers.empty())
  {
    if ((current_time - *latest_frontier_time_).seconds() < path_smoothness_period_)
    {
      for (auto id: local_scored_frontiers)
      {
        double frontier_cost = frontier_scores_[id].second.second;
        add_virtual_edge(id, frontier_cost, true);
        use_local_frontiers = true;
      }
    }
    else
    {
      latest_frontier_time_ = current_time;
    }
  }  
  
  // 不满足局部保持条件时，把本轮所有前沿连接到虚拟目标，执行一次全局选择，
  // 并从当前时刻重新开始保持计时。
  if (!use_local_frontiers)
  {
    for (auto& [id, score_pair] : frontier_scores_)
    {
      double frontier_cost = score_pair.second.second;
      add_virtual_edge(id, frontier_cost, true);
    }
    latest_frontier_time_ = current_time;
  }

  // 在“已有通行边 + 目标附近连接 + 前沿未知空间连接”组成的临时无向图上，
  // 从机器人当前节点到虚拟目标执行 Dijkstra 最短路径搜索。
  auto path = graaf::algorithm::dijkstra_shortest_path(search_graph, current_node_idx_, virtual_goal);
  std::vector<Eigen::Vector3d> path_points;
  // 记录倒数第二个顶点是否为前沿。倒数第一个通常是刚加入的虚拟目标。
  bool has_frontier_in_path = false;
  if (path)
  {
    // Dijkstra 返回的 total_weight 是“图路径边权 * 系数 + 最后一条虚拟连接边 * 系数”
    // 的加权总和。这里再逐段取原始边权，拆出调参前的两段代价，供 RViz 显示。
    last_path_cost_breakdown_.has_path = true;
    last_path_cost_breakdown_.start_node = current_node_idx_;
    last_path_cost_breakdown_.total_cost = path->total_weight;
    last_path_cost_breakdown_.reason = use_local_frontiers ? "local_frontiers" : "global_frontiers";
    if (path->vertices.size() >= 2)
    {
      auto prev_it = path->vertices.begin();
      for (auto it = std::next(prev_it); it != path->vertices.end(); ++it)
      {
        const auto from = *prev_it;
        const auto to = *it;
        const bool is_virtual_edge = (from == virtual_goal || to == virtual_goal);
        if (is_virtual_edge)
        {
          const auto selected_node = (from == virtual_goal) ? to : from;
          const auto virtual_cost_it = virtual_edge_costs.find(selected_node);
          const double virtual_edge_cost =
              virtual_cost_it != virtual_edge_costs.end() ?
                  virtual_cost_it->second :
                  graaf::get_weight(search_graph.get_edge(from, to));
          last_path_cost_breakdown_.selected_node = selected_node;
          last_path_cost_breakdown_.virtual_edge_cost = virtual_edge_cost;
          const auto& selected = graph_.get_vertex(selected_node);
          last_path_cost_breakdown_.marker_position = Eigen::Vector3d(
              selected.pose.position.x, selected.pose.position.y, selected.pose.position.z);
          const auto is_frontier_it = virtual_edge_is_frontier.find(selected_node);
          const bool selected_is_frontier_edge =
              is_frontier_it != virtual_edge_is_frontier.end() && is_frontier_it->second;
          auto debug_it = frontier_cost_debug_.find(selected_node);
          last_path_cost_breakdown_.ends_at_frontier =
              selected_is_frontier_edge && debug_it != frontier_cost_debug_.end();
          last_path_cost_breakdown_.ends_at_goal_radius_node = !last_path_cost_breakdown_.ends_at_frontier;
          if (last_path_cost_breakdown_.ends_at_frontier)
          {
            last_path_cost_breakdown_.frontier = debug_it->second;
          }
          else
          {
            const auto dist_it = virtual_goal_distances.find(selected_node);
            last_path_cost_breakdown_.goal_node_distance =
                dist_it != virtual_goal_distances.end() ?
                    dist_it->second :
                    (last_path_cost_breakdown_.marker_position - goal).norm();
          }
        }
        else
        {
          const double edge_weight = graaf::get_weight(graph_.get_edge(from, to));
          last_path_cost_breakdown_.graph_path_cost += edge_weight;
          last_path_cost_breakdown_.graph_edges.emplace_back(from, to, edge_weight);
        }
        prev_it = it;
      }
    }
    for (auto& candidate : last_path_cost_breakdown_.candidate_costs)
    {
      candidate.selected = candidate.node == last_path_cost_breakdown_.selected_node;
    }
    std::sort(last_path_cost_breakdown_.candidate_costs.begin(), last_path_cost_breakdown_.candidate_costs.end(),
              [](const CandidateCostDebug& lhs, const CandidateCostDebug& rhs) {
                return lhs.total_cost < rhs.total_cost;
              });

    // 把 graaf 返回的顶点序列转换为算法对外暴露的三维点序列。
    size_t idx = 0;
    for (const auto& node_id : path->vertices)
    {
      const auto& node = search_graph.get_vertex(node_id);
      const auto& pos = node.pose.position;
      path_points.push_back(Eigen::Vector3d(pos.x, pos.y, pos.z));

      // 若虚拟目标前的最后一个真实节点是前沿，路径在进入未知区域时还应经过
      // 实际前沿边界，而不只停在图节点中心，因此额外插入前沿采样点的质心。
      if (idx == path->vertices.size() - 2 && trav_class_idx_ < node.trav_properties.size() &&
          node.trav_properties[trav_class_idx_].is_frontier)
      {
        if (!node.trav_properties[trav_class_idx_].frontier_points.empty())
        {
          // 使用所有 frontier_points 的三维算术平均值作为单一边界路点。
          Eigen::Vector3d mean_frontier(0.0, 0.0, 0.0);
          double n_frontier_points = node.trav_properties[trav_class_idx_].frontier_points.size();
          for (const auto& frontier_point : node.trav_properties[trav_class_idx_].frontier_points)
          {
            mean_frontier += Eigen::Vector3d(frontier_point.x, frontier_point.y, frontier_point.z);
          }
          mean_frontier /= n_frontier_points;
          path_points.push_back(mean_frontier);
        }

        // 保存本轮实际选中的前沿中心，供下一轮局部滞回和调试圆环使用。
        latest_frontier_ = Eigen::Vector3d(node.pose.position.x, node.pose.position.y, node.pose.position.z);
        has_frontier_in_path = true;
      }
      idx++;
    }
  }
  if (!has_frontier_in_path)
  {
    // 直接通过目标半径内节点抵达目标、或搜索失败时，路径中可能没有前沿。
    // 此时清除空间锚点，下一轮不能继续进行“上次前沿附近”的局部保持。
    latest_frontier_.reset();
    RCLCPP_WARN(logger_, "NO FRONTIER IN PATH!!!!!");
  }

  // search_graph 是本轮局部副本，函数返回后虚拟目标会随副本一起销毁。
  return path_points;
}

std::string Planner::format_path_cost_breakdown() const
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2);

  if (!last_path_cost_breakdown_.has_path)
  {
    ss << "Path cost breakdown\n";
    ss << "No valid Dijkstra path";
    if (!last_path_cost_breakdown_.reason.empty())
    {
      ss << "\nreason = " << last_path_cost_breakdown_.reason;
    }
    return ss.str();
  }

  ss << "Path cost breakdown\n";
  ss << "selection_mode = " << last_path_cost_breakdown_.reason << "\n";
  ss << "start_node = " << last_path_cost_breakdown_.start_node
     << ", selected_node = " << last_path_cost_breakdown_.selected_node << "\n";
  ss << "total_cost = graph_path_cost(" << last_path_cost_breakdown_.graph_path_cost
     << ") * graph_path_cost_factor(" << graph_path_cost_factor_
     << ") + virtual_edge_cost(" << last_path_cost_breakdown_.virtual_edge_cost
     << ") * virtual_edge_cost_factor(" << virtual_edge_cost_factor_
     << ") = " << last_path_cost_breakdown_.total_cost << "\n";

  ss << "graph_path_cost = sum(" << last_path_cost_breakdown_.graph_edges.size()
     << " NavigationGraph edges)";
  if (!last_path_cost_breakdown_.graph_edges.empty())
  {
    ss << "\n  ";
    constexpr size_t max_edges_to_show = 8;
    size_t shown = 0;
    for (const auto& [from, to, weight] : last_path_cost_breakdown_.graph_edges)
    {
      if (shown > 0)
      {
        ss << " + ";
      }
      if (shown >= max_edges_to_show)
      {
        ss << "...";
        break;
      }
      ss << "w(" << from << "->" << to << ")=" << weight;
      shown++;
    }
  }
  ss << "\n";

  if (last_path_cost_breakdown_.ends_at_frontier)
  {
    const auto& frontier = last_path_cost_breakdown_.frontier;
    ss << "virtual_edge_cost = frontier_cost\n";
    if (frontier.used_frontier_scores)
    {
      ss << "frontier_score = frontier_scores_selected_bin = " << frontier.frontier_score << "\n";
      ss << "frontier_cost = frontier_path_distance(" << frontier.frontier_path_distance
         << ") * (1 - frontier_score_factor(" << frontier_score_factor_
         << ") * ln(frontier_score(" << frontier.frontier_score << ")))\n";
      ss << "frontier_cost_multiplier = " << frontier.cost_multiplier
         << ", frontier_cost = " << frontier.frontier_cost << "\n";
    }
    else
    {
      ss << "frontier_cost = frontier_path_distance(" << frontier.frontier_path_distance
         << ") * frontier_dist_cost_factor(" << frontier_dist_cost_factor_ << ")\n";
      ss << "frontier_cost = " << frontier.frontier_cost << "\n";
    }
    ss << "weighted_virtual_edge_cost = " << last_path_cost_breakdown_.virtual_edge_cost
       << " * virtual_edge_cost_factor(" << virtual_edge_cost_factor_
       << ") = " << last_path_cost_breakdown_.virtual_edge_cost * virtual_edge_cost_factor_ << "\n";
  }
  else if (last_path_cost_breakdown_.ends_at_goal_radius_node)
  {
    ss << "virtual_edge_cost = goal_cost\n";
    ss << "goal_cost = goal_dist_cost_factor(" << goal_dist_cost_factor_
       << ") * node_goal_dist(" << last_path_cost_breakdown_.goal_node_distance
       << ") = " << last_path_cost_breakdown_.virtual_edge_cost << "\n";
    ss << "weighted_virtual_edge_cost = " << last_path_cost_breakdown_.virtual_edge_cost
       << " * virtual_edge_cost_factor(" << virtual_edge_cost_factor_
       << ") = " << last_path_cost_breakdown_.virtual_edge_cost * virtual_edge_cost_factor_ << "\n";
  }
  else
  {
    ss << "virtual_edge_cost source = unknown";
  }

  return ss.str();
}

std::string Planner::format_candidate_cost_ranking() const
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2);
  ss << "Candidate cost ranking\n";
  ss << "selection_mode = " << last_path_cost_breakdown_.reason << "\n";

  if (last_path_cost_breakdown_.candidate_costs.empty())
  {
    ss << "No reachable candidates connected to virtual_goal";
    return ss.str();
  }

  ss << "selected_node = " << last_path_cost_breakdown_.selected_node
     << ", selected_total = " << last_path_cost_breakdown_.total_cost << "\n";
  constexpr size_t max_candidates_to_show = 10;
  const size_t count = std::min(max_candidates_to_show, last_path_cost_breakdown_.candidate_costs.size());
  for (size_t i = 0; i < count; ++i)
  {
    const auto& candidate = last_path_cost_breakdown_.candidate_costs[i];
    ss << (candidate.selected ? "* " : "  ") << (i + 1) << ". node " << candidate.node
       << " total=" << candidate.total_cost
       << " graph=" << candidate.graph_path_cost << "*" << graph_path_cost_factor_;
    if (candidate.is_frontier)
    {
      ss << " frontier=" << candidate.virtual_edge_cost << "*" << virtual_edge_cost_factor_;
      if (candidate.frontier.used_frontier_scores)
      {
        ss << " score=" << candidate.frontier.frontier_score;
      }
      else
      {
        ss << " score=N/A";
      }
      ss << " unknown_dist=" << candidate.frontier.frontier_path_distance;
    }
    else
    {
      ss << " goal=" << candidate.virtual_edge_cost << "*" << virtual_edge_cost_factor_
         << " node_goal_dist=" << candidate.goal_node_distance;
    }
    ss << "\n";
  }

  if (last_path_cost_breakdown_.candidate_costs.size() > count)
  {
    ss << "... " << (last_path_cost_breakdown_.candidate_costs.size() - count)
       << " more candidates";
  }
  return ss.str();
}

visualization_msgs::msg::MarkerArray Planner::get_path_cost_breakdown_visualization(const rclcpp::Time& stamp,
                                                                                    std::string frame_id) const
{
  visualization_msgs::msg::MarkerArray markers;

  if (!last_path_cost_breakdown_.has_path)
  {
    for (const auto& ns : {std::string("path_cost_breakdown"), std::string("candidate_cost_ranking")})
    {
      visualization_msgs::msg::Marker delete_marker;
      delete_marker.action = visualization_msgs::msg::Marker::DELETE;
      delete_marker.header.frame_id = frame_id;
      delete_marker.header.stamp = stamp;
      delete_marker.ns = ns;
      delete_marker.id = 1;
      markers.markers.push_back(delete_marker);
    }
    return markers;
  }

  double min_x = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  double max_z = std::numeric_limits<double>::lowest();
  for (const auto& [id, node] : graph_.get_vertices())
  {
    const auto& pos = node.pose.position;
    min_x = std::min(min_x, static_cast<double>(pos.x));
    max_x = std::max(max_x, static_cast<double>(pos.x));
    min_y = std::min(min_y, static_cast<double>(pos.y));
    max_y = std::max(max_y, static_cast<double>(pos.y));
    max_z = std::max(max_z, static_cast<double>(pos.z));
  }
  if (graph_.get_vertices().empty())
  {
    min_x = last_path_cost_breakdown_.marker_position.x();
    max_x = last_path_cost_breakdown_.marker_position.x();
    min_y = last_path_cost_breakdown_.marker_position.y();
    max_y = last_path_cost_breakdown_.marker_position.y();
    max_z = last_path_cost_breakdown_.marker_position.z();
  }

  const double graph_width = std::max(1.0, max_x - min_x);
  const double graph_height = std::max(1.0, max_y - min_y);
  const double x_offset = std::max(8.0, graph_width * 0.45);
  const double y_offset = std::max(8.0, graph_height * 0.45);
  const double z_offset = std::max(4.0, std::max(graph_width, graph_height) * 0.10);
  const double text_z = max_z + z_offset;

  visualization_msgs::msg::Marker breakdown_marker;
  breakdown_marker.header.frame_id = frame_id;
  breakdown_marker.header.stamp = stamp;
  breakdown_marker.ns = "path_cost_breakdown";
  breakdown_marker.id = 1;
  breakdown_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  breakdown_marker.action = visualization_msgs::msg::Marker::ADD;
  breakdown_marker.pose.position.x = min_x - x_offset;
  breakdown_marker.pose.position.y = max_y + y_offset;
  breakdown_marker.pose.position.z = text_z;
  breakdown_marker.scale.z = path_cost_breakdown_text_size_;
  breakdown_marker.color.a = 1.0;
  breakdown_marker.color.r = 1.0;
  breakdown_marker.color.g = 0.9;
  breakdown_marker.color.b = 0.1;
  breakdown_marker.text = format_path_cost_breakdown();
  markers.markers.push_back(breakdown_marker);

  visualization_msgs::msg::Marker ranking_marker;
  ranking_marker.header.frame_id = frame_id;
  ranking_marker.header.stamp = stamp;
  ranking_marker.ns = "candidate_cost_ranking";
  ranking_marker.id = 1;
  ranking_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  ranking_marker.action = visualization_msgs::msg::Marker::ADD;
  ranking_marker.pose.position.x = max_x + x_offset;
  ranking_marker.pose.position.y = max_y + y_offset;
  ranking_marker.pose.position.z = text_z;
  ranking_marker.scale.z = candidate_cost_ranking_text_size_;
  ranking_marker.color.a = 1.0;
  ranking_marker.color.r = 0.3;
  ranking_marker.color.g = 1.0;
  ranking_marker.color.b = 1.0;
  ranking_marker.text = format_candidate_cost_ranking();
  markers.markers.push_back(ranking_marker);
  return markers;
}

// 生成前沿选择的 RViz MarkerArray：
//   - 清理上一帧旧 Marker；
//   - 用绿色圆盘显示 local_frontier_radius_；
//   - 显示路径保持时间窗剩余时间；
//   - 用 Jet 色图显示每个前沿的代价，并可选显示“分数/代价”文本。
visualization_msgs::msg::MarkerArray Planner::get_score_visualization(const rclcpp::Time& stamp,
                                                                      std::string frame_id,
                                                                      bool with_id_text) const
{
  visualization_msgs::msg::MarkerArray markers;

  // 先发 DELETEALL，避免候选前沿数量减少后，旧立方体或文字继续残留在 RViz。
  visualization_msgs::msg::Marker delete_all_infcost;
  delete_all_infcost.action = visualization_msgs::msg::Marker::DELETEALL;
  delete_all_infcost.header.frame_id = frame_id;
  delete_all_infcost.header.stamp = stamp;
  delete_all_infcost.ns = "frontier_inf_cost";
  delete_all_infcost.id = 0;
  markers.markers.push_back(delete_all_infcost);

  visualization_msgs::msg::Marker delete_all_idtext;
  delete_all_idtext.action = visualization_msgs::msg::Marker::DELETEALL;
  delete_all_idtext.header.frame_id = frame_id;
  delete_all_idtext.header.stamp = stamp;
  delete_all_idtext.ns = "frontier_id_text";
  delete_all_idtext.id = 0;
  markers.markers.push_back(delete_all_idtext);

  // 若存在上次选中的前沿，以它为中心画一个直径为 2*local_frontier_radius_ 的
  // 半透明绿色薄圆柱，表示下一轮局部候选的空间范围。
  if (latest_frontier_) {
    visualization_msgs::msg::Marker frontier_marker;
    frontier_marker.header.frame_id = frame_id;
    frontier_marker.header.stamp = stamp;
    frontier_marker.ns = "local_frontier";
    frontier_marker.id = 0;
    frontier_marker.type = visualization_msgs::msg::Marker::CYLINDER;
    frontier_marker.action = visualization_msgs::msg::Marker::ADD;
    frontier_marker.pose.position.x = latest_frontier_->x();
    frontier_marker.pose.position.y = latest_frontier_->y();
    frontier_marker.pose.position.z = latest_frontier_->z();
    frontier_marker.scale.x = local_frontier_radius_ * 2;
    frontier_marker.scale.y = local_frontier_radius_ * 2;
    frontier_marker.scale.z = 0.1;
    frontier_marker.color.a = 0.5;
    frontier_marker.color.r = 0.0;
    frontier_marker.color.g = 1.0;
    frontier_marker.color.b = 0.0;
    markers.markers.push_back(frontier_marker);
  }
  else{
    // 没有有效前沿锚点时，请求删除上一帧的局部范围圆环。
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    delete_marker.header.frame_id = frame_id;
    delete_marker.header.stamp = stamp;
    delete_marker.ns = "local_frontier";
    delete_marker.id = 0;
    markers.markers.push_back(delete_marker);
  }

  // 显示从最近一次全局前沿选择开始经过的时间，以及保持时间窗的理论剩余值。
  // 文本位置放在局部圆环的右上方并抬高 1 m，减少与图节点重叠。
  if (latest_frontier_time_)
  {
    double time_diff = (stamp - *latest_frontier_time_).seconds();
    visualization_msgs::msg::Marker text_marker;
    text_marker.header.frame_id = frame_id;
    text_marker.header.stamp = stamp;
    text_marker.ns = "local_frontier_time";
    text_marker.id = 0;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose.position.x = latest_frontier_->x() + local_frontier_radius_;
    text_marker.pose.position.y = latest_frontier_->y() + local_frontier_radius_;
    text_marker.pose.position.z = latest_frontier_->z() + 1.0;  // 将文字抬到节点上方
    text_marker.scale.z = 1.5;           // TEXT_VIEW_FACING 只使用 scale.z 表示字高
    text_marker.color.a = 1.0;
    text_marker.color.r = 0.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    std::ostringstream ss;
    ss << "Exploitation\nTimeout: " << path_smoothness_period_ - time_diff;
    text_marker.text = ss.str();
    markers.markers.push_back(text_marker);
  }

  // Marker id 从 1 开始，0 留给上面的清理/范围类 Marker。
  int id_counter = 1;

  // 仅用于可视化的固定显示范围：将 [30, 400] 线性归一化到 [0, 1]，
  // 超出范围的值会饱和。该范围不会反过来影响 Dijkstra 的实际代价。
  double min_val = 30.0;
  double max_val = 400.0;

  for (const auto& [id, score_pair] : frontier_scores_)
  {
    const auto& node = score_pair.first;
    double frontier_score = score_pair.second.first;
    double frontier_cost = score_pair.second.second;
    // 当前实现无条件显示所有前沿。下面保留的旧条件表明这里曾只打算显示
    // “代价无穷但分数为正”的异常候选。
    // if (frontier_cost == std::numeric_limits<double>::infinity() && frontier_score > 0)
    if (true)
    {
      double norm_cost = (frontier_cost - min_val) / (max_val - min_val);
      norm_cost = std::min(1.0, std::max(0.0, norm_cost));
      std_msgs::msg::ColorRGBA color = colormapJet(norm_cost);
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = frame_id;
      marker.header.stamp = stamp;
      marker.ns = "frontier_inf_cost";
      marker.id = id_counter;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = node.pose;
      marker.scale.x = 0.5;
      marker.scale.y = 0.5;
      marker.scale.z = 0.5;
      // 旧的固定半透明红色方案保留在此；当前实际颜色使用 Jet 色图。
      // marker.color.a = 0.5;
      // marker.color.r = 1.0;
      // marker.color.g = 0.0;
      // marker.color.b = 0.0;
      marker.color = color;
      markers.markers.push_back(marker);
    }
    if (with_id_text)
    {
      // 在每个前沿节点上叠加白色文字，内容为“方向分数/连接代价”。
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.frame_id = frame_id;
      text_marker.header.stamp = stamp;
      text_marker.ns = "frontier_id_text";
      text_marker.id = id_counter;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose = node.pose;
      text_marker.pose.position.z += 0.1;  // 略微抬高，减少与立方体重叠
      text_marker.scale.z = 0.5;           // TEXT_VIEW_FACING 只使用 scale.z
      text_marker.color.a = 1.0;
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 1.0;
      std::ostringstream ss;
      // ss << "Score: " << frontier_score << " Cost: " << frontier_cost;
      // 固定保留两位小数，使不同 Marker 的显示宽度和精度较稳定。
      ss << std::fixed << std::setprecision(2);
      ss << frontier_score << "/" << frontier_cost;
      text_marker.text = ss.str();
      markers.markers.push_back(text_marker);
    }
    id_counter++;
  }
  return markers;
}


}  // namespace graphnav_planner
