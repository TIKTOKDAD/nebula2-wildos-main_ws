#include <rclcpp/rclcpp.hpp>
#include <graaflib/graph.h>
#include <graaflib/algorithm/shortest_path/dijkstra_shortest_path.h>
#include <unordered_map>
#include <map>

#include "graphnav_planner/planner.hpp"

namespace graphnav_planner
{

Planner::Planner(rclcpp::Logger logger) : logger_(logger)
{
}

void Planner::set_trav_class(std::string trav_class)
{
  trav_class_ = trav_class;
}

void Planner::update_graph(graphnav_msgs::msg::NavigationGraph::ConstSharedPtr graph)
{
  graph_ = graaf::undirected_graph<graphnav_msgs::msg::Node, double>();
  for (size_t i = 0; i < graph->nodes.size(); i++)
  {
    graphnav_msgs::msg::Node node = graph->nodes[i];
    graph_.add_vertex(node, i);
  }
  auto trav_class_it = std::find(graph->trav_classes.begin(), graph->trav_classes.end(), trav_class_);
  if (trav_class_it == graph->trav_classes.end())
  {
    RCLCPP_WARN(logger_, "Traversability class %s not found in graph", trav_class_.c_str());
    trav_class_idx_ = 0;
    return;
  }
  trav_class_idx_ = std::distance(graph->trav_classes.begin(), trav_class_it);
  for (auto edge : graph->edges)
  {
    if (trav_class_idx_ < edge.traversability.size())
    {
      double weight = edge.traversability[trav_class_idx_].traversability_cost;
      graph_.add_edge(edge.from_idx, edge.to_idx, weight);
    }
  }
  current_node_idx_ = graph->current_node_idx;
  unexplored_space_map_ = compute_unexplored_space_map();
}

UnexploredSpaceMap Planner::compute_unexplored_space_map()
{
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
  double resolution = 1.0;
  double margin = 10.0;
  UnexploredSpaceMap unexplored_map(min_x, max_x, min_y, max_y, margin, resolution);
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

std::vector<Eigen::Vector3d> Planner::plan_to_goal(Eigen::Vector3d& goal, double goal_radius, rclcpp::Time current_time)
{
  if (!unexplored_space_map_)
  {
    return {};
  }
  unexplored_space_map_->compute_distance_from(goal.x(), goal.y());
  graphnav_msgs::msg::Node virtual_goal_node;
  virtual_goal_node.pose.position.x = goal.x();
  virtual_goal_node.pose.position.y = goal.y();
  virtual_goal_node.pose.position.z = goal.z();
  graaf::vertex_id_t virtual_goal = graph_.add_vertex(virtual_goal_node);

  frontier_scores_.clear();

  std::vector<graaf::vertex_id_t> local_scored_frontiers;
  bool is_scored_graph = true;

  for (const auto& [id, node] : graph_.get_vertices())
  {
    Eigen::Vector3d node_pos;
    node_pos.x() = node.pose.position.x;
    node_pos.y() = node.pose.position.y;
    node_pos.z() = node.pose.position.z;

    if (trav_class_idx_ < node.trav_properties.size() && node.trav_properties[trav_class_idx_].is_frontier)
    {
      double frontier_path_distance = std::numeric_limits<double>::max();
      double frontier_score = -1.0;
      double frontier_cost = std::numeric_limits<double>::max();

      for (const auto& frontier_pt : node.trav_properties[trav_class_idx_].frontier_points)
      {
        double d = unexplored_space_map_->query_distance_to(frontier_pt.x, frontier_pt.y);
        frontier_path_distance = std::min(frontier_path_distance, d);
      }

      // compute heading to the goal
      Eigen::Vector3d heading = (goal - node_pos).normalized();
      bool has_frontier_scores = false;
      double cur_frontier_dist_cost_factor = std::numeric_limits<double>::max();
      for (const auto& kv: node.properties)
      {
        if (kv.key == "frontier_scores")
        {
          has_frontier_scores = true;
          int num_bins = kv.value.size();
          double angle_per_bin = 2 * M_PI / num_bins;
          double heading_angle = std::atan2(heading.y(), heading.x());
          if (heading_angle < 0)
            heading_angle += 2 * M_PI;
          int best_bin = static_cast<int>(std::round(heading_angle / angle_per_bin)) % num_bins;
          frontier_score = kv.value[best_bin];
          cur_frontier_dist_cost_factor = 1.0 - frontier_score_factor_ * std::log(frontier_score);

          if (latest_frontier_){
            double distance_to_latest_frontier = (node_pos - *latest_frontier_).norm();
            if (distance_to_latest_frontier < local_frontier_radius_ && frontier_score > min_local_frontier_score_){
              local_scored_frontiers.push_back(id);
            }
          }
          
          // if (frontier_score>0 && frontier_path_distance == std::numeric_limits<double>::max()){
          //   RCLCPP_INFO(logger_, "Node %s frontier score in best bin %d is %f, setting frontier_dist_cost_factor to %f",
          //              uuid_to_string(node.uuid).c_str(), best_bin, frontier_score, cur_frontier_dist_cost_factor);
          // }
        }
      }
      if (!is_scored_graph || !has_frontier_scores)
      {
        is_scored_graph = false;
        frontier_cost = frontier_path_distance * frontier_dist_cost_factor_;
        // RCLCPP_WARN(logger_, "Node %ld is a frontier but has no frontier_scores property", id);
      
        // add to local frontier based on just distance
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
      frontier_scores_[id] = std::make_pair(node, std::make_pair(frontier_score, frontier_cost));
    }

    // connect nodes to goal if within goal_radius
    double node_goal_dist = (node_pos - goal).norm();
    if (node_goal_dist < goal_radius)
    {
      double goal_cost = goal_dist_cost_factor_ * node_goal_dist;
      graph_.add_edge(id, virtual_goal, goal_cost);
    }
  }
  

  // if current_time is within path_smoothness_period_ of latest_frontier_time_, prefer local frontiers
  bool use_local_frontiers = false;
  if (!local_scored_frontiers.empty())
  {
    if ((current_time - *latest_frontier_time_).seconds() < path_smoothness_period_)
    {
      for (auto id: local_scored_frontiers)
      {
        double frontier_cost = frontier_scores_[id].second.second;
        graph_.add_edge(virtual_goal, id, frontier_cost);
        use_local_frontiers = true;
      }
    }
    else
    {
      latest_frontier_time_ = current_time;
    }
  }  
  
  if (!use_local_frontiers)
  {
    for (auto& [id, score_pair] : frontier_scores_)
    {
      double frontier_cost = score_pair.second.second;
      graph_.add_edge(virtual_goal, id, frontier_cost);
    }
    latest_frontier_time_ = current_time;
  }

  auto path = graaf::algorithm::dijkstra_shortest_path(graph_, current_node_idx_, virtual_goal);
  std::vector<Eigen::Vector3d> path_points;
  bool has_frontier_in_path = false;
  if (path)
  {
    size_t idx = 0;
    for (const auto& node_id : path->vertices)
    {
      const auto& node = graph_.get_vertex(node_id);
      const auto& pos = node.pose.position;
      path_points.push_back(Eigen::Vector3d(pos.x, pos.y, pos.z));
      if (idx == path->vertices.size() - 2 && trav_class_idx_ < node.trav_properties.size() &&
          node.trav_properties[trav_class_idx_].is_frontier)
      {
        // if second to last, add frontier point as well
        if (!node.trav_properties[trav_class_idx_].frontier_points.empty())
        {
          Eigen::Vector3d mean_frontier(0.0, 0.0, 0.0);
          double n_frontier_points = node.trav_properties[trav_class_idx_].frontier_points.size();
          for (const auto& frontier_point : node.trav_properties[trav_class_idx_].frontier_points)
          {
            mean_frontier += Eigen::Vector3d(frontier_point.x, frontier_point.y, frontier_point.z);
          }
          mean_frontier /= n_frontier_points;
          path_points.push_back(mean_frontier);
        }

        latest_frontier_ = Eigen::Vector3d(node.pose.position.x, node.pose.position.y, node.pose.position.z);
        has_frontier_in_path = true;
      }
      idx++;
    }
  }
  if (!has_frontier_in_path)
  {
    latest_frontier_.reset();
    RCLCPP_WARN(logger_, "NO FRONTIER IN PATH!!!!!");
  }

  graph_.remove_vertex(virtual_goal);
  return path_points;
}

visualization_msgs::msg::MarkerArray Planner::get_score_visualization(const rclcpp::Time& stamp,
                                                                      std::string frame_id,
                                                                      bool with_id_text) const
{
  visualization_msgs::msg::MarkerArray markers;

  // delete all previous markers
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

  //visualize ring of local_frontier_radius around latest_frontier_
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
    // delete
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    delete_marker.header.frame_id = frame_id;
    delete_marker.header.stamp = stamp;
    delete_marker.ns = "local_frontier";
    delete_marker.id = 0;
    markers.markers.push_back(delete_marker);
  }

  // add text for time_diff
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
    text_marker.pose.position.z = latest_frontier_->z() + 1.0;  // raise text above the node
    text_marker.scale.z = 1.5;           // only scale.z is used for text
    text_marker.color.a = 1.0;
    text_marker.color.r = 0.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    std::ostringstream ss;
    ss << "Exploitation\nTimeout: " << path_smoothness_period_ - time_diff;
    text_marker.text = ss.str();
    markers.markers.push_back(text_marker);
  }

  // visualize cubes for nodes with inf frontier cost
  int id_counter = 1;

  // normalize costs to [0,1] for visualization
  double min_val = 30.0;
  double max_val = 400.0;

  for (const auto& [id, score_pair] : frontier_scores_)
  {
    const auto& node = score_pair.first;
    double frontier_score = score_pair.second.first;
    double frontier_cost = score_pair.second.second;
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
      // marker.color.a = 0.5;
      // marker.color.r = 1.0;
      // marker.color.g = 0.0;
      // marker.color.b = 0.0;
      marker.color = color;
      markers.markers.push_back(marker);
    }
    if (with_id_text)
    {
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.frame_id = frame_id;
      text_marker.header.stamp = stamp;
      text_marker.ns = "frontier_id_text";
      text_marker.id = id_counter;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose = node.pose;
      text_marker.pose.position.z += 0.1;  // raise text above the node
      text_marker.scale.z = 0.5;           // only scale.z is used for text
      text_marker.color.a = 1.0;
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 1.0;
      std::ostringstream ss;
      // ss << "Score: " << frontier_score << " Cost: " << frontier_cost;
      // upto 2 decimal places
      ss << std::fixed << std::setprecision(2);
      ss << frontier_score << "/" << frontier_cost;
      text_marker.text = ss.str();
      markers.markers.push_back(text_marker);
    }
    id_counter++;
  }
  // }
  return markers;
}


}  // namespace graphnav_planner
