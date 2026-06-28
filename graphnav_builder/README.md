# graphnav_builder

`graphnav_builder` incrementally constructs the geometric sparse navigation
graph described in Section 4-B and Algorithms 1-5 of
[WildOS](https://arxiv.org/abs/2602.19308).

The package intentionally separates ROS communication from the paper
algorithm:

```text
Odometry + local GridMap
        |
ApproximateTimeSynchronizer
        |
bounded MessageBuffer + historical TF
        |
TraversabilityGrid
        |
SparseGraphBuilder (Algorithms 1-5)
        |
graphnav_msgs/NavigationGraph
```

## Inputs

- `nav_msgs/msg/Odometry`
- `grid_map_msgs/msg/GridMap`

The GridMap requires a configured `traversability_layer`. Optional
`elevation_layer`, `observed_layer`, and `traversability_cost_layer` values
are supported.

By default:

- finite values greater than or equal to `safe_threshold` are free;
- finite values below the threshold are obstacles;
- non-finite values are unknown;
- an invalid `observed_layer` value also marks the cell unknown.

Set `traversability_semantics: lower_is_safer` when the upstream map uses the
opposite convention.

Set `unknown_value_policy: negative_or_non_finite` when the upstream map also
uses negative finite values as unknown sentinels.

## Coordinate frames

The published graph always uses `global_frame`. Inputs already in this frame
are processed directly. Other frames are transformed at their message
timestamps when `use_tf` is true.

Only planar frame transforms are accepted. Roll or pitch would invalidate the
2.5-D GridMap model and therefore reject that input frame.

## Paper parameters

The defaults match Table 1:

| Parameter | Value |
|---|---:|
| reliable geometric range \(r_max\) | 10.0 m |
| `expected_map_resolution` | 0.1 m |
| `max_free_radius` | 4.0 m |
| `num_samples` | 1000 |
| `traversable_radius` | 0.5 m |
| `edge_radius` | 8.0 m |

The local map data structure is the rectangular `GridMap` described by
`length_x × length_y`, matching Figure 3 and the algorithms' use of
“map bounds”. The paper's `r_max=10 m` describes reliable geometric sensing;
it does not by itself specify that the matrix is 10 m × 10 m or 20 m × 20 m.

`crop_to_local_radius` is enabled in the deployment configuration. Cells
outside `local_map_radius` become unknown and cannot be sampled or traversed.
Unknown cells directly adjacent to reliable free space remain valid frontier
endpoints, so the radial sensing boundary can still drive exploration.

`sample_against_new_nodes: true` rejects overlap both against the previous
graph and against nodes accepted earlier in the same update. Setting it to
false reproduces Algorithm 3 literally, but can create a dense initial graph.

## Geometric frontiers

A graph node always lies in free space. A frontier point is stored at the
center of an unknown cell adjacent to free space. The owning graph node becomes
a frontier node through `is_frontier=true`.

`frontier_connectivity: 4` requires the free and unknown cells to share an
edge. Set it to 8 to also treat diagonal contact as a frontier.

Frontier reachability uses a dedicated rule: every rasterized path cell before
the endpoint must be free and provide more than `traversable_radius` obstacle
clearance, while the final endpoint must be the selected unknown frontier
cell. Historical frontier points are removed when they become known, leave
the current map, fall within another node's explored radius, or lose their
safe approach path. The owning node is excluded from that explored-radius
cleanup, matching Figure 3(h)'s "another node" rule and preventing a retained
frontier from being deleted and reconstructed every frame.

Frontier identity uses the current GridMap `(row, column)` cell rather than a
world-axis coordinate quantization. This remains unique for rotated and
rolling GridMaps.

## Verifying the upstream GridMap

The repository does not contain the producer of `traversability_map`, so its
physical dimensions cannot be inferred statically. On the first received map,
the node logs:

- `length_x × length_y`;
- cell rows and columns;
- resolution and frame;
- free, obstacle, and unknown counts on the rectangular boundary;
- the actual number of Free/Unknown frontier candidates after radial masking.

If no Free/Unknown frontier exists, the node warns that exploration has no
geometric candidate. `require_frontier_candidates` can reject such updates
when a deployment requires continuous exploration. Once the deployment size is known,
`expected_map_length_x` and `expected_map_length_y` can be configured; setting
`strict_map_size_check` rejects mismatching maps.

## Safety extensions

`validate_historical_edges` is separate from Algorithm 5. It validates the
visible portion of historical edges against newly observed obstacles. Edges
partly outside the current map are clipped to the map rectangle; the unseen
portion is not invalidated.

Line checks use supercover rasterization so cells touched at diagonal corners
are included.

When no existing node is safely reachable from the robot, the default
`ensure_robot_anchor` mode adds a deterministic node at the robot pose. The
published `current_node_idx` prefers a node connected to the robot by a path
that satisfies both obstacle and unknown clearance.

Each update reports graph size, connected components, current-component size,
reachable and unreachable frontier counts, degree statistics, frontier path
checks, edge candidates/validations, update latency, TF drops, and
processing-buffer drops. A separate `Graph stage timings` line reports
`distance_fields`, `update_nodes`, `sample_nodes`, `robot_anchor`,
`update_frontiers`, `validate_edges`, `build_edges`, `current_node`, and
`total`. Slow-update warnings include the same stage breakdown. Warnings are
also emitted when frontiers exist but none are reachable from the current
node.

## Edge costs

`edge_cost_mode: euclidean` is the default expected by
`graphnav_planner`. The optional `integrated_traversability` mode scales
Euclidean distance by mean normalized risk along the edge. Non-unit cost
layers must configure `cost_min`, `cost_max`, and
`cost_higher_is_riskier`; `strict_cost_range` rejects out-of-contract values.

## ROS input reliability

Odometry and GridMap use independent explicit QoS profiles. The deployment
defaults use BEST_EFFORT odometry and RELIABLE GridMap input. Match these
parameters to the actual publisher endpoints reported by
`ros2 topic info <topic> -v`.

Synchronized messages wait for timestamp-correct TF only up to
`tf_message_max_age` or `tf_max_lookup_attempts`. A permanently unavailable
transform therefore cannot block the processing queue indefinitely; timed-out
messages are dropped with diagnostics.

The deployment defaults set both `grid_qos_depth` and `message_buffer_size` to
1 with `drop_old_messages: true`. If map production is faster than graph
construction, the builder therefore processes the newest available local map
instead of accumulating stale rolling-map updates.

## Debug memory

The paper uses the sparse graph as global memory. Consequently, dense global
GridMap reconstruction is not performed. Optional RViz markers use a bounded
sparse cell dictionary and are disabled by default.

## Launch

```bash
ros2 launch graphnav_builder graphnav_builder.launch.py \
  namespace:=spot1 global_frame:=spot1/odom
```

With the default relative topic names this subscribes to
`/spot1/odom` and `/spot1/traversability_map`, then publishes
`/spot1/nav_graph`.
