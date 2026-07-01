# Visual Navigation

ROS 2 navigation package for WildOS and baseline implementations. This package contains the main navigation pipelines, scoring logic, and utility nodes.

## Modules

| Module | Description |
|---|---|
| `wildos/` | **WildOS** — Full navigation pipeline with ExploRFM inference, graph scoring, and object search |
| `explorfm_triangulation/` | Particle-filter-based object triangulation nodes |
| `imgfrontier_nav/` | Image frontier navigation baseline (vision-only, no geometry) |
| `lrn/` | [LRN](https://arxiv.org/abs/2504.13149) baseline (vision-only, no geometry) |
| `geofrontier_nav/` | Geometric frontier navigation with fixed goal scoring |
| `gps/` | GPS visualization and metric logging nodes |
| `utils/` | Shared scoring and utility functions |

## Key Files

| File | Description |
|---|---|
| `wildos/nav.py` | WildOS main node — runs ExploRFM inference and publishes scored navigation graph |
| `wildos/goalagnostic_scoring.py` | Goal-agnostic frontier scoring combining traversability and frontier predictions |
| `utils/scoring.py` | Graph scoring utilities shared across navigation methods |
| `explorfm_triangulation/obj_mask_triangulation.py` | Object mask triangulation (used during WildOS deployment) |
| `explorfm_triangulation/explorfm_triangulator.py` | Standalone ExploRFM triangulation node (for testing) |
| `imgfrontier_nav/viz_net.py` | ExploRFM output visualization (debugging tool) |

> See the [main README](../README.md) for launch commands and deployment instructions.

## Configuration

YAML config files for each exectuable are in `configs/`:

| Config | Used By |
|---|---|
| `wildos_nav_conf.yaml` | WildOS navigation |
| `imgfrontier_nav_conf.yaml` | Image frontier baseline |
| `lrn_nav_conf.yaml` | LRN baseline |
| `geofrontier_nav_conf.yaml` | Geometric frontier navigation |
| `explorfm_triangulator_conf.yaml` | Standalone ExploRFM triangulation |
| `triangulation3d_objsearch_conf.yaml` | Object search triangulation |

## WildOS synchronization tuner

`wildos_sync_tuner` observes the configured image, CameraInfo, odometry,
NavigationGraph, GridMap, TF, and scored-graph topics. It reports their receive
rates, timestamp delays, publisher QoS, timestamp span, and the queue depth that
would have been needed for each observed synchronization group. It only reads
topics; it does not modify a YAML file or any running node.

Build and run it from the workspace root while the complete navigation system
is active and moving normally:

```bash
colcon build --symlink-install --packages-select visual_navigation
source install/setup.bash
ros2 run visual_navigation wildos_sync_tuner \
  --config visual_navigation/configs/wildos_nav_conf.yaml \
  --graph-config graphnav_builder/config/graphnav_builder.yaml \
  --duration 60 \
  --ros-args -p use_sim_time:=true
```

Omit `--ros-args -p use_sim_time:=true` on a physical robot. For a conservative
QoS-depth estimate, provide a measured worst-case model inference duration with
`--processing-seconds`; otherwise the tool uses the scored-graph output period
as an upper-bound estimate:

```bash
ros2 run visual_navigation wildos_sync_tuner \
  --duration 120 --percentile 99 --processing-seconds 1.2
```

### ObjectMask/LiDAR triangulation tuner

`objmask_sync_tuner` performs the corresponding measurement for
`triangulation3d_objsearch_conf.yaml`. Keep the target object visible so WildOS
continues publishing `ObjectMaskWithTf` messages:

```bash
ros2 run visual_navigation objmask_sync_tuner \
  --config visual_navigation/configs/triangulation3d_objsearch_conf.yaml \
  --duration 60 \
  --ros-args -p use_sim_time:=true
```

It estimates `syncsub_queue_size` from the actual number of newer LiDAR clouds
that arrived before each delayed mask, and estimates `syncsub_slop` from the
nearest mask/cloud timestamp offsets. On a physical robot, omit the simulated
time arguments. If fewer than three particle outputs are available, pass the
measured worst processing duration to obtain firm QoS and processing-buffer
recommendations:

```bash
ros2 run visual_navigation objmask_sync_tuner \
  --duration 120 --percentile 99 --processing-seconds 0.4
```

## Method Details

### WildOS
WildOS scores frontier nodes of the navigation graph using ExploRFM predictions. The scoring combines traversability (is it safe?), visual frontier confidence (where to explore?), and object similarity (does it match the query?). When `do_object_search` is enabled, the `obj_mask_triangulation` node is automatically launched to estimate coarse goal positions using a particle filter.

### Image Frontier Navigation (Baseline)
Assumes a single geometric frontier at the center-bottom pixel of each camera image. Projects a path from the bottom-center pixel to the chosen visual frontier using the depth image and sends a goal at `lookahead_dist` along the projected path to the local planner.

### LRN (Baseline)
A purely vision-based baseline that does not use geometric information for exploration. It scores angular bins around the robot using visual frontier scores and the goal heading.
