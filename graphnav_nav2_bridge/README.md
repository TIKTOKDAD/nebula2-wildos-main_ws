# graphnav_nav2_bridge

Scheme-A execution adapter for the Clearpath A300 simulation:

```text
/scored_nav_graph + /imgnav_waypoint
                 -> graphnav_planner/path
                 -> /goal_pose
                 -> /graphnav_navigate_to_pose
                 -> Navfn + MPPI
                 -> velocity_smoother
                 -> collision_monitor
                 -> /a300_0000/cmd_vel (TwistStamped)
```

The Nav2 costmaps and collision monitor consume
`/a300_0000/sensors/lidar3d_0/scan`. The stack is mapless: both costmaps are
rolling windows in `odom`, which is appropriate because graphnav's path follower
supplies a nearby look-ahead goal rather than the final long-range destination.
The dedicated `/graphnav_navigate_to_pose` action prevents legacy Nav2 clients
or RViz tools on `/navigate_to_pose` from repeatedly preempting this stack.
If `/goal_pose` stops for more than two seconds, the bridge cancels the active
goal so that Nav2 does not continue toward a stale look-ahead point.

## Build

```bash
cd ~/nebula2-wildos-main
source /opt/ros/jazzy/setup.bash
colcon build --packages-select graphnav_nav2_bridge
source install/setup.bash
```

## Run

Start the simulator, DLIO, graph builder, WildOS, and `graphnav_planner`
first. Then run the downstream Nav2 execution package:

```bash
ros2 launch graphnav_nav2_bridge graphnav_nav2.launch.py
```

For a single command that starts `graphnav_planner` as well:

```bash
ros2 launch graphnav_nav2_bridge graphnav_nav2.launch.py \
  start_graphnav_planner:=true
```

For configuration-only diagnostics, the bridge can also be disabled with
`start_bridge:=false`, ensuring that no navigation goal is forwarded.

The final command topic is configurable, while defaulting to the requested
`/a300_0000/cmd_vel`:

```bash
ros2 launch graphnav_nav2_bridge graphnav_nav2.launch.py \
  cmd_vel_topic:=/a300_0000/cmd_vel
```

Required runtime interfaces:

- `/clock`
- `/tf` and `/tf_static`, including `odom -> base_link` and
  `base_link -> lidar3d_0_sensor_link`
- `/dlio/odom_node/odom`
- `/a300_0000/sensors/lidar3d_0/scan`
- `/scored_nav_graph`
- `/imgnav_waypoint`
- a subscriber/bridge for `/a300_0000/cmd_vel`

Useful checks:

```bash
ros2 lifecycle get /planner_server
ros2 lifecycle get /controller_server
ros2 action info /graphnav_navigate_to_pose
ros2 topic info /goal_pose -v
ros2 topic info /a300_0000/cmd_vel -v
ros2 topic echo /collision_monitor_state
```
