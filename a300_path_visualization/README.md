# A300 Path Visualization

该 ROS 2 包把 A300 的 DLIO 里程计累计成一条标准 `nav_msgs/Path`，用于在
RViz 中显示车辆已经走过的实际轨迹。

它不负责路径规划，也不记录 GPS/CSV；唯一的数据流是：

```text
/dlio/odom_node/odom
        -> a300_path_recorder
        -> /a300_0000/driven_path
```

## 构建

```bash
cd ~/nebula2-wildos-main
source /opt/ros/jazzy/setup.bash
colcon build --packages-select a300_path_visualization --symlink-install
source install/setup.bash
```

## 启动

先启动 Gazebo、A300 和 DLIO，再执行：

```bash
ros2 launch a300_path_visualization a300_path_visualization.launch.py
```

## 在 RViz 中显示

1. 把 `Fixed Frame` 设置为 `odom`。
2. 点击 `Add`，选择 `Path`。
3. 把 Path 的 `Topic` 设置为 `/a300_0000/driven_path`。
4. 根据需要调整颜色和线宽。

配置文件为 `config/a300_path_visualization.yaml`。其中
`min_sample_distance` 控制轨迹点间距，`max_path_points` 防止历史轨迹无限增长，
`publish_rate` 只控制完整 Path 的发送频率，不影响轨迹点采样精度。
