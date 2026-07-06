# DLIO Odometry Twist 坐标转换器

## 问题

`nav_msgs/msg/Odometry` 规定 pose 位于 `header.frame_id`，twist 位于
`child_frame_id`。当前 A300 仿真中，DLIO 消息声明：

```text
header.frame_id: odom
child_frame_id: base_link
```

但实测机器人转向后，速度仍沿 odom 世界轴表达。例如车体真实向前约
`0.17 m/s` 时，DLIO 可能输出 `linear.x=-0.035, linear.y=-0.167`。
Nav2 把 `linear.x` 当成车体前进速度，因此会错误判断机器车几乎没有前进。

## 转换

pose 四元数给出 `world_from_body`。节点使用其逆旋转：

```text
v_body = R(world_from_body)^T * v_world
```

平面情况下等价于：

```text
vx_body = cos(yaw) * vx_world + sin(yaw) * vy_world
vy_body = -sin(yaw) * vx_world + cos(yaw) * vy_world
```

节点同时转换线速度、角速度和完整 6x6 twist covariance。pose、pose
covariance、header 和时间戳保持不变。

## 话题

```text
/dlio/odom_node/odom
        -> dlio_odom_twist_adapter
        -> /dlio/odom_node/odom_body_twist
```

GraphNav 仍可使用原始话题读取高精度 pose；Nav2 controller 使用修正话题读取
符合 ROS 约定的车体速度。

## 启动

仿真：

```bash
ros2 launch dlio_odom_twist_adapter dlio_odom_twist_adapter.launch.py \
  use_sim_time:=true
```

实车：

```bash
ros2 launch dlio_odom_twist_adapter dlio_odom_twist_adapter.launch.py \
  use_sim_time:=false
```

## 验证

```bash
ros2 topic echo /dlio/odom_node/odom_body_twist --field twist.twist
```

A300 正常前进或转弯前进时应满足：

```text
linear.x > 0
abs(linear.y) 接近 0
```

转换前后速度向量模长应近似相等，因为旋转只改变坐标表达，不改变物理速度。

## 安全约束

只有确认输入 twist 沿世界坐标轴表达时才应使用此节点。如果未来 DLIO 已修复为
标准车体系 twist，再次旋转会得到错误结果。输入和输出话题必须不同，节点会在
启动时拒绝同名配置以避免消息回环。
