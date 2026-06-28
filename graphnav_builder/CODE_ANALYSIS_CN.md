# graphnav_builder 代码分析

## 1. 包的目标与数据流

`graphnav_builder` 将每帧局部 `GridMap` 转换为一个会跨帧保留的稀疏无向导航图。它实现 WildOS 论文第 4-B 节的算法 1--5，并加入了几项面向真实 ROS 部署的安全与可观测性扩展。

```text
Odometry + GridMap
      │（近似时间同步）
      ▼
MessageBuffer ──等待历史 TF──► TraversabilityGrid
      │                              │
      │                              ▼
      │                    SparseGraphBuilder（算法 1--5）
      │                              │
      └────诊断 / 可选 RViz──────────► NavigationGraph
```

职责被刻意拆开：`builder_node.py` 处理 ROS 通信和时间一致性；`traversability_grid.py` 处理 GridMap 的物理存储和几何语义；`graph_builder.py` 是不依赖 rclpy 的纯算法层。这种边界使算法可以不启动 ROS 即被单元测试，也避免 TF、QoS 的问题渗入算法实现。

## 2. 核心数据约定

### 坐标和索引

- `Cell = (row, column)`：矩阵索引，不是世界米制坐标。
- `XY = (x, y)`：全局平面坐标，单位为米。
- `GraphNodeState`：节点的 `Pose`、UUID、自由半径、探索半径和前沿点。
- `GraphState`：节点列表、无向边集合、边代价字典和当前节点索引。

边一律以端点升序元组 `(smaller_idx, larger_idx)` 存储。节点删除后列表索引会改变，因此算法 2 必须同时重映射边和边代价；这是此包最重要的状态一致性约束之一。

### 栅格状态

`TraversabilityGrid` 将每个格分类为：

- `UNKNOWN`：值非有限、负值哨兵（按配置）或观测掩码无效；
- `FREE`：已观测，且通行性数值满足安全阈值；
- `OBSTACLE`：已观测但不满足安全阈值。

`active` 与上述状态独立。圆形可靠半径以外的单元会改为 `UNKNOWN` 且 `active=False`，因此不能采样和穿越；不过它们保留 `frontier_endpoint_allowed=True`，可以成为路径的最后一个前沿端点。这个细节让“矩形存储、圆形可靠传感范围”同时成立。

## 3. 核心算法：graph_builder.py

### 算法 1：`update_navigation_graph`

每帧严格按以下顺序执行：

1. 计算未知区域净空和障碍物净空；未知净空把地图矩形外也当作未知。
2. 运行算法 2 更新/清除历史节点。
3. 运行算法 3 从可靠自由空间采样新节点。
4. 若没有安全可达节点，视配置添加机器人锚点。
5. 运行算法 4 维护前沿点。
6. 视配置用本帧障碍验证旧边。
7. 运行算法 5 添加新边并计算代价。
8. 更新 `current_node_idx` 并返回持久 `GraphState`。

不能交换第 2 步和后续步骤：算法 3--5 都依赖算法 2 完成的节点删除和索引重映射。

### 算法 2：`update_nodes`

若旧节点在当前可靠局部图内，代码以最新的未知/障碍净空更新 `free_radius`，并用 `max(旧值, 当前未知净空)` 更新 `explored_radius`。自由半径为零的节点被删除；当前局部图外的节点不因“当前不可见”而删除，继续作为全局历史。删除后通过 `index_remap` 重建边集合和代价字典。

### 算法 3：`sample_new_nodes`

从自由格随机抽样。候选格必须位于可选工作区内，且未知和障碍净空的最小值严格大于 `traversable_radius`。`SpatialHashIndex` 会查询附近既有节点，若候选落入任何节点的 `free_radius` 覆盖圆则拒绝。

`sample_against_new_nodes=False` 复现论文中仅与 `V_(t-1)` 比较的严格含义；默认 `True` 还会排斥本帧已经接受的新节点，使初始图更稀疏。

### 算法 4：`update_frontier_nodes`

前沿定义为“与自由格相邻的未知格”。前沿点放在未知格中心，但该点归属到一个自由空间图节点，节点自己不会移动到未知区。更新分两段：

1. 清理旧前沿：若它变已知、离开允许地图区域、落入任一节点的历史探索半径，或从归属节点无法以安全净空接近，就删除。
2. 发现新前沿：对每个 Free/Unknown 边界格，寻找最近的、可通过自由格接近的节点；每个 `frontier_key` 在一次更新中只能归属一次。

`frontier_connectivity=4` 只接受共享边的接触，值为 8 时也接受对角接触。前沿路径仅允许最后一格未知，前面的自由格必须有超过机器人半径的障碍净空。

### 算法 5：`build_edges`

每个节点只与 `edge_radius` 内、索引更大的候选节点尝试连边，所以无向边不会重复。`clearance_collision_free` 要求 supercover 栅格化线段所触及的所有格都是自由格，并同时远离未知边界和障碍物超过 `traversable_radius`。

边代价默认是三维欧氏距离；`integrated_traversability` 模式会乘以 `1 + weight * mean_risk`。该模式影响规划偏好，不会放宽碰撞或净空判定。

### 额外安全机制

- `validate_existing_edges`：只验证旧边在当前局部地图中可见的部分；地图外的未知历史段不会导致误删。若可见段出现障碍或净空过窄，删除该边。
- `ensure_robot_anchor_node`：随机采样遗漏机器人附近时，只要机器人处于有足够净空的自由格，就添加一个确定性锚点。
- `update_current_node`：优先选择从机器人安全可达的最近节点；只有地图数据不可用或无安全候选时，才回退到几何最近点并把 `current_node_is_safe` 标为假。
- `graph_diagnostics`：通过 DFS 统计连通分量、当前分量与前沿可达性，只读状态，不会改变导航图。

## 4. 地图几何：utils/traversability_grid.py

### GridMap 解码

ROS `GridMap` 图层可能是列主序，也可能是行主序，并可作为循环缓冲区滚动。`layer_shape` 先依据布局标签确定逻辑维度；`decode_layer` 再处理 `data_offset`、物理存储顺序和 `outer_start_index` / `inner_start_index`，最终统一返回逻辑行主序数组。上层代码因此不需要知道底层缓冲区是否刚刚滚动过。

### 坐标变换

地图自身姿态可能有 yaw，输入 frame 也可能通过 TF 转到全局 frame。`map_yaw` 是两者之和。`world_to_map_axes` / `map_axes_to_world` 互为逆变换；`xy_to_cell` 与 `cell_to_xy` 使用它们保证旋转地图仍能正确索引。

`transforms.py` 只接受平面 TF。四元数会先归一化；若 roll 或 pitch 超出容差就报错，因为倾斜坐标系无法保持 2.5D 栅格的水平距离含义。

### 线段和碰撞

`supercover_cells` 不使用会漏角点的普通 Bresenham 语义。线段恰穿过格点时，两个正交侧格都会加入结果，因此障碍物不能从边的角落被“擦过”。

普通边使用 `line_cells`，端点必须都在地图内；历史边验证使用 `clip_segment_to_bounds`，先以 Liang--Barsky 算法裁剪到旋转矩形，保证只检查当前能观测到的部分。

### 距离场

`distance_field` 使用可分离的一维平方欧氏距离变换（EDT）：先按列、再按行变换，每个维度线性复杂度。`clearance_field` 进一步减去一个单元半对角线，得到保守的“到目标单元边界”距离，避免把中心距离误当作机器人足迹净空。

当 `include_map_exterior=True` 时，代码在矩形外补一圈零距离虚拟目标，等价于将本地地图外视为未知。这使节点与新边不会贴着局部观测边缘生成。

## 5. ROS 适配：builder_node.py

### 参数和 QoS

`declare_graph_parameters` 声明所有参数，`read_parameters` 在启动时读取并转为 Python 基本类型。里程计和 GridMap 的 QoS 独立配置：典型组合为 BEST_EFFORT 里程计与 RELIABLE GridMap，避免其中一种发布端不兼容导致同步器没有输入。

### 同步、缓冲与 TF

`ApproximateTimeSynchronizer` 配对里程计和地图，回调 `listener_callback` 只把它们推入 `MessageBuffer`。定时器 `process_buffer` 始终先处理最旧项，且在消息自己的时间戳查询：

- `global_frame <- odom_frame`；
- `global_frame <- grid_map_frame`。

TF 暂不可用时，最旧消息保留重试；超过 `tf_message_max_age` 或 `tf_max_lookup_attempts` 后才丢弃。实时缓冲模式满载时丢最旧消息，严格顺序模式满载时拒绝最新消息，两个计数都会进入诊断日志。

### 处理和发布

`do_processing` 先转换机器人位置，再构造 `TraversabilityGrid`，可选应用圆形掩码并验证输入地图合同。全未知地图会跳过更新、保留历史图。有效地图送入纯算法层后，节点发布 `NavigationGraph`，并可选把自由/障碍格写入只用于 RViz 的 `GlobalTraversabilityMemory`。

`validate_grid_contract` 不在源码中臆测上游地图尺寸：它记录首次收到的真实几何、边界状态和前沿数量，再按部署者可选配置的期望长度做告警或严格拒绝。

## 6. 其他工具模块

| 文件 | 作用 | 关键约束 |
|---|---|---|
| `utils/graph_data.py` | 定义图状态、UUID、距离函数 | 节点索引与边键必须同步 |
| `utils/graph_messages.py` | 内部状态转 ROS `NavigationGraph` | 空图索引安全；非有限半径不会发布 |
| `utils/message_buffer.py` | 有界消息队列 | 使用单调时钟计算 TF 等待时间 |
| `utils/spatial_index.py` | 空间哈希、半径与精确最近邻 | 桶下界堆搜索可在谓词筛选后保持精确 |
| `utils/global_memory.py` | 有界稀疏调试视图 | 不参与图构建；容量满后仍可覆写已有格 |
| `utils/viz.py` | 稀疏记忆转 RViz MarkerArray | 先 DELETEALL，按 stride 降低显示密度 |
| `utils/transforms.py` | 平面 TF 与位置转换 | 明确拒绝 roll/pitch |

## 7. 启动、配置与打包

- `config/graphnav_builder.yaml` 是默认参数的单一事实来源。注释区分了矩形 GridMap 存储、圆形可靠感知范围、前沿连通性、历史边验证、代价模式、TF 超时和纯调试功能。
- `launch/graphnav_builder.launch.py` 提供命名空间、仿真时钟、配置文件、日志等级和 `global_frame` 覆盖；使用 `FindPackageShare`，因此安装后也能正确定位 YAML。
- `setup.py` 安装 Python 包、ament 索引、README、launch 与 YAML，并注册 `graphnav_builder.builder_node:main` 为 `ros2 run` 入口。

## 8. 测试覆盖范围

`test/helpers.py` 构造带非零 `data_offset`、不同存储序、循环索引、可选图层、地图 yaw 和 TF 的最小 GridMap，避免测试依赖实际 ROS 发布器。

- `test_traversability_grid.py`：解码、观察语义、坐标往返、圆形裁切、EDT、supercover、前沿路径、历史边裁剪、风险归一化。
- `test_graph_builder.py`：算法 1--5、采样模式、前沿生命周期、历史边、安全当前节点、机器人锚点、连通性诊断。
- `test_map_shift.py`：滚动局部地图下的全局节点坐标、历史节点/边、重叠区更新、前沿和循环缓冲区。
- `test_ros_adapters.py`：QoS、消息缓冲策略、序列化、TF 超时、frame 合同与运行时地图合同。
- `test_flake8.py` / `test_pep257.py`：静态风格与文档字符串规则。

## 9. 使用和维护时的注意点

1. 不要把 `UNKNOWN` 当作障碍物或自由区的普通别名：它影响采样、边净空、前沿终点和地图外边界，语义不同。
2. 不要在更新节点列表后保留旧边索引；必须像 `update_nodes` 一样重映射边与 `edge_costs`。
3. 改动线段栅格化时必须保留 supercover 的角点分支，否则会形成“从障碍物角落穿过”的安全漏洞。
4. 改动 TF 处理时必须用消息时间戳而不是最新变换，否则运动机器人上的地图与里程计会空间错位。
5. `GlobalTraversabilityMemory` 只是可视化；不要把它接入算法，否则会改变论文所用的稀疏图全局记忆模型和内存上界。
