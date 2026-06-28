# =============================================================================
# graphnav_planner ROS 2 C++ 包构建配置
# =============================================================================
# colcon 构建该包时会调用本文件。它主要完成以下工作：
#   1. 查找 ROS 2、Eigen、TF2、导航图消息和 graaf 图算法库等依赖；
#   2. 编译图规划器组件 graphnav_planner_component；
#   3. 编译路径跟随器组件 path_follower_component；
#   4. 为两个组件注册可由 ros2 run/launch 启动的节点入口；
#   5. 安装共享库、可执行程序和公共头文件。
#
# 本文件只控制“如何编译和安装”，不会设置运行时话题、ROS 参数或规划算法权重。
# 运行时参数和话题重映射主要位于 `launch/graphnav_planner.launch.yml`。

# 要求 CMake 版本至少为 3.8。低于该版本时直接停止配置。
cmake_minimum_required(VERSION 3.8)

# 声明当前工程/ROS 包名。该名称应与 package.xml 中的 <name> 一致。
# `${PROJECT_NAME}` 在本文件中展开为 `graphnav_planner`。
project(graphnav_planner)

# 使用 C++20 编译。
# 本项目代码使用结构化绑定、std::optional 等现代 C++ 特性，编译器必须支持该标准。
set(CMAKE_CXX_STANDARD 20)

# 如果编译器无法支持 C++20，则配置失败，不允许自动退回旧标准。
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GNU GCC 或 Clang 编译器下开启常用警告。
# -Wall：开启大部分常见警告；
# -Wextra：补充未使用参数等额外检查；
# -Wpedantic：检查不符合标准 C++ 的编译器扩展。
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# 查找构建依赖
# -----------------------------------------------------------------------------
# REQUIRED 表示依赖缺失时立即终止 CMake 配置。依赖通常由 ROS 2 环境、
# 当前工作空间或系统安装提供，并应与 package.xml 中的依赖声明保持一致。

# ROS 2 ament_cmake 构建系统，提供 ament_target_dependencies、ament_package 等宏。
find_package(ament_cmake REQUIRED)

# ROS 2 C++ 客户端库，用于 Node、Publisher、Subscriber、参数、日志和时间。
find_package(rclcpp REQUIRED)

# ROS 2 组件机制，用于把节点编译成共享库并注册到 component container。
find_package(rclcpp_components REQUIRED)

# 本项目自定义导航图消息，包括 NavigationGraph、Node、Edge、UUID 等。
find_package(graphnav_msgs REQUIRED)

# RViz Marker/MarkerArray 消息，用于发布前沿评分和图结构可视化。
find_package(visualization_msgs REQUIRED)

# nav_msgs/Path 和 nav_msgs/Odometry，分别用于规划路径输出和机器人位置输入。
find_package(nav_msgs REQUIRED)

# std_msgs/Header、ColorRGBA 等基础消息类型。
find_package(std_msgs REQUIRED)

# grid_map_msgs/GridMap，用于发布未探索空间和目标距离调试栅格。
find_package(grid_map_msgs REQUIRED)

# Eigen3 线性代数库，用于三维向量、旋转矩阵、四元数和距离计算。
find_package(Eigen3 REQUIRED)

# TF2 ROS 接口，提供 Buffer 和 TransformListener。
find_package(tf2_ros REQUIRED)

# TF2 核心坐标变换库。
find_package(tf2 REQUIRED)

# TF2 与 Eigen 类型之间的转换支持。
find_package(tf2_eigen REQUIRED)

# TF2 与 geometry_msgs/PoseStamped 等 ROS 几何消息之间的转换支持。
find_package(tf2_geometry_msgs REQUIRED)

# PoseStamped、Point 等几何消息。
find_package(geometry_msgs REQUIRED)

# graaf_vendor 提供 graaflib 图数据结构和 Dijkstra 最短路算法。
find_package(graaf_vendor REQUIRED)


# -----------------------------------------------------------------------------
# 图导航规划器组件
# -----------------------------------------------------------------------------

# 创建共享库 `libgraphnav_planner_component.so`。
# 共享库包含：
#   planner.cpp：NavigationGraph 转换、前沿代价、未知空间距离图和 Dijkstra 规划；
#   planner_node.cpp：ROS 订阅/发布、TF 转换、参数读取和组件节点封装。
add_library(graphnav_planner_component SHARED
  src/planner.cpp
  src/planner_node.cpp
)

# 设置该组件的头文件搜索路径。
target_include_directories(graphnav_planner_component PUBLIC
  # 在源码工作空间中编译时，使用当前包的 include 目录。
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

  # 包安装后，其他目标通过安装前缀下的 include 目录引用公共头文件。
  $<INSTALL_INTERFACE:include>

  # 加入 Eigen3 头文件目录。Eigen 是模板库，主要通过头文件使用。
  ${EIGEN3_INCLUDE_DIRS}
)

# 将 ROS 2 和第三方依赖附加到图规划器组件。
# ament_target_dependencies 会处理 include 路径、编译定义、库链接和依赖顺序。
ament_target_dependencies(graphnav_planner_component
  # ROS 2 节点与组件框架。
  rclcpp
  rclcpp_components

  # 输入导航图和调试/路径消息。
  graphnav_msgs
  visualization_msgs
  nav_msgs
  std_msgs
  geometry_msgs
  grid_map_msgs

  # 坐标变换相关依赖。
  tf2_ros
  tf2
  tf2_eigen
  tf2_geometry_msgs

  # 无向图容器和 Dijkstra 最短路径算法。
  graaf_vendor
)

# 当前调用没有额外库参数，因此不会增加新的链接库。
# 实际依赖链接已由上面的 ament_target_dependencies 完成；保留该空调用不改变现有行为。
target_link_libraries(graphnav_planner_component)

# 向 rclcpp_components 注册 `graphnav_planner::PlannerNode`。
# PLUGIN 必须与 planner_node.cpp 中 RCLCPP_COMPONENTS_REGISTER_NODE 的完整类名一致。
# EXECUTABLE 同时生成名为 `planner_node` 的独立启动入口，因此可以执行：
#   ros2 run graphnav_planner planner_node
rclcpp_components_register_node(graphnav_planner_component PLUGIN "graphnav_planner::PlannerNode" EXECUTABLE planner_node)


# -----------------------------------------------------------------------------
# 路径跟随器组件
# -----------------------------------------------------------------------------

# 创建路径跟随器共享库 `libpath_follower_component.so`。
# path_follower_node.cpp 接收规划器发布的 nav_msgs/Path 和里程计，
# 在路径上选择前视 waypoint，并发布 geometry_msgs/PoseStamped 局部目标。
add_library(path_follower_component SHARED src/path_follower_node.cpp)

# 路径跟随器使用与规划器相同的公共头文件/Eigen 搜索路径约定。
target_include_directories(path_follower_component PUBLIC
  # 源码构建阶段的头文件目录。
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

  # 安装后提供给其他包的头文件目录。
  $<INSTALL_INTERFACE:include>

  # Eigen3 头文件目录。
  ${EIGEN3_INCLUDE_DIRS}
)

# 声明路径跟随器实际使用的 ROS 2 依赖。
ament_target_dependencies(path_follower_component
  # ROS 2 节点及组件注册。
  rclcpp
  rclcpp_components

  # 输入 Path/Odometry，输出 PoseStamped。
  nav_msgs
  geometry_msgs

  # 将路径、里程计和目标转换到统一坐标系。
  tf2_ros
  tf2
  tf2_eigen
  tf2_geometry_msgs

  # geometry_msgs 在当前列表中重复出现；CMake/ament 通常会去重，
  # 因而不会改变链接结果。这里保留原有配置，不做功能性修改。
  geometry_msgs
)

# 没有额外手工链接库；依赖链接由 ament_target_dependencies 处理。
target_link_libraries(path_follower_component)

# 注册 `graphnav_planner::PathFollowerNode` 组件，并创建独立可执行入口
# `path_follower_node`。Launch 文件也可以通过组件容器加载同一个 PLUGIN。
rclcpp_components_register_node(path_follower_component PLUGIN "graphnav_planner::PathFollowerNode" EXECUTABLE path_follower_node)


# -----------------------------------------------------------------------------
# 安装编译产物
# -----------------------------------------------------------------------------

# 安装共享库和可执行程序，使 source `install/setup.bash` 后可被 ROS 2 查找。
install(TARGETS
  # 图规划器组件共享库。
  graphnav_planner_component

  # 图规划器独立可执行程序，由 rclcpp_components_register_node 生成。
  planner_node

  # 路径跟随器组件共享库。
  path_follower_component

  # 静态库归档目标的安装位置；当前目标主要是 SHARED，此规则作为通用约定保留。
  ARCHIVE DESTINATION lib

  # `.so` 共享库安装到工作空间的 lib 目录，供组件容器动态加载。
  LIBRARY DESTINATION lib

  # 独立节点可执行程序安装到 `lib/graphnav_planner`，
  # 这是 `ros2 run graphnav_planner ...` 默认查找的位置。
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 文件，使 `ros2 launch graphnav_planner graphnav_planner.launch.yml`
# 能在 source install/setup.bash 后从包的 share 目录找到启动文件。
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# 安装公共头文件目录。
# 源码中的 `include/graphnav_planner/planner.hpp` 最终安装到
# `<install-prefix>/include/graphnav_planner/planner.hpp`。
install(DIRECTORY include/
  DESTINATION include
)

# -----------------------------------------------------------------------------
# 测试与静态检查
# -----------------------------------------------------------------------------

# 只有 colcon/CMake 启用 BUILD_TESTING 时才配置 lint 测试。
if(BUILD_TESTING)
  # 自动发现 package.xml 中声明的 ament lint 测试依赖。
  find_package(ament_lint_auto REQUIRED)

  # 将版权检查标记为已找到，相当于跳过 ament_copyright。
  # 当所有源码补齐规范版权头后，应删除该行以重新启用检查。
  set(ament_cmake_copyright_FOUND TRUE)

  # 将 cpplint 标记为已找到，相当于跳过 ament_cpplint。
  # 若希望执行完整 C++ 风格检查，可删除该行。
  set(ament_cmake_cpplint_FOUND TRUE)

  # 根据 package.xml 的 test_depend 自动注册其余 lint 测试。
  ament_lint_auto_find_test_dependencies()
endif()

# 完成 ament 包导出，生成 package config、环境 hook 和 ament 索引信息。
# 该调用通常必须位于 CMakeLists.txt 末尾。
ament_package()
