cmake_minimum_required(VERSION 3.8)
project(a300_path_visualization)

# 使用较严格的编译警告，尽早发现类型转换、未使用变量等常见问题。
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 本包仅依赖 ROS 2 C++ 客户端库和标准导航消息。
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 轨迹节点：订阅 Odometry，累计车辆实际走过的位置并发布 nav_msgs/Path。
add_executable(a300_path_recorder src/a300_path_recorder.cpp)
target_compile_features(a300_path_recorder PUBLIC cxx_std_17)
ament_target_dependencies(
  a300_path_recorder
  rclcpp
  nav_msgs
  geometry_msgs
)

# ROS 2 约定把可执行文件安装到 lib/<package_name>，供 ros2 run/launch 查找。
install(
  TARGETS a300_path_recorder
  DESTINATION lib/${PROJECT_NAME}
)

# 安装参数文件和 launch 文件，构建后可直接从 install/share 中加载。
install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
