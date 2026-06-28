cmake_minimum_required(VERSION 3.8)
project(graaf_vendor)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY graaf_src/include/
  DESTINATION include
)

ament_export_include_directories(include)

ament_package()
