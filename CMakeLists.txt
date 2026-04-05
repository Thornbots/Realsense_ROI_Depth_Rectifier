cmake_minimum_required(VERSION 3.8)
project(roi_depth_query)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(message_filters REQUIRED)

add_executable(roi_depth_node src/roi_depth_node.cpp)

target_include_directories(roi_depth_node PRIVATE
  ${realsense2_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(roi_depth_node
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  realsense2_camera_msgs
  cv_bridge
  message_filters
)

target_link_libraries(roi_depth_node
  ${realsense2_LIBRARY}
  ${OpenCV_LIBS}
)

install(TARGETS roi_depth_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
