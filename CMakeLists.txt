cmake_minimum_required(VERSION 3.8)
project(roi_depth_query)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(message_filters REQUIRED)
find_package(rcl_interfaces REQUIRED)

# ── roi_depth_node (composable) ──────────────────────────────────────────────
add_library(roi_depth_node SHARED src/roi_depth_node.cpp)
ament_target_dependencies(roi_depth_node
  rclcpp rclcpp_components
  sensor_msgs std_msgs vision_msgs
  realsense2_camera_msgs cv_bridge message_filters)
target_link_libraries(roi_depth_node
  ${realsense2_LIBRARY} ${OpenCV_LIBS})
target_include_directories(roi_depth_node PRIVATE
  ${realsense2_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
rclcpp_components_register_node(roi_depth_node
  PLUGIN "roi_depth_query::RoiDepthNode"
  EXECUTABLE roi_depth_node_exe)

# ── detection_picker_node (composable) ────────────────────────────────────
# Bridges /detections_output (Detection2DArray, network space)
# to /roi (Detection2D, color image space)
add_library(detection_picker_node SHARED src/detection_picker_node.cpp)
ament_target_dependencies(detection_picker_node
  rclcpp rclcpp_components vision_msgs)
rclcpp_components_register_node(detection_picker_node
  PLUGIN "roi_depth_query::DetectionRoiRelayNode"
  EXECUTABLE detection_picker_node_exe)

# ── extrinsics_relay_node (standalone executable) ────────────────────────────
add_executable(extrinsics_relay_node src/extrinsics_relay_node.cpp)
ament_target_dependencies(extrinsics_relay_node
  rclcpp realsense2_camera_msgs rcl_interfaces)

# ── install ──────────────────────────────────────────────────────────────────
install(TARGETS
    roi_depth_node
    detection_picker_node
    extrinsics_relay_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()
