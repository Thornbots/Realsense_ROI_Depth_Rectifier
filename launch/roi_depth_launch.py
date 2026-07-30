"""
launch/roi_depth_launch.py

Standalone test/dev launch for this package: starts the D435i driver
(align_depth disabled), the extrinsics relay (required so roi_depth_node
can build its colour->depth LUT), and roi_depth_node itself.

roi_depth_node now subscribes /detections_output (Detection2DArray,
NETWORK image space) directly -- it does the network->color bbox scaling
internally (network_width/height, color_width/height params below), so
there is no separate relay node to launch. By default /detections_output
is expected to come from somewhere else (e.g. a hand-published test
message, or your own node).

This file does NOT include the YOLOv8/TensorRT inference chain or the
DJI serial bridge — see realsense_yolov8_nitros_bridge's
isaac_ros_yolov8_realsense.launch.py for the full production pipeline
(camera -> inference -> depth -> serial bridge).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rs_launch = os.path.join(
        get_package_share_directory("realsense2_camera"),
        "launch", "rs_launch.py"
    )

    # Matches the default camera_name/camera_namespace ("camera"/"camera") used
    # by realsense2_camera's stock rs_launch.py, hence the doubled "camera/camera"
    # prefix below and in roi_depth_node's depth_ns/color_ns/extrinsics defaults.
    extrinsics_topic = "/camera/camera/extrinsics/depth_to_color"

    launch_args = [
        DeclareLaunchArgument(
            "detections_topic", default_value="/detections_output"),
        DeclareLaunchArgument("network_width", default_value="640"),
        DeclareLaunchArgument("network_height", default_value="640"),
        DeclareLaunchArgument("color_width", default_value="640"),
        DeclareLaunchArgument("color_height", default_value="480"),
    ]

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            # ---------- keep alignment OFF ----------
            "align_depth.enable": "false",
            # publish extrinsics so the LUT can be exact
            "publish_tf": "true",
            # usual streams
            "enable_depth": "true",
            "enable_color": "true",
            # tune to your use-case
            "depth_module.depth_profile": "640x480x15",
            "rgb_camera.color_profile":   "640x480x15",
        }.items(),
    )

    # Required: without this, roi_depth_node waits forever for extrinsics and
    # never builds its LUT, so it silently never publishes /cv/panel_detections.
    extrinsics_relay = Node(
        package="roi_depth_query",
        executable="extrinsics_relay_node",
        name="extrinsics_relay",
        output="screen",
        parameters=[{
            "extrinsics_topic": extrinsics_topic,
            "target_node":      "/roi_depth_node",
        }],
    )

    roi_depth = Node(
        package="roi_depth_query",
        # NOTE: the composable-node macro names the standalone executable
        # "roi_depth_node_exe" (see CMakeLists.txt EXECUTABLE arg) — the
        # library/plugin target itself is "roi_depth_node", but that is not
        # an installed binary you can `ros2 run`.
        executable="roi_depth_node_exe",
        name="roi_depth_node",
        output="screen",
        parameters=[{
            "depth_ns":          "/camera/depth",
            "color_ns":          "/camera/color",
            "depth_scale":       0.001,   # D435i Z16 default (mm -> m)
            "min_depth_m":       0.1,
            "max_depth_m":       10.0,
            "detections_topic":  LaunchConfiguration("detections_topic"),
            "network_width":     LaunchConfiguration("network_width"),
            "network_height":    LaunchConfiguration("network_height"),
            "color_width":       LaunchConfiguration("color_width"),
            "color_height":      LaunchConfiguration("color_height"),
        }],
    )

    return LaunchDescription(launch_args + [realsense, extrinsics_relay, roi_depth])
