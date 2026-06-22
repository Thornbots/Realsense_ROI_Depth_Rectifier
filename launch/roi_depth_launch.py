"""
launch/roi_depth_launch.py

Standalone test/dev launch for this package: starts the D435i driver
(align_depth disabled), the extrinsics relay (required so roi_depth_node
can build its colour->depth LUT), and roi_depth_node itself.

By default /roi (vision_msgs/Detection2D, COLOR image space) is expected
to come from somewhere else (e.g. a hand-published test message, or your
own node). Set use_detection_picker:=true to also launch
detection_picker_node, which republishes /detections_output
(Detection2DArray, NETWORK image space) as /roi.

This file does NOT include the YOLOv8/TensorRT inference chain or the
DJI serial bridge — see realsense_yolov8_nitros_bridge's
isaac_ros_yolov8_realsense.launch.py for the full production pipeline
(camera -> inference -> ROI -> depth -> serial bridge).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
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
            "use_detection_picker", default_value="false",
            description="Also launch detection_picker_node to turn "
                         "/detections_output (Detection2DArray, network space) "
                         "into /roi (Detection2D, color space). Leave false if "
                         "you publish /roi yourself."),
        DeclareLaunchArgument(
            "detections_topic", default_value="/detections_output"),
        DeclareLaunchArgument("roi_topic", default_value="/roi"),
        DeclareLaunchArgument("network_width", default_value="640"),
        DeclareLaunchArgument("network_height", default_value="640"),
        DeclareLaunchArgument("color_width", default_value="640"),
        DeclareLaunchArgument("color_height", default_value="480"),
        DeclareLaunchArgument("min_score", default_value="0.0"),
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
    # never builds its LUT, so it silently never publishes /roi_point.
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
            "depth_ns":        "/camera/depth",
            "color_ns":        "/camera/color",
            "depth_scale":     0.001,   # D435i Z16 default (mm -> m)
            "min_depth_m":     0.1,
            "max_depth_m":     10.0,
        }],
    )

    detection_picker = Node(
        package="roi_depth_query",
        executable="detection_picker_node_exe",
        name="detection_picker_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_detection_picker")),
        parameters=[{
            "detections_topic": LaunchConfiguration("detections_topic"),
            "roi_topic":        LaunchConfiguration("roi_topic"),
            "network_width":    LaunchConfiguration("network_width"),
            "network_height":   LaunchConfiguration("network_height"),
            "color_width":      LaunchConfiguration("color_width"),
            "color_height":     LaunchConfiguration("color_height"),
            "min_score":        LaunchConfiguration("min_score"),
        }],
    )

    return LaunchDescription(
        launch_args + [realsense, extrinsics_relay, roi_depth, detection_picker])
