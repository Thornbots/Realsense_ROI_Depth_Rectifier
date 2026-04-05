"""
launch/roi_depth_launch.py

Starts the D435i driver (align_depth disabled) + the roi_depth_node.
Pass your ROI via /roi (vision_msgs/Detection2D) from another node.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rs_launch = os.path.join(
        get_package_share_directory("realsense2_camera"),
        "launch", "rs_launch.py"
    )

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
            "depth_module.depth_profile": "848x480x30",
            "rgb_camera.color_profile":   "848x480x30",
        }.items(),
    )

    roi_depth = Node(
        package="roi_depth_query",
        executable="roi_depth_node",
        name="roi_depth_node",
        output="screen",
        parameters=[{
            "depth_ns":        "/camera/camera/depth",
            "color_ns":        "/camera/camera/color",
            "extrinsics_topic": "/camera/camera/extrinsics/depth_to_color",
            "depth_scale":     0.001,   # D435i Z16 default
            "min_depth_m":     0.1,
            "max_depth_m":     10.0,
        }],
    )

    return LaunchDescription([realsense, roi_depth])
