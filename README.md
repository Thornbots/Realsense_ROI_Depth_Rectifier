# Realsense ROI Depth Rectifier

Computes the depth **and bearing angles** for YOLO-style detections
using an Intel RealSense camera, without running `rs2::align` on the full frame.

## Architecture

```
/detections_output (Detection2DArray, network space, ALL detections)
        │
        ▼
  roi_depth_node   ← scales bbox network->color, samples depth LUT,
                      deprojects bbox centre + 4 corners, per detection
        │
        └──▶  /cv/panel_detections  (dji_serial_bridge/msg/PanelDetectionArray)
```

`roi_depth_node` is the driving node: `/detections_output` triggers each
publish (depth is a cached resource, matched by stamp within
`depth_max_age_s`), so output rate/stamps track the detector 1:1 and a
stalled depth stream can't pair with a fresh detection. There is no
picking step here. Every detection with valid depth goes out in the
array. `thornbots_pkg`'s `target_selector.py` (downstream, post-depth) does
team filtering, 3D robot grouping, and the per-frame panel pick, then
republishes the winner as a singular `PanelDetection` on
`/cv/panel_detection`.

`extrinsics_relay_node` is a one-shot helper that forwards the
`/extrinsics/depth_to_color` topic into `roi_depth_node`'s parameter server
so the LUT can be built.

## Published topic

| Topic | Type | Description |
|---|---|---|
| `/cv/panel_detections` | `dji_serial_bridge/msg/PanelDetectionArray` | One entry per detection with valid depth: 4 bbox corners + center deprojected to 3-D in the camera body frame (metres), plus depth, confidence, class_id |

`header.frame_id` comes from the `output_frame_id` parameter (default
`camera`, matching the URDF's camera link). `header.stamp` matches the
driving `/detections_output` stamp, not the depth image's. Corner order is
TL, TR, BR, BL, and all points assume a planar panel at the single sampled
depth. Bbox rotation (`theta`) isn't applied, since upstream YOLOv8 boxes
are axis-aligned.

### Coordinate convention: ROS REP-103

Each point is in the ROS REP-103 camera body frame: **X forward, Y left, Z
up**. Internally `rs2_deproject_pixel_to_point` runs at the measured mean
depth, producing an undistorted point in the librealsense optical frame (X
right, Y down, Z forward), which is then remapped:

```
ros.x =  rs.z   (forward)
ros.y = -rs.x   (left)
ros.z = -rs.y   (up)
```

No external FOV parameters are needed; everything comes from the live
`/camera/color/camera_info` stream.

## Parameters (`roi_depth_node`)

| Parameter | Default | Description |
|---|---|---|
| `depth_ns` | `/camera/depth` | Namespace for depth topics |
| `color_ns` | `/camera/color` | Namespace for color topics |
| `output_frame_id` | `camera` | `frame_id` written into the published `PanelDetectionArray` |
| `depth_scale` | `0.001` | Depth unit → metres (D435i default) |
| `min_depth_m` | `0.1` | Reject depth samples closer than this |
| `max_depth_m` | `10.0` | Reject depth samples farther than this |
| `center_sample_fraction` | `0.25` | Inner fraction of bbox to sample for depth (0.25 → inner 6.25% of area) |
| `depth_max_age_s` | `0.05` | Max `|detection_stamp - depth_stamp|` before a detection is dropped instead of paired with stale/future depth |
| `max_detections` | `16` | Cap on detections processed per `/detections_output` callback |
| `detections_topic` | `/detections_output` | Driving input (Detection2DArray, network space) |
| `network_width`/`network_height` | `640`/`640` | TensorRT input size, for bbox scaling |
| `color_width`/`color_height` | `640`/`480` | Color stream size, for bbox scaling |

## Build

```bash
cd <your_ws>
colcon build --packages-select roi_depth_query
source install/setup.bash
```

## Launch

```bash
# Standalone: camera + roi_depth_node, feed /detections_output yourself
ros2 launch roi_depth_query roi_depth_launch.py
```

For the full production pipeline (RealSense → YOLOv8/TensorRT →
roi_depth_node → `thornbots_pkg`'s `target_selector` → DJI serial bridge), use
`realsense_yolov8_nitros_bridge`'s `isaac_ros_yolov8_realsense.launch.py`
instead, which wires `/cv/panel_detections` through to `thornbots_pkg`.
