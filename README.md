# Realsense ROI Depth Rectifier

Computes the depth **and bearing angles** for YOLO-style detections
using an Intel RealSense camera, without running `rs2::align` on the full frame.

## Architecture

```
/detections_output (Detection2DArray, network space, ALL detections)
        │
        ▼
  roi_depth_node   ← maps bbox network->color, samples depth LUT,
                      deprojects bbox centre + 4 corners, per detection
        │
        └──▶  /cv/panel_detections  (dji_serial_bridge/msg/PanelDetectionArray)
```

`roi_depth_node` is the driving node: `/detections_output` triggers each
publish (depth is a cached resource, matched by stamp within
`depth_max_age_s`), so output rate/stamps track the detector 1:1 and a
stalled depth stream can't pair with a fresh detection. There is no
picking step here. Every detection with valid depth goes out in the
array. `thornbots_pkg`'s `target_selector` (downstream, post-depth) does
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

### Coordinate convention

Points use [REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)
body axes (X forward, Y left, Z up), not its optical axes. `deprojectToRos` in
[`roi_depth_node.cpp`](src/roi_depth_node.cpp) converts librealsense's optical
point at the mean depth using the live colour `camera_info`. `output_frame_id`
names the URDF `camera` link, not realsense-ros's optical frames.

### Network space to colour space

The DNN image encoder letterboxes the colour image into the network input:
one scale for both axes, then zero padding split evenly. For 640x480 into
640x640 the scale is 1 and there are 80 black rows above and below the
picture. Isaac ROS 3.2 (`ResizeNode` with `keep_aspect_ratio`, the
`dnn_image_encoder.launch.py` default) and 4.6 (`DnnImageEncoderNode`)
both do this. So `roi_depth_node` subtracts the padding and divides by the
scale:

```
color_x = (net_x - pad_x) * color_w / resized_w
color_y = (net_y - pad_y) * color_h / resized_h
```

Before 2026-09-26 the node stretched instead (`color_y = net_y * 0.75`),
which is right only at the image centre row. At the top and bottom of the
picture it was off by 60 px, and every box came out 25% too short.

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
| `network_width`/`network_height` | `640`/`640` | TensorRT input size, for undoing the letterbox |
| `color_width`/`color_height` | `640`/`480` | Color stream size, for undoing the letterbox |

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

For production, run the YOLO launch and `thornbots_pkg`'s `auto.launch.py`
using the [two-terminal recipe](../realsense-yolov8-nitros-bridge/README.md#full-robot-pipeline).
The YOLO launch supplies detections; `auto.launch.py` owns aiming and the
serial bridge. Stop the standalone launch above first: both open the camera.

For the diagnostic overlay, run `ros2 run roi_depth_query
detection_picker_visualizer`. It letterboxes `/color/image_raw` into network
space, synchronizes it with `/detections_output`, and publishes
`yolov8_processed_image` with confidence, centrality, priority and team factors.
It preserves the former picker's pixel-space score; the live selector uses 3D
grouping, so this is an approximation of that decision. Parameters live in
[the C++ node](src/detection_picker_visualizer.cpp).

`colcon test --packages-select roi_depth_query` includes the timestamp
synchronizer gtests (exact and approximate matching, strict slop, queue
eviction and arrival-order ties) and the existing stamp-difference tests.
