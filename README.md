# Realsense ROI Depth Rectifier

Computes the depth **and bearing angles** for YOLO-style detections
using an Intel RealSense camera, without running `rs2::align` on the full frame.
With armor keypoints it also solves each plate's pose and normal.

## Architecture

```
/detections_output (Detection2DArray)       input_mode: detections (default)
  or /cv/armor_keypoints (ArmorKeypointsArray)  input_mode: armor_keypoints
        │   network space, ALL detections
        ▼
  roi_depth_node   ← maps bbox/keypoints network->color, samples depth LUT,
                      deprojects bbox centre + 4 corners, or solves the
                      plate pose from keypoints, per detection
        │
        └──▶  /cv/panel_detections  (dji_serial_bridge/msg/PanelDetectionArray)
```

`roi_depth_node` is the driving node: the input array triggers each
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
| `/cv/panel_detections` | `dji_serial_bridge/msg/PanelDetectionArray` | One entry per detection with valid depth: 4 corners + center in the camera body frame (metres), depth, confidence, class_id, and the plate normal when `has_normal` |

Each driving input publishes exactly one array, including an empty array when no
detections survive or calibration or depth is missing or stale.

`header.frame_id` comes from the `output_frame_id` parameter (default
`camera`, matching the URDF's camera link). `header.stamp` matches the
driving input stamp, not the depth image's. Corner order is TL, TR, BR, BL.
Box path (`has_normal` false): bbox corners and centre at the single
sampled depth, a fronto-parallel planar panel; bbox rotation (`theta`)
isn't applied, since upstream boxes are axis-aligned. Pose path
(`has_normal` true): corners are the light-bar ends of the
[solved plate](#keypoint-plate-pose).

### Coordinate convention

Points and normals use [REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)
body axes (X forward, Y left, Z up), not its optical axes (X right, Y down,
Z forward): body = (opt.z, −opt.x, −opt.y), `opticalToBody` in
[`plate_pose.hpp`](src/plate_pose.hpp). Both paths deproject or solve in
optical with the live colour `camera_info`, then convert. A plate facing
the camera has normal ≈ (−1, 0, 0). `output_frame_id` names the URDF
`camera` link, not realsense-ros's optical frames.

### Keypoint plate pose

With `input_mode: armor_keypoints`, a detection whose four keypoint
confidences all reach `keypoint_min_confidence` is solved as a plate:

1. Keypoints (TL, TR, BR, BL light-bar ends) go network->colour with the
   [letterbox inverse](#network-space-to-colour-space).
2. `cv::solvePnP`-family `SOLVEPNP_IPPE` (planar; the bar rectangle is not
   square, so not `IPPE_SQUARE`) against the colour `camera_info` K and D.
   Model points are ±spacing/2 × ±length/2 from
   [`config/armor_geometry.yaml`](config/armor_geometry.yaml). Of the two
   IPPE solutions the lower reprojection error wins.
3. Range: the bbox depth sample (as the box path) and the PnP range are
   fused by inverse variance. Depth σ = `depth_noise_base_m` +
   `depth_noise_range_coeff`·z², the tracker's model; PnP σ =
   √2·z²·σpx / (f·spacing·cos view), σpx = `keypoint_px_std`. The PnP
   bearing is kept and the whole pose is scaled to the fused range. No
   valid depth, or depth more than 3σ from the PnP range (a sample past a
   narrow or oblique plate): PnP range alone.
4. `normal` = outward plate normal (toward the camera), converted to the
   [body frame](#coordinate-convention); `corners` = the model points under
   the fused pose; `center` = the fused plate centre.
5. `normal_yaw_std` is a pixel-noise model (bar-length perspective plus
   width foreshortening, floored at `normal_yaw_std_floor_rad`). When the
   other IPPE solution also reprojects within `max_reproj_err_px` and its
   yaw differs by more than 2σ, the flip is unresolved: position is kept
   and `has_normal` is false.

A solve that throws, lands behind the camera, has its normal facing away
or exceeds `max_reproj_err_px` falls back to the box path with
`has_normal` false; so do low-confidence keypoints and `has_keypoints`
false. The detection is never dropped for a bad solve.

**Flip ambiguity.** A small plane under perspective has two poses that
reproject almost equally, mirrored about the line of sight. With 1 px
keypoint noise on the placeholder 135×55 mm plate (f = 615 px, 15° cant,
`test_plate_pose`, 2000 trials), the lower-error pick is the wrong one
40–50% of the time at every range:

| Range | Yaw 0°: median yaw err | Yaw 30°: median | 30°, right solution picked | PnP range σ |
|---|---|---|---|---|
| 2 m | 13° | 11° | 5° | 0.07–0.09 m |
| 4 m | 18° | 20° | 8° | 0.25–0.34 m |
| 6 m | 24° | 26° | 11° | 0.53–0.68 m |

Withholding unresolved flips keeps them out of the tracker; widening σ
instead would let a flipped yaw pass its gate. `test_plate_pose` prints the
fraction of normals still published. Resolving the flip needs a prior this
node lacks, such as the 15° cant against gravity.

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
| `input_mode` | `detections` | `detections` (Detection2DArray) or `armor_keypoints` (ArmorKeypointsArray) |
| `keypoints_topic` | `/cv/armor_keypoints` | Driving input in `armor_keypoints` mode |
| `plate.light_bar_spacing_m` | `0.135` | **Placeholder** (old small armor): centre-to-centre light-bar distance; set in `config/armor_geometry.yaml` |
| `plate.light_bar_length_m` | `0.055` | **Placeholder** (old small armor): one light bar's length |
| `keypoint_min_confidence` | `0.5` | All 4 keypoints must reach this to attempt a pose |
| `keypoint_px_std` | `1.0` | Keypoint noise, network px, for the range and yaw σ models |
| `max_reproj_err_px` | `3.0` | RMS reprojection gate, colour px |
| `normal_yaw_std_floor_rad` | `0.03` | Lower bound on `normal_yaw_std` |
| `depth_noise_base_m`/`depth_noise_range_coeff` | `0.03`/`0.01` | Depth σ = base + coeff·z², for range fusion |

The two plate values come from the new dot-matrix panel's CAD, which
doesn't exist yet; [context](../docs/ARCC_2026_SENTRY_CONTEXT.md#armor-geometry-and-building-limits).
`roi_depth_launch.py` loads `config/armor_geometry.yaml`; any other launch
running this node in keypoint mode must load it too.

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
eviction and arrival-order ties), the stamp-difference tests, and
`test_plate_pose` (pose recovery, noise Monte Carlo, optical->body,
fusion weights, degenerate solves; OpenCV only, no rclcpp).
