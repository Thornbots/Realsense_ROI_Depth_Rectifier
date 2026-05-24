# Realsense ROI Depth Rectifier

Efficiently computes the depth **and bearing angles** for YOLO-style detections
using an Intel RealSense camera, without running `rs2::align` on the full frame.

## Architecture

```
/detections_output (Detection2DArray, network space)
        │
        ▼
 detection_picker_node   ← picks highest-confidence detection, scales bbox
        │
        ▼
   /roi (Detection2D, color image space)
        │
        ▼
  roi_depth_node   ← samples depth LUT, depropjects bbox centre
        │
        ├──▶  /roi_depth_m      (Float32)  mean depth, metres
        ├──▶  /roi_angle_h_rad  (Float32)  horizontal bearing, radians
        └──▶  /roi_angle_v_rad  (Float32)  vertical bearing,   radians
```

`extrinsics_relay_node` is a one-shot helper that forwards the
`/extrinsics/depth_to_color` topic into `roi_depth_node`'s parameter server
so the LUT can be built.

---

## Published topic

| Topic | Type | Description |
|---|---|---|
| `/roi_point` | `geometry_msgs/PointStamped` | 3-D position of the detected object in the camera body frame (metres) |

The `header.frame_id` is set by the `output_frame_id` parameter (default `camera_color_frame`).  The `header.stamp` matches the depth image timestamp.

### Coordinate convention — ROS REP-103

The point is expressed in the ROS REP-103 camera body frame: **X forward, Y left, Z up**.

Internally, `rs2_deproject_pixel_to_point` is called at the measured mean depth, producing an undistorted 3-D point in the librealsense optical frame (X right, Y down, Z forward), which is then remapped:

```
ros.x =  rs.z   (forward)
ros.y = -rs.x   (left)
ros.z = -rs.y   (up)
```

No external FOV parameters are required — all information is derived from the live `/camera/color/camera_info` stream.

---

## Parameters (`roi_depth_node`)

| Parameter | Default | Description |
|---|---|---|
| `depth_ns` | `/camera/depth` | Namespace for depth topics |
| `color_ns` | `/camera/color` | Namespace for color topics |
| `output_frame_id` | `camera_color_frame` | `frame_id` written into the published `PointStamped` |
| `depth_scale` | `0.001` | Depth unit → metres (D435i default) |
| `min_depth_m` | `0.1` | Reject depth samples closer than this |
| `max_depth_m` | `10.0` | Reject depth samples farther than this |
| `center_sample_fraction` | `0.25` | Inner fraction of bbox to sample for depth (0.25 → inner 6.25% of area) |

---

## Build

```bash
cd <your_ws>
colcon build --packages-select roi_depth_query
source install/setup.bash
```

## Launch

```bash
ros2 launch roi_depth_query roi_depth_launch.py
```
