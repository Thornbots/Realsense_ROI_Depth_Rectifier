# ROI depth query

Follow [workspace rules](../AGENTS.md) and [CI](../docs/CI.md).
ROS package: `roi_depth_query`. Read [output contract](README.md#published-topic)
and [architecture](README.md#architecture) before changing behavior.

## Scope

- Own depth/bearing from upstream detections; detection belongs to
  `realsense-yolov8-nitros-bridge`, selection/tracking to `thornbots_pkg`.
- Points use REP 103 body axes, not optical; boxes arrive in network space.
  A wrong [axis](README.md#coordinate-convention) or
  [letterbox](README.md#network-space-to-colour-space) conversion yields
  plausible, misaimed points rather than errors.
- Keep the C++ diagnostic overlay's pixel-space score and timestamp matching;
  [README](README.md#launch) explains its difference from live selection.
- Diff received stamps with `absStampDiffS` in `src/stamp_diff.hpp`; constructing
  `rclcpp::Time` from them can throw after a wall-clock step.

## Open

Robot acceptance remains in [hardware status](../JAZZY_FLASH.md#hardware-status).
