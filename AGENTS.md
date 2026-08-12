# Realsense_ROI_Depth_Rectifier — agent notes

Computes depth **and bearing angles** for YOLO-style detections by sampling a
depth LUT per ROI, instead of running `rs2::align` on the full frame.
**Reference docs live in `README.md`** — architecture diagram, the published
topic, the REP-103 coordinate convention, and the `roi_depth_node` parameter
list. Read it before changing the output contract. This file is only the
operating contract.

**The ROS package name is `roi_depth_query`, not the directory name** —
`--packages-select Realsense_ROI_Depth_Rectifier` selects nothing.

Parent conventions in `../CLAUDE.md` apply, notably: **in-code comments under
10 lines**; longer prose goes to a `## Notes` subheading in `README.md`.

## Running anything

Never hand-roll `docker exec`. Use `../isaac_ros_common/scripts/dexec.sh`, the
only path with correct env parity (ROS_DOMAIN_ID, FastDDS profile, both
workspace installs, `-u admin` for GUI). Load the `isaac-ros-docker` skill
before your first container command.

```bash
# all paths below are relative to this package dir
../isaac_ros_common/scripts/dexec.sh -- colcon build --packages-select roi_depth_query
../isaac_ros_common/scripts/dexec.sh -d -- ros2 launch roi_depth_query roi_depth_launch.py
```

**Shadowed by `/workspaces/ros2_ws`** (`Dockerfile.thornbots`,
`RECLONE_DEPTH`). Once built locally, a `src/` edit is live under `dexec.sh`
but not in the user's terminal, which resolves to the image-baked clone.
Confirm with `../isaac_ros_common/scripts/dexec.sh -- ros2 pkg prefix roi_depth_query`. C++, so a
source change always needs a rebuild; `--symlink-install` won't help.

## Scope

- Consumes `/detections_output` from
  `../realsense-yolov8-nitros-bridge` and adds depth + bearing. Detection
  itself belongs upstream there; target selection/tracking and the aiming
  math belong to `../sentry_pkg`.
- **Bearings are REP-103** (x forward, y left, z up) — the bbox is in network
  space and gets scaled to color space here. Getting either convention wrong
  produces plausible-looking numbers aimed the wrong way, not an error.
- Part of the current CV priority — see `../SESSION_NOTES.md`.
- Its own git repo (`Thornbots/Realsense_ROI_Depth_Rectifier`).
