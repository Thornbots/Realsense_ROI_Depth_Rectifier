# Realsense_ROI_Depth_Rectifier: agent notes

Computes depth **and bearing angles** for YOLO-style detections by sampling a
depth LUT per ROI, instead of running `rs2::align` on the full frame.
**Reference docs live in `README.md`**: architecture diagram, the published
topic, the REP-103 coordinate convention, and the `roi_depth_node` parameter
list. Read it before changing the output contract. This file only covers the
operating contract.

**The ROS package name is `roi_depth_query`, not the directory name.**
`--packages-select Realsense_ROI_Depth_Rectifier` selects nothing.

**Shadowed by `/workspaces/ros2_ws`** (`Dockerfile.thornbots` copies this
directory in at build time). Once built locally, a `src/` edit is live under
`dexec.sh` but not in the user's terminal, which resolves to the image-baked
snapshot. Confirm with
`../isaac_ros_common/scripts/dexec.sh -- ros2 pkg prefix roi_depth_query`. C++,
so a source change always needs a rebuild; `--symlink-install` won't help.

## Scope

- Consumes `/detections_output` from `../realsense-yolov8-nitros-bridge` and
  adds depth + bearing. Detection itself belongs upstream there; target
  selection/tracking and the aiming math belong to `../thornbots_pkg`.
- Bearings are REP-103 (x forward, y left, z up). The bbox is in network space
  and gets scaled to color space here. Getting either convention wrong produces
  plausible-looking numbers aimed the wrong way, not an error.
- Its own git repo (`Thornbots/Realsense_ROI_Depth_Rectifier`).

## Open

- **Jazzy (`main`)** builds clean and passes its tests in the Isaac
  ROS 4.6 container. It targets realsense-ros 4.56+: node-private topics
  (`~/color/...`) and a latched extrinsics topic, which
  `extrinsics_relay_node` subscribes to TRANSIENT_LOCAL. Robot validation:
  [hardware status](../JAZZY_FLASH.md#hardware-status).

## Committing

This package is a submodule of `thornbots_workspace`, on branch `nightly`. Commit
and push here first, then bump this gitlink in `../` — one logical change, one
bump, never a gitlink pointing at an unpushed commit. Full rule in
`../CLAUDE.md` § Packages.

## Rules

- Never build an `rclcpp::Time` from a received stamp: it throws on negative
  sec and a wall-clock step can produce one. Diff stamps with
  `absStampDiffS` (`src/stamp_diff.hpp`).

## CI

GitHub CI runs on PRs targeting main/nightly and pushes to both branches;
manual runs are available. Shared lint is pinned to workspace `884bfe63ea4e` (tag `ci-tooling-884bfe6`). Existing diagnostics are recorded in
`.github/quality-baseline.json`; new diagnostics fail. Do not expand the
baseline to hide regressions. Syntax errors always fail.
Jazzy CI builds the portable stack and runs this package's registered tests.
