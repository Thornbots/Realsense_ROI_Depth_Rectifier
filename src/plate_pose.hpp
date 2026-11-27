// Copyright 2026 Thornbots
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PLATE_POSE_HPP_
#define PLATE_POSE_HPP_

// Armor-plate pose from 4 light-bar keypoints: OpenCV only, no rclcpp, so
// gtest can drive it. Design: README.md#keypoint-plate-pose.

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <opencv2/core.hpp>

namespace roi_depth_query
{

// Outer light-bar ends, metres. Plate frame: x right, y down, z into the
// plate (away from a facing camera); origin at the plate centre.
struct PlateGeometry
{
  double bar_spacing_m;  // centre-to-centre distance between the two light bars
  double bar_length_m;   // length of one light bar
};

// TL, TR, BR, BL, matching ArmorKeypoints order.
std::array<cv::Point3d, 4> plateModelPoints(const PlateGeometry & g);

struct PlatePose
{
  bool ok{false};
  std::string reject;          // why ok is false; empty when ok
  cv::Matx33d R;               // plate -> optical rotation
  cv::Vec3d t;                 // plate centre, optical frame (X right, Y down, Z forward), m
  double reproj_err_px{0.0};   // RMS of the chosen IPPE solution
  double alt_reproj_err_px{0.0};  // RMS of the other IPPE solution (flip ambiguity)
  cv::Vec3d alt_normal_optical;   // outward normal of the other IPPE solution
  cv::Vec3d outwardNormalOptical() const {return {-R(0, 2), -R(1, 2), -R(2, 2)};}
};

// SOLVEPNP_IPPE against colour intrinsics K/dist (OpenCV model). Picks the
// lower-reprojection-error solution; rejects a normal facing away from the
// camera, a non-positive depth or an RMS error above max_reproj_err_px.
PlatePose solvePlatePose(
  const std::array<cv::Point2d, 4> & px, const PlateGeometry & g,
  const cv::Matx33d & K, const cv::Mat & dist, double max_reproj_err_px);

// Optical (X right, Y down, Z forward) -> REP 103 body (X fwd, Y left, Z up).
inline cv::Vec3d opticalToBody(const cv::Vec3d & o) {return {o[2], -o[0], -o[1]};}

// Depth-camera range noise, same model as thornbots_pkg target_tracker.
inline double depthRangeStd(double z, double base_m, double coeff)
{
  return base_m + coeff * z * z;
}

// PnP range noise from apparent bar spacing l = f L cos(view) / z:
// sigma_z = sqrt(2) z^2 sigma_px / (f L cos(view)); sqrt(2) fitted to the
// Monte Carlo in test_plate_pose. cos_view = |outward normal . unit ray|.
inline double pnpRangeStd(
  double z, double f_px, double bar_spacing_m, double px_std, double cos_view = 1.0)
{
  return std::sqrt(2.0) * z * z * px_std / (f_px * bar_spacing_m * std::max(cos_view, 0.2));
}

struct FusedRange
{
  double z;         // fused optical depth of the plate centre, m
  double std;       // its 1-sigma, m
  double w_depth;   // inverse-variance weight given to the depth camera, 0..1
};

FusedRange fuseRange(double z_depth, double std_depth, double z_pnp, double std_pnp);

// Normal yaw (about the camera up axis) 1-sigma from keypoint pixel noise:
// perspective bar-length difference plus width foreshortening, see README.
double normalYawStd(
  const cv::Vec3d & t_optical, const cv::Vec3d & n_optical, const PlateGeometry & g,
  double f_px, double px_std, double floor_rad);

// Reported normal yaw 1-sigma: the pixel-noise model at the chosen solution.
double plateYawStd(
  const PlatePose & pose, const PlateGeometry & g, double f_px, double px_std,
  double floor_rad);

// Unresolved flip: the other IPPE solution also fits within max_reproj_err_px
// and its yaw differs by more than 2 sigma. Callers then withhold the normal
// (a widened sigma would let a flipped yaw pass the tracker's gate).
bool plateYawAmbiguous(
  const PlatePose & pose, const PlateGeometry & g, double f_px, double px_std,
  double floor_rad, double max_reproj_err_px);

}  // namespace roi_depth_query

#endif  // PLATE_POSE_HPP_
