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

#include "plate_pose.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace roi_depth_query
{

std::array<cv::Point3d, 4> plateModelPoints(const PlateGeometry & g)
{
  const double hw = g.bar_spacing_m / 2.0;
  const double hh = g.bar_length_m / 2.0;
  return {cv::Point3d(-hw, -hh, 0), cv::Point3d(hw, -hh, 0),
    cv::Point3d(hw, hh, 0), cv::Point3d(-hw, hh, 0)};
}

PlatePose solvePlatePose(
  const std::array<cv::Point2d, 4> & px, const PlateGeometry & g,
  const cv::Matx33d & K, const cv::Mat & dist, double max_reproj_err_px)
{
  PlatePose out;
  const auto model = plateModelPoints(g);
  const std::vector<cv::Point3d> obj(model.begin(), model.end());
  const std::vector<cv::Point2d> img(px.begin(), px.end());

  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<double> errs;
  try {
    cv::solvePnPGeneric(
      obj, img, K, dist, rvecs, tvecs, false, cv::SOLVEPNP_IPPE,
      cv::noArray(), cv::noArray(), errs);
  } catch (const cv::Exception &) {
    out.reject = "solvePnP threw";
    return out;
  }
  if (rvecs.empty() || errs.size() != rvecs.size()) {
    out.reject = "no IPPE solution";
    return out;
  }
  // OpenCV sorts IPPE solutions by error already; do not rely on it.
  std::size_t best = 0;
  for (std::size_t i = 1; i < errs.size(); ++i) {
    if (errs[i] < errs[best]) {best = i;}
  }
  cv::Mat R;
  cv::Rodrigues(rvecs[best], R);
  out.R = cv::Matx33d(R);
  out.t = cv::Vec3d(tvecs[best].reshape(1, 3));
  out.reproj_err_px = errs[best];
  const std::size_t alt = errs.size() > 1 ? 1 - best : best;
  out.alt_reproj_err_px = errs[alt];
  cv::Mat R_alt;
  cv::Rodrigues(rvecs[alt], R_alt);
  out.alt_normal_optical = -cv::Vec3d(R_alt.col(2));

  if (!std::isfinite(out.t[2]) || out.t[2] <= 0.0) {
    out.reject = "plate behind camera";
  } else if (out.outwardNormalOptical().dot(out.t) >= 0.0) {
    out.reject = "normal faces away from camera";
  } else if (!(out.reproj_err_px <= max_reproj_err_px)) {
    out.reject = "reprojection error too large";
  } else {
    out.ok = true;
  }
  return out;
}

FusedRange fuseRange(double z_depth, double std_depth, double z_pnp, double std_pnp)
{
  const double wd = 1.0 / (std_depth * std_depth);
  const double wp = 1.0 / (std_pnp * std_pnp);
  return {(wd * z_depth + wp * z_pnp) / (wd + wp), std::sqrt(1.0 / (wd + wp)), wd / (wd + wp)};
}

double normalYawStd(
  const cv::Vec3d & t_optical, const cv::Vec3d & n_optical, const PlateGeometry & g,
  double f_px, double px_std, double floor_rad)
{
  // theta: plate yaw relative to the line of sight, in the body horizontal plane.
  const cv::Vec3d ray = opticalToBody(t_optical);
  const cv::Vec3d n = opticalToBody(n_optical);
  const double theta = std::atan2(n[1], n[0]) - std::atan2(-ray[1], -ray[0]);
  const double z = cv::norm(t_optical);
  const double W = g.bar_spacing_m, H = g.bar_length_m;
  // d(bar length difference)/dtheta, var 4 sigma^2; d(width)/dtheta, var sigma^2.
  const double d_persp = f_px * H * W * std::cos(theta) / (z * z);
  const double d_width = f_px * W * std::sin(theta) / z;
  const double s2 = px_std * px_std;
  const double info = d_persp * d_persp / (4.0 * s2) + d_width * d_width / s2;
  const double sd = info > 0.0 ? 1.0 / std::sqrt(info) : M_PI;
  return std::clamp(sd, floor_rad, M_PI);
}

double plateYawStd(
  const PlatePose & pose, const PlateGeometry & g, double f_px, double px_std,
  double floor_rad)
{
  return normalYawStd(pose.t, pose.outwardNormalOptical(), g, f_px, px_std, floor_rad);
}

bool plateYawAmbiguous(
  const PlatePose & pose, const PlateGeometry & g, double f_px, double px_std,
  double floor_rad, double max_reproj_err_px)
{
  if (pose.alt_reproj_err_px > max_reproj_err_px) {return false;}
  const cv::Vec3d a = opticalToBody(pose.alt_normal_optical);
  const cv::Vec3d b = opticalToBody(pose.outwardNormalOptical());
  const double gap = std::atan2(a[1], a[0]) - std::atan2(b[1], b[0]);
  return std::abs(std::atan2(std::sin(gap), std::cos(gap))) >
         2.0 * plateYawStd(pose, g, f_px, px_std, floor_rad);
}

}  // namespace roi_depth_query
