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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "plate_pose.hpp"

using roi_depth_query::PlateGeometry;
using roi_depth_query::opticalToBody;

namespace
{
constexpr double kF = 615.0;  // D435 colour 640x480, approximately
constexpr double kGate = 3.0;  // max_reproj_err_px default
const PlateGeometry kGeom{0.135, 0.055};
const cv::Matx33d kK(kF, 0, 320, 0, kF, 240, 0, 0, 1);
const cv::Mat kNoDist = cv::Mat::zeros(1, 5, CV_64F);

cv::Vec3d outward(const cv::Matx33d & R) {return {-R(0, 2), -R(1, 2), -R(2, 2)};}
double deg(double r) {return r * 180.0 / M_PI;}
double rad(double d) {return d * M_PI / 180.0;}
double wrap(double a) {return std::atan2(std::sin(a), std::cos(a));}

// Plate at optical t, yawed by yaw (about optical -Y, i.e. body up) relative
// to facing the camera, outward normal tilted up by cant (the 15 deg rule).
cv::Matx33d plateRotation(double yaw, double cant)
{
  const double cy = std::cos(yaw), sy = std::sin(yaw), cc = std::cos(-cant), sc = std::sin(-cant);
  const cv::Matx33d Ry(cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  const cv::Matx33d Rx(1, 0, 0, 0, cc, -sc, 0, sc, cc);
  return Ry * Rx;
}

std::array<cv::Point2d, 4> project(const cv::Matx33d & R, const cv::Vec3d & t)
{
  const auto m = roi_depth_query::plateModelPoints(kGeom);
  std::array<cv::Point2d, 4> px;
  for (int i = 0; i < 4; ++i) {
    const cv::Vec3d p = R * cv::Vec3d(m[i].x, m[i].y, m[i].z) + t;
    px[i] = {kF * p[0] / p[2] + 320.0, kF * p[1] / p[2] + 240.0};
  }
  return px;
}

double bodyYaw(const cv::Vec3d & n_optical)
{
  const cv::Vec3d n = opticalToBody(n_optical);
  return std::atan2(n[1], n[0]);
}

struct Stats
{
  double rms_yaw_deg, median_yaw_deg, oracle_median_deg, flip_frac, rms_pos_m, range_std_m;
  double model_yaw_std_deg, published_frac, covered_frac;  // covered: |err| <= 3 * reported yaw std
  int ok;
};

Stats monteCarlo(double range, double yaw, double cant, double px_std, int n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, px_std);
  const cv::Vec3d t(0.3, 0.1, range);
  const cv::Matx33d R = plateRotation(yaw, cant);
  const double true_yaw = bodyYaw(outward(R));
  const auto clean = project(R, t);
  std::vector<double> yaw_err, oracle_err;
  double pos2 = 0.0, z_sum = 0.0, z2_sum = 0.0;
  int flips = 0, covered = 0, published = 0;
  for (int k = 0; k < n; ++k) {
    auto px = clean;
    for (auto & p : px) {
      p.x += noise(rng); p.y += noise(rng);
    }
    const auto pose = roi_depth_query::solvePlatePose(px, kGeom, kK, kNoDist, kGate);
    if (!pose.ok) {continue;}
    const double e = wrap(bodyYaw(pose.outwardNormalOptical()) - true_yaw);
    yaw_err.push_back(std::abs(e));
    // Flipped: the rejected IPPE solution was closer to the truth.
    const double e_alt = std::abs(wrap(bodyYaw(pose.alt_normal_optical) - true_yaw));
    if (e_alt < std::abs(e)) {++flips;}
    oracle_err.push_back(std::min(e_alt, std::abs(e)));
    // Normals the node would publish (unresolved flips withheld).
    if (!roi_depth_query::plateYawAmbiguous(pose, kGeom, kF, px_std, 0.03, kGate)) {
      ++published;
      const double sd = roi_depth_query::plateYawStd(pose, kGeom, kF, px_std, 0.03);
      if (std::abs(e) <= 3.0 * sd) {++covered;}
    }
    pos2 += std::pow(cv::norm(pose.t - t), 2);
    z_sum += pose.t[2];
    z2_sum += pose.t[2] * pose.t[2];
  }
  Stats s{};
  s.ok = static_cast<int>(yaw_err.size());
  double sq = 0.0;
  for (double e : yaw_err) {
    sq += e * e;
  }
  s.rms_yaw_deg = deg(std::sqrt(sq / s.ok));
  std::nth_element(yaw_err.begin(), yaw_err.begin() + s.ok / 2, yaw_err.end());
  s.median_yaw_deg = deg(yaw_err[s.ok / 2]);
  std::nth_element(oracle_err.begin(), oracle_err.begin() + s.ok / 2, oracle_err.end());
  s.oracle_median_deg = deg(oracle_err[s.ok / 2]);
  s.covered_frac = published ? static_cast<double>(covered) / published : 1.0;
  s.published_frac = static_cast<double>(published) / s.ok;
  s.flip_frac = static_cast<double>(flips) / s.ok;
  s.rms_pos_m = std::sqrt(pos2 / s.ok);
  const double zm = z_sum / s.ok;
  s.range_std_m = std::sqrt(std::max(0.0, z2_sum / s.ok - zm * zm));
  s.model_yaw_std_deg = deg(
    roi_depth_query::normalYawStd(t, outward(R), kGeom, kF, px_std, 0.0));
  return s;
}
}  // namespace

TEST(PlatePose, OpticalToBodyIsRep103) {
  const cv::Vec3d b = opticalToBody({1.0, 2.0, 3.0});  // right, down, forward
  EXPECT_DOUBLE_EQ(b[0], 3.0);   // forward
  EXPECT_DOUBLE_EQ(b[1], -1.0);  // left
  EXPECT_DOUBLE_EQ(b[2], -2.0);  // up
  // A plate facing the camera has its outward normal pointing back at it.
  const cv::Vec3d n = opticalToBody(outward(plateRotation(0.0, 0.0)));
  EXPECT_NEAR(n[0], -1.0, 1e-12);
  // The cant tilts the outward normal up in body frame.
  EXPECT_GT(opticalToBody(outward(plateRotation(0.0, rad(15))))[2], 0.2);
}

TEST(PlatePose, NoiseFreeRecoversPose) {
  for (double yaw_deg : {-50.0, -20.0, 0.0, 35.0}) {
    for (double range : {1.0, 3.0, 6.0}) {
      const cv::Vec3d t(-0.4, 0.2, range);
      const cv::Matx33d R = plateRotation(rad(yaw_deg), rad(15));
      const auto pose = roi_depth_query::solvePlatePose(project(R, t), kGeom, kK, kNoDist, 1.0);
      ASSERT_TRUE(pose.ok) << pose.reject;
      EXPECT_LT(cv::norm(pose.t - t), 1e-6);
      EXPECT_LT(cv::norm(pose.outwardNormalOptical() - outward(R)), 1e-6);
      EXPECT_LT(pose.reproj_err_px, 1e-6);
    }
  }
}

// 1 px keypoint noise, 15 deg cant. Prints the table quoted in the README.
TEST(PlatePose, NoisyYawAndPosition) {
  std::printf(
    "range yaw | rms_yaw med_yaw oracle_med flip | rms_pos z_std mc/model | "
    "yaw_std_model covered\n");
  for (double yaw_deg : {0.0, 30.0}) {
    for (double range : {2.0, 4.0, 6.0}) {
      const auto s = monteCarlo(range, rad(yaw_deg), rad(15), 1.0, 2000, 7);
      const cv::Vec3d t(0.3, 0.1, range);  // as in monteCarlo
      const double cos_view = std::abs(outward(plateRotation(rad(yaw_deg), rad(15))).dot(
          cv::normalize(t)));
      const double z_model =
        roi_depth_query::pnpRangeStd(range, kF, kGeom.bar_spacing_m, 1.0, cos_view);
      std::printf(
        "%3.0f m %3.0f | %5.1f   %5.1f   %5.1f      %4.2f | "
        "%5.3f   %5.3f/%5.3f   | %5.1f    %4.2f\n",
        range, yaw_deg, s.rms_yaw_deg, s.median_yaw_deg, s.oracle_median_deg, s.flip_frac,
        s.rms_pos_m, s.range_std_m, z_model, s.model_yaw_std_deg, s.covered_frac);
      std::printf("        published normals %4.2f\n", s.published_frac);
      EXPECT_GT(s.ok, 1900);
      // PnP range noise follows the model within 30%.
      EXPECT_NEAR(s.range_std_m / z_model, 1.0, 0.3);
      // On published normals the reported std covers the actual error.
      EXPECT_GT(s.covered_frac, 0.9);
    }
  }
}

TEST(PlatePose, FusionWeights) {
  const auto eq = roi_depth_query::fuseRange(4.0, 0.2, 4.4, 0.2);
  EXPECT_NEAR(eq.z, 4.2, 1e-12);
  EXPECT_NEAR(eq.w_depth, 0.5, 1e-12);
  EXPECT_NEAR(eq.std, 0.2 / std::sqrt(2.0), 1e-12);
  // Depth twice as noisy: weight 1/4 : 1 -> 0.2.
  const auto dn = roi_depth_query::fuseRange(4.0, 0.4, 5.0, 0.2);
  EXPECT_NEAR(dn.w_depth, 0.2, 1e-12);
  EXPECT_NEAR(dn.z, 4.8, 1e-12);
  // Models: depth grows as base + k z^2, PnP as z^2 / (f L).
  EXPECT_NEAR(roi_depth_query::depthRangeStd(4.0, 0.03, 0.01), 0.19, 1e-12);
  EXPECT_NEAR(
    roi_depth_query::pnpRangeStd(4.0, 615.0, 0.135, 1.0), std::sqrt(2.0) * 16.0 / 83.025, 1e-12);
  EXPECT_NEAR(
    roi_depth_query::pnpRangeStd(4.0, 615.0, 0.135, 1.0, 0.5),
    2.0 * roi_depth_query::pnpRangeStd(4.0, 615.0, 0.135, 1.0), 1e-12);
}

TEST(PlatePose, YawStdModel) {
  const cv::Vec3d t(0, 0, 4.0);
  const auto n30 = outward(plateRotation(rad(30), 0.0));
  const auto n0 = outward(plateRotation(0.0, 0.0));
  const double s30 = roi_depth_query::normalYawStd(t, n30, kGeom, kF, 1.0, 0.0);
  const double s0 = roi_depth_query::normalYawStd(t, n0, kGeom, kF, 1.0, 0.0);
  EXPECT_LT(s30, s0);  // frontal yaw is the worst determined
  EXPECT_NEAR(s30, 1.0 / (kF * kGeom.bar_spacing_m * 0.5 / 4.0), 0.02);
  EXPECT_DOUBLE_EQ(roi_depth_query::normalYawStd(t, n30, kGeom, kF, 1.0, 0.5), 0.5);
}

TEST(PlatePose, DegenerateRejected) {
  const cv::Vec3d t(0.0, 0.0, 3.0);
  const cv::Matx33d R = plateRotation(rad(20), rad(15));
  // Left/right swapped: the plate is seen from behind.
  auto px = project(R, t);
  std::swap(px[0], px[1]);
  std::swap(px[2], px[3]);
  auto pose = roi_depth_query::solvePlatePose(px, kGeom, kK, kNoDist, 2.0);
  EXPECT_FALSE(pose.ok);
  EXPECT_EQ(pose.reject, "normal faces away from camera");

  // One keypoint 25 px off: no plane pose reprojects within 2 px.
  px = project(R, t);
  px[2].x += 25.0;
  pose = roi_depth_query::solvePlatePose(px, kGeom, kK, kNoDist, 2.0);
  EXPECT_FALSE(pose.ok);
  EXPECT_EQ(pose.reject, "reprojection error too large");

  // All four on one line.
  px = {cv::Point2d(300, 240), cv::Point2d(310, 240), cv::Point2d(320, 240), cv::Point2d(330, 240)};
  EXPECT_FALSE(roi_depth_query::solvePlatePose(px, kGeom, kK, kNoDist, 2.0).ok);
}
