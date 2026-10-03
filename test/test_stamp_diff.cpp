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

#include <limits>

#include "rclcpp/time.hpp"
#include "stamp_diff.hpp"

using builtin_interfaces::msg::Time;

static Time mk(int32_t s, uint32_t ns = 0)
{
  Time t;
  t.sec = s;
  t.nanosec = ns;
  return t;
}

TEST(StampDiff, SameClock)
{
  EXPECT_NEAR(roi_depth_query::absStampDiffS(mk(100, 500000000), mk(100, 450000000)), 0.05, 1e-9);
}

TEST(StampDiff, Epoch1969Versus2026)
{
  // Boot clock (37 s) against the NTP-stepped clock (2026-10-03 13:13 UTC).
  EXPECT_NEAR(roi_depth_query::absStampDiffS(mk(1790946835), mk(37)), 1790946798.0, 1e-3);
}

TEST(StampDiff, NegativeSecDoesNotThrow)
{
  // This is the input that made rclcpp::Time(msg) throw.
  EXPECT_THROW(rclcpp::Time(mk(-5), RCL_ROS_TIME), std::runtime_error);
  double d = 0;
  EXPECT_NO_THROW(d = roi_depth_query::absStampDiffS(mk(-5), mk(1790946835)));
  EXPECT_NEAR(d, 1790946840.0, 1e-3);
  EXPECT_NO_THROW(
    roi_depth_query::absStampDiffS(
      mk(std::numeric_limits<int32_t>::min()), mk(std::numeric_limits<int32_t>::max())));
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
