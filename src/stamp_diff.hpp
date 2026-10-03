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

// |a - b| of two message stamps in seconds, in plain int64 nanoseconds.
// rclcpp::Time(msg) throws on a negative sec ("cannot store a negative time
// point"), and a wall-clock step (1969 -> 2026 at boot) can put one in a
// stamp; the throw killed the whole composable container (run 38, 2026-10-03).
// Never construct an rclcpp::Time from a stamp that arrived over the wire.
#pragma once

#include <cmath>
#include <cstdint>

#include "builtin_interfaces/msg/time.hpp"

namespace roi_depth_query
{

inline int64_t stampNs(const builtin_interfaces::msg::Time & t)
{
  return static_cast<int64_t>(t.sec) * 1000000000LL + static_cast<int64_t>(t.nanosec);
}

inline double absStampDiffS(
  const builtin_interfaces::msg::Time & a, const builtin_interfaces::msg::Time & b)
{
  // int32 sec bounds the difference well inside int64.
  return std::abs(static_cast<double>(stampNs(a) - stampNs(b))) * 1e-9;
}

}  // namespace roi_depth_query
