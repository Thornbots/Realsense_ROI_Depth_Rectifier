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
#include <string>
#include <string_view>

#include "detection_label.hpp"

using roi_depth_query::parseDetectionLabel;

TEST(DetectionLabel, AcceptsSupportedDecimalClasses)
{
  for (int id = 0; id <= 7; ++id) {
    EXPECT_EQ(parseDetectionLabel(std::to_string(id)), id);
  }
  EXPECT_EQ(parseDetectionLabel("007"), 7);
}

TEST(DetectionLabel, RejectsMalformedLabels)
{
  for (const std::string_view label : {"", "red", "0red", "2.0", "1 2", " 2", "2 ", "+2", "-0"}) {
    EXPECT_FALSE(parseDetectionLabel(label).has_value()) << label;
  }
  EXPECT_FALSE(parseDetectionLabel(std::string_view("2\0x", 3)).has_value());
}

TEST(DetectionLabel, RejectsUnsupportedClasses)
{
  for (const std::string_view label : {"-1", "8", "18", "99999999999999999999999999"}) {
    EXPECT_FALSE(parseDetectionLabel(label).has_value()) << label;
  }
}
