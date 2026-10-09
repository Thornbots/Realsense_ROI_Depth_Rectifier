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

#include "visualizer_sync.hpp"

namespace
{
TEST(VisualizerSync, ExactRequiresMatchingStampAndDropsOlderMessages) {
  VisualizerSync<int, std::string> sync(10, false, 0.05);
  EXPECT_FALSE(sync.left(100, 1));
  EXPECT_FALSE(sync.left(200, 2));
  ASSERT_EQ(sync.right(200, "3"), std::make_pair(2, std::string("3")));
  EXPECT_FALSE(sync.right(100, "4"));
}
TEST(VisualizerSync,
     ApproximateMatchesNearestAndRetainsOlderUnmatchedMessages) {
  VisualizerSync<int, int> sync(10, true, 0.00000005);
  EXPECT_FALSE(sync.left(100, 1));
  EXPECT_FALSE(sync.left(200, 2));
  ASSERT_EQ(sync.right(220, 3), std::make_pair(2, 3));
  ASSERT_EQ(sync.right(110, 4), std::make_pair(1, 4));
}
TEST(VisualizerSync, ApproximateSlopIsStrictAndTiesFollowArrivalOrder) {
  VisualizerSync<int, int> sync(10, true, 0.00000005);
  EXPECT_FALSE(sync.left(200, 1));
  EXPECT_FALSE(sync.right(250, 2));
  EXPECT_EQ(sync.right(240, 3), std::make_pair(1, 3));
  VisualizerSync<int, int> ties(10, true, 0.00000005);
  EXPECT_FALSE(ties.left(220, 4));
  EXPECT_FALSE(ties.left(180, 5));
  EXPECT_EQ(ties.right(200, 6), std::make_pair(4, 6));
}
TEST(VisualizerSync, OverflowEvictsMinimumStampAndDuplicatePreservesOrder) {
  VisualizerSync<int, int> sync(2, true, 0.00000001);
  EXPECT_FALSE(sync.left(200, 1));
  EXPECT_FALSE(sync.left(100, 2));
  EXPECT_FALSE(sync.left(300, 3));
  EXPECT_FALSE(sync.right(100, 4));
  EXPECT_FALSE(sync.left(200, 5));
  EXPECT_EQ(sync.right(200, 6), std::make_pair(5, 6));
}
}  // namespace
