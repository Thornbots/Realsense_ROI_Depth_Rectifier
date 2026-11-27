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

#ifndef DETECTION_LABEL_HPP_
#define DETECTION_LABEL_HPP_

#include <charconv>
#include <optional>
#include <string_view>

namespace roi_depth_query
{

inline std::optional<int> parseDetectionLabel(std::string_view label)
{
  if (label.empty() || label.front() < '0' || label.front() > '9') {
    return std::nullopt;
  }
  int id = 0;
  const auto end = label.data() + label.size();
  const auto result = std::from_chars(label.data(), end, id);
  if (result.ec != std::errc{} || result.ptr != end || id > 7) {
    return std::nullopt;
  }
  return id;
}

}  // namespace roi_depth_query

#endif  // DETECTION_LABEL_HPP_
