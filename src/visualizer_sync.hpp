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
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <utility>

// Python message_filters matching for two streams: approximate matches remove
// only the chosen stamps; exact matches also discard older queued stamps.
template<class Left, class Right>
class VisualizerSync {
public:
  VisualizerSync(std::size_t queue_size, bool approximate, double slop)
  : queue_size_(queue_size),
    approximate_(approximate),
    slop_ns_(static_cast<int64_t>(slop * 1e9)) {}
  std::optional<std::pair<Left, Right>> left(int64_t stamp, Left message)
  {
    add(left_, stamp, message);
    return match(stamp, true);
  }
  std::optional<std::pair<Left, Right>> right(int64_t stamp, Right message)
  {
    add(right_, stamp, message);
    return match(stamp, false);
  }

private:
  template<class Queue, class Message>
  void add(Queue & queue, int64_t stamp, Message message)
  {
    const auto it = queue.find(stamp);
    if (it != queue.end()) {
      it->second.first = message;
    } else {
      queue[stamp] = {message, order_++};
    }
    while (queue.size() > queue_size_) {
      queue.erase(queue.begin());
    }
  }
  template<class Queue>
  int64_t nearest(const Queue & queue, int64_t stamp)
  {
    auto best = queue.begin();
    for (auto it = queue.begin(); it != queue.end(); ++it) {
      const auto distance = std::abs(it->first - stamp),
        current = std::abs(best->first - stamp);
      if (distance < current ||
        (distance == current && it->second.second < best->second.second))
      {
        best = it;
      }
    }
    return best->first;
  }
  std::optional<std::pair<Left, Right>> match(
    int64_t stamp,
    bool left_arrived)
  {
    if (left_.empty() || right_.empty()) {
      return std::nullopt;
    }
    int64_t left_stamp = stamp, right_stamp = stamp;
    if (approximate_) {
      if (left_arrived) {
        right_stamp = nearest(right_, stamp);
      } else {
        left_stamp = nearest(left_, stamp);
      }
      if (std::abs(left_stamp - right_stamp) >= slop_ns_) {
        return std::nullopt;
      }
    }
    const auto left = left_.find(left_stamp);
    const auto right = right_.find(right_stamp);
    if (left == left_.end() || right == right_.end()) {
      return std::nullopt;
    }
    const std::pair<Left, Right> result{left->second.first,
      right->second.first};
    if (approximate_) {
      left_.erase(left);
      right_.erase(right);
    } else {
      left_.erase(left_.begin(), left_.upper_bound(left_stamp));
      right_.erase(right_.begin(), right_.upper_bound(right_stamp));
    }
    return result;
  }
  std::size_t queue_size_;
  bool approximate_;
  int64_t slop_ns_;
  uint64_t order_ = 0;
  std::map<int64_t, std::pair<Left, uint64_t>> left_;
  std::map<int64_t, std::pair<Right, uint64_t>> right_;
};
