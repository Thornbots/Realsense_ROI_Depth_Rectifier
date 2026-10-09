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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <dji_serial_bridge/msg/ref_sys_status.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "visualizer_sync.hpp"

namespace
{
using Detections = vision_msgs::msg::Detection2DArray;
using Image = sensor_msgs::msg::Image;
int round_even(double value) {return static_cast<int>(std::nearbyint(value));}
int64_t stamp(const builtin_interfaces::msg::Time & time)
{
  return static_cast<int64_t>(time.sec) * 1000000000LL + time.nanosec;
}
std::string number(double value)
{
  char text[64];
  std::snprintf(text, sizeof(text), "%.2f", value);
  return text;
}
struct Factors
{
  int id = -1;
  double confidence = 0, centrality = 0, score = 0;
  bool priority = false, excluded = false, eligible = false;
};
}  // namespace

class DetectionPickerVisualizer : public rclcpp::Node {
public:
  DetectionPickerVisualizer()
  : Node("detection_picker_visualizer")
  {
    const auto detections =
      declare_parameter("detections_topic", "/detections_output");
    const auto image = declare_parameter("image_topic", "/color/image_raw");
    const auto output =
      declare_parameter("output_topic", "yolov8_processed_image");
    const auto referee =
      declare_parameter("ref_sys_topic", "/dji_serial_bridge/ref_sys");
    width_ = declare_parameter("network_width", 640);
    height_ = declare_parameter("network_height", 640);
    min_score_ = declare_parameter("min_score", 0.0);
    center_weight_ = declare_parameter("center_weight", 1.0);
    priority_bonus_ = declare_parameter("priority_class_bonus", 0.5);
    const auto ids =
      declare_parameter<std::vector<int64_t>>("priority_class_ids", {2, 6});
    priority_ids_.insert(ids.begin(), ids.end());
    blue_ = declare_parameter("is_blue_fallback", true);
    approximate_ = declare_parameter("use_approx_sync", false);
    slop_ = declare_parameter("sync_slop", 0.05);
    queue_size_ = declare_parameter("queue_size", 10);
    half_diag_ = 0.5 * std::hypot(width_, height_);
    sync_ = std::make_unique<
      VisualizerSync<Detections::ConstSharedPtr, Image::ConstSharedPtr>>(
        queue_size_, approximate_, slop_);
    pub_ = create_publisher<Image>(output, queue_size_);
    const auto qos = rclcpp::SensorDataQoS().keep_last(queue_size_);
    detections_sub_ = create_subscription<Detections>(
        detections, qos, [this](Detections::ConstSharedPtr msg) {
        const auto key = stamp(msg->header.stamp);
        const auto pair = sync_->left(key, msg);
        if (pair) {
          draw(*pair->first, pair->second);
        }
        });
    image_sub_ = create_subscription<Image>(
        image, qos, [this](Image::ConstSharedPtr msg) {
        const auto key = stamp(msg->header.stamp);
        const auto pair = sync_->right(key, msg);
        if (pair) {
          draw(*pair->first, pair->second);
        }
        });
    referee_sub_ = create_subscription<dji_serial_bridge::msg::RefSysStatus>(
        referee, rclcpp::SensorDataQoS(),
      [this](dji_serial_bridge::msg::RefSysStatus::ConstSharedPtr msg) {
        if (blue_ != msg->is_on_blue_team) {
          RCLCPP_INFO(get_logger(),
                        "Team colour set to %s (excluding class IDs %s)",
                        msg->is_on_blue_team ? "BLUE" : "RED",
                        msg->is_on_blue_team ? "0-3" : "4-7");
        }
        blue_ = msg->is_on_blue_team;
        });
    RCLCPP_INFO(get_logger(),
                "detection_picker_visualizer ready: %s + %s -> %s, network "
                "%dx%d, sync %s",
                detections.c_str(), image.c_str(), output.c_str(), width_,
                height_, approximate_ ? "approx" : "exact");
  }

private:
  Factors factors(const vision_msgs::msg::Detection2D & det) const
  {
    Factors result;
    double best = -1.0;
    for (const auto & hyp : det.results) {
      if (!(hyp.hypothesis.score > best)) {
        continue;
      }
      best = hyp.hypothesis.score;
      try {
        std::size_t parsed;
        const auto id = std::stoll(hyp.hypothesis.class_id, &parsed);
        if (hyp.hypothesis.class_id.find_first_not_of(" \t\n\r\v\f", parsed) !=
          std::string::npos ||
          id < std::numeric_limits<int>::min() ||
          id > std::numeric_limits<int>::max())
        {
          result.id = -1;
        } else {
          result.id = static_cast<int>(id);
        }
      } catch (const std::exception &) {
        result.id = -1;
      }
    }
    result.confidence = std::max(best, 0.0);
    result.excluded = blue_ ? result.id >= 0 && result.id <= 3 :
      result.id >= 4 && result.id <= 7;
    result.centrality = std::clamp(
        1.0 - std::hypot(det.bbox.center.position.x - width_ * 0.5,
                         det.bbox.center.position.y - height_ * 0.5) /
                  half_diag_,
        0.0, 1.0);
    result.priority = priority_ids_.count(result.id) != 0;
    result.score = result.confidence + center_weight_ * result.centrality +
      (result.priority ? priority_bonus_ : 0.0);
    result.eligible = !result.excluded && result.confidence >= min_score_;
    return result;
  }
  static void label(
    cv::Mat & image, const std::vector<std::string> & lines,
    cv::Point low, cv::Point high, const cv::Scalar & accent,
    double scale, int thickness)
  {
    int line_h = 0, block_w = 0;
    for (const auto & text : lines) {
      int baseline;
      const auto size = cv::getTextSize(text, 0, scale, thickness, &baseline);
      line_h = std::max(line_h, size.height);
      block_w = std::max(block_w, size.width);
    }
    line_h += 4;
    block_w += 6;
    const int block_h = line_h * static_cast<int>(lines.size()) + 2;
    const int x = std::max(0, std::min(low.x, image.cols - block_w));
    const int y = low.y - block_h >= 0 ? low.y - block_h : high.y;
    cv::rectangle(image, {x, y}, {x + block_w, y + block_h},
                  cv::Scalar(0, 0, 0), -1);
    cv::rectangle(image, {x, y}, {x + 3, y + block_h}, accent, -1);
    for (std::size_t i = 0; i < lines.size(); ++i) {
      cv::putText(image, lines[i],
        {x + 5, y + line_h * static_cast<int>(i + 1) - 3}, 0, scale,
                  cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }
  }
  void draw(const Detections & detections, Image::ConstSharedPtr image_msg)
  {
    auto converted = cv_bridge::toCvCopy(image_msg, image_msg->encoding);
    const double scale =
      std::min(static_cast<double>(width_) / converted->image.cols,
                 static_cast<double>(height_) / converted->image.rows);
    const int resized_w = static_cast<int>(converted->image.cols * scale);
    const int resized_h = static_cast<int>(converted->image.rows * scale);
    cv::Mat image;
    cv::resize(converted->image, image, {resized_w, resized_h}, 0, 0,
               cv::INTER_LINEAR);
    const int top = (height_ - resized_h) / 2, left = (width_ - resized_w) / 2;
    cv::copyMakeBorder(image, image, top, height_ - resized_h - top, left,
                       width_ - resized_w - left, cv::BORDER_CONSTANT, 0);
    const int lw = std::max(round_even((height_ + width_) / 2.0 * 0.003), 2);
    const int tf = std::max(lw - 1, 1);
    std::vector<Factors> infos;
    int best_idx = -1;
    double best_score = -1.0;
    for (const auto & det : detections.detections) {
      infos.push_back(factors(det));
      if (infos.back().eligible && infos.back().score > best_score) {
        best_score = infos.back().score;
        best_idx = static_cast<int>(infos.size()) - 1;
      }
    }
    const std::array<std::string, 8> names = {
      "blue_hero", "blue_std", "blue_stry", "blue_na",
      "red_hero", "red_std", "red_stry", "red_na"};
    for (std::size_t i = 0; i < infos.size(); ++i) {
      const auto & det = detections.detections[i];
      const auto & info = infos[i];
      const bool pick = static_cast<int>(i) == best_idx;
      const cv::Point low(
        round_even(det.bbox.center.position.x - det.bbox.size_x / 2),
        round_even(det.bbox.center.position.y - det.bbox.size_y / 2));
      const cv::Point high(
        round_even(det.bbox.center.position.x + det.bbox.size_x / 2),
        round_even(det.bbox.center.position.y + det.bbox.size_y / 2));
      const cv::Scalar color = pick ? cv::Scalar(0, 255, 0) :
        info.excluded ? cv::Scalar(255, 60, 60) :
        !info.eligible ? cv::Scalar(160, 160, 160) :
        cv::Scalar(255, 215, 0);
      cv::rectangle(image, low, high, color,
                    pick ? lw + 1 :
                      info.excluded || !info.eligible ? std::max(lw - 1, 1) :
                                                        lw);
      const auto name = info.id >= 0 && info.id < 8 ?
        names[info.id] :
        "id" + std::to_string(info.id);
      std::vector<std::string> lines = {
        "#" + std::to_string(i) + " " + name + (pick ? "  <PICK>" : ""),
        "conf " + number(info.confidence) + "  cen " +
        number(info.centrality),
        "score " + number(info.score) + (info.priority ? "  +prio" : "")};
      if (info.excluded) {
        lines.push_back("EXCLUDED: ally");
      } else if (!info.eligible) {
        lines.push_back("< min_score " + number(min_score_));
      }
      label(image, lines, low, high, color, lw / 3.0, tf);
    }
    pub_->publish(
        *cv_bridge::CvImage(image_msg->header, image_msg->encoding, image)
      .toImageMsg());
  }
  int width_, height_, queue_size_;
  double min_score_, center_weight_, priority_bonus_, half_diag_, slop_;
  bool blue_, approximate_;
  std::set<int64_t> priority_ids_;
  std::unique_ptr<
    VisualizerSync<Detections::ConstSharedPtr, Image::ConstSharedPtr>>
  sync_;
  rclcpp::Publisher<Image>::SharedPtr pub_;
  rclcpp::Subscription<Detections>::SharedPtr detections_sub_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_;
  rclcpp::Subscription<dji_serial_bridge::msg::RefSysStatus>::SharedPtr
    referee_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectionPickerVisualizer>());
  rclcpp::shutdown();
}
