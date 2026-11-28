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

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <dji_serial_bridge/msg/armor_keypoints_array.hpp>
#include <dji_serial_bridge/msg/panel_detection_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "roi_depth_node.hpp"

using namespace std::chrono_literals;
using Panels = dji_serial_bridge::msg::PanelDetectionArray;

class FrameOutput : public ::testing::TestWithParam<std::string>
{
protected:
  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
      rclcpp::Parameter("input_mode", GetParam() == "exact" ? "armor_keypoints" : GetParam()),
      rclcpp::Parameter("exact_stamp_sync", GetParam() == "exact"),
      rclcpp::Parameter("extrinsics.rotation", std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1})});
    subject = roi_depth_query::make_roi_depth_node(options);
    peer = std::make_shared<rclcpp::Node>("frame_output_test");
    output = peer->create_subscription<Panels>(
      "/cv/panel_detections", 10, [this](Panels::ConstSharedPtr m) {received.push_back(*m);});
    boxes = peer->create_publisher<vision_msgs::msg::Detection2DArray>("/detections_output", 10);
    keypoints = peer->create_publisher<dji_serial_bridge::msg::ArmorKeypointsArray>(
      "/cv/armor_keypoints", 10);
    depth = peer->create_publisher<sensor_msgs::msg::Image>(
      "/camera/depth/image_rect_raw", rclcpp::QoS(10).reliable());
    depth_info = peer->create_publisher<sensor_msgs::msg::CameraInfo>(
      "/camera/depth/camera_info", rclcpp::QoS(10).reliable());
    color_info = peer->create_publisher<sensor_msgs::msg::CameraInfo>(
      "/camera/color/camera_info", rclcpp::QoS(10).reliable());
    executor.add_node(subject);
    executor.add_node(peer);
    spin(500ms);
  }

  void spin(std::chrono::milliseconds duration)
  {
    auto end = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < end) {
      executor.spin_some();
      std::this_thread::sleep_for(2ms);
    }
  }

  void send(int sec, bool detection, bool valid_box = true)
  {
    if (GetParam() == "detections") {
      vision_msgs::msg::Detection2DArray msg;
      msg.header.stamp.sec = sec;
      if (detection) {
        vision_msgs::msg::Detection2D d;
        d.bbox.center.position.x = valid_box ? 320 : -1000;
        d.bbox.center.position.y = 320;
        d.bbox.size_x = d.bbox.size_y = 40;
        msg.detections.push_back(d);
      }
      boxes->publish(msg);
    } else {
      dji_serial_bridge::msg::ArmorKeypointsArray msg;
      msg.header.stamp.sec = sec;
      if (detection) {
        dji_serial_bridge::msg::ArmorKeypoints d;
        d.center_x = valid_box ? 320 : -1000;
        d.center_y = 320;
        d.size_x = d.size_y = 40;
        msg.detections.push_back(d);
      }
      keypoints->publish(msg);
    }
    spin(150ms);
  }

  void calibrate()
  {
    sensor_msgs::msg::CameraInfo info;
    info.width = 640; info.height = 480;
    info.k = {600, 0, 320, 0, 600, 240, 0, 0, 1};
    info.distortion_model = "plumb_bob";
    depth_info->publish(info);
    color_info->publish(info);
    spin(400ms);
  }

  void send_depth(int sec)
  {
    sensor_msgs::msg::Image msg;
    msg.header.stamp.sec = sec;
    msg.width = 640; msg.height = 480;
    msg.encoding = "16UC1"; msg.step = 1280;
    msg.data.resize(msg.step * msg.height);
    for (size_t i = 0; i < msg.data.size(); i += 2) {
      msg.data[i] = 0xe8; msg.data[i + 1] = 0x03;
    }
    depth->publish(msg);
    spin(100ms);
  }

  rclcpp::executors::SingleThreadedExecutor executor;
  rclcpp::Node::SharedPtr subject;
  rclcpp::Node::SharedPtr peer;
  rclcpp::Subscription<Panels>::SharedPtr output;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr boxes;
  rclcpp::Publisher<dji_serial_bridge::msg::ArmorKeypointsArray>::SharedPtr keypoints;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info, color_info;
  std::vector<Panels> received;
};

TEST_P(FrameOutput, PublishesOncePerInput)
{
  if (GetParam() == "exact") {GTEST_SKIP();}
  send(1, true);  // No calibration.
  calibrate();
  send(2, true);  // No depth.
  send_depth(3);
  send(4, true);  // Stale depth.
  send(3, false);  // No detections.
  send(3, true, false);  // Detection outside the image.
  send(3, true);  // Valid detection.
  ASSERT_EQ(received.size(), 6u);
  const int stamps[] = {1, 2, 4, 3, 3, 3};
  for (size_t i = 0; i < received.size(); ++i) {
    EXPECT_EQ(received[i].header.stamp.sec, stamps[i]);
    EXPECT_EQ(received[i].header.frame_id, "camera");
    EXPECT_EQ(received[i].detections.size(), i == 5 ? 1u : 0u);
  }
}

TEST_P(FrameOutput, ExactSyncWaitsAndPreservesOrder)
{
  if (GetParam() != "exact") {GTEST_SKIP();}
  send(1, true);
  send_depth(2);
  send(2, true);
  send_depth(1);
  EXPECT_TRUE(received.empty());
  calibrate();
  ASSERT_EQ(received.size(), 2u);
  EXPECT_EQ(received[0].header.stamp.sec, 1);
  EXPECT_EQ(received[1].header.stamp.sec, 2);
  EXPECT_EQ(received[0].detections.size(), 1u);
  EXPECT_EQ(received[1].detections.size(), 1u);
  send_depth(3);
  send(3, true);
  ASSERT_EQ(received.size(), 3u);
  EXPECT_EQ(received[2].header.stamp.sec, 3);
  EXPECT_EQ(received[2].detections.size(), 1u);
}

INSTANTIATE_TEST_SUITE_P(Modes, FrameOutput,
  ::testing::Values("detections", "armor_keypoints", "exact"));

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
