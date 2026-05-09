// detection_roi_relay_node.cpp
//
// Bridges the YOLOv8 decoder output to the roi_depth_node ROI input.
//
// The YOLOv8 decoder (isaac_ros_yolov8) publishes:
//   /detections_output  — vision_msgs/Detection2DArray
//   Bounding box coordinates are in NETWORK image space (network_w × network_h,
//   typically 640×640).
//
// The roi_depth_node subscribes to:
//   /roi  — vision_msgs/Detection2D (singular)
//   Bounding box coordinates must be in COLOR image space (color_w × color_h,
//   typically 640×480).
//
// This node:
//   1. Subscribes to /detections_output
//   2. Picks the highest-confidence detection from the array
//   3. Scales bbox from network space → color image space
//   4. Republishes as a singular Detection2D on /roi
//
// Parameters:
//   detections_topic  (string, default "/detections_output")
//   roi_topic         (string, default "/roi")
//   network_width     (int,    default 640)  — TensorRT input width
//   network_height    (int,    default 640)  — TensorRT input height
//   color_width       (int,    default 640)  — color stream width
//   color_height      (int,    default 480)  — color stream height
//   min_score         (double, default 0.0)  — ignore detections below this score
//
// Note on coordinate spaces:
//   The dnn_image_encoder resizes (and letter-boxes or stretches) the color frame
//   from (color_w × color_h) to (network_w × network_h) before inference.
//   isaac_ros_dnn_image_encoder uses simple bilinear resize with NO letterboxing,
//   so the mapping is a simple scale:
//       x_color = x_network * (color_w / network_w)
//       y_color = y_network * (color_h / network_h)
//   For 640×480 color → 640×640 network: x scale=1.0, y scale=0.75.
//   With a square model and non-square input the y-axis is the critical one.

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"

namespace roi_depth_query
{

class DetectionRoiRelayNode : public rclcpp::Node
{
public:
    explicit DetectionRoiRelayNode(const rclcpp::NodeOptions & opts = rclcpp::NodeOptions())
    : Node("detection_roi_relay_node", opts)
    {
        detections_topic_ = declare_parameter<std::string>("detections_topic", "/detections_output");
        roi_topic_        = declare_parameter<std::string>("roi_topic",         "/roi");
        network_w_        = declare_parameter<int>("network_width",   640);
        network_h_        = declare_parameter<int>("network_height",  640);
        color_w_          = declare_parameter<int>("color_width",     640);
        color_h_          = declare_parameter<int>("color_height",    480);
        min_score_        = declare_parameter<double>("min_score",    0.0);

        scale_x_ = static_cast<double>(color_w_) / static_cast<double>(network_w_);
        scale_y_ = static_cast<double>(color_h_) / static_cast<double>(network_h_);

        sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            detections_topic_, 10,
            [this](vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) {
                onDetections(msg);
            });

        pub_ = create_publisher<vision_msgs::msg::Detection2D>(roi_topic_, 10);

        RCLCPP_INFO(get_logger(),
            "DetectionRoiRelayNode ready\n"
            "  %s (Detection2DArray) -> %s (Detection2D)\n"
            "  network %dx%d -> color %dx%d  (scale x=%.4f y=%.4f)",
            detections_topic_.c_str(), roi_topic_.c_str(),
            network_w_, network_h_, color_w_, color_h_,
            scale_x_, scale_y_);
    }

private:
    void onDetections(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg)
    {
        if (msg->detections.empty()) {
            return;
        }

        // Pick highest-confidence detection above min_score
        const vision_msgs::msg::Detection2D * best = nullptr;
        double best_score = min_score_;

        for (const auto & det : msg->detections) {
            double score = 0.0;
            for (const auto & hyp : det.results) {
                score = std::max(score, hyp.hypothesis.score);
            }
            if (score > best_score) {
                best_score = score;
                best = &det;
            }
        }

        if (!best) {
            RCLCPP_DEBUG(get_logger(),
                "No detection above min_score=%.3f (got %zu candidates)",
                min_score_, msg->detections.size());
            return;
        }

        // Scale bbox from network space → color image space
        vision_msgs::msg::Detection2D out;
        out.header = best->header;
        out.results = best->results;

        out.bbox.center.position.x = best->bbox.center.position.x * scale_x_;
        out.bbox.center.position.y = best->bbox.center.position.y * scale_y_;
        out.bbox.size_x = best->bbox.size_x * scale_x_;
        out.bbox.size_y = best->bbox.size_y * scale_y_;

        pub_->publish(out);

        RCLCPP_DEBUG(get_logger(),
            "Relay: best det score=%.3f  net bbox(cx=%.1f cy=%.1f w=%.1f h=%.1f)"
            " -> color bbox(cx=%.1f cy=%.1f w=%.1f h=%.1f)",
            best_score,
            best->bbox.center.position.x, best->bbox.center.position.y,
            best->bbox.size_x, best->bbox.size_y,
            out.bbox.center.position.x, out.bbox.center.position.y,
            out.bbox.size_x, out.bbox.size_y);
    }

    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr pub_;

    std::string detections_topic_, roi_topic_;
    int network_w_, network_h_, color_w_, color_h_;
    double min_score_;
    double scale_x_, scale_y_;
};

}  // namespace roi_depth_query

RCLCPP_COMPONENTS_REGISTER_NODE(roi_depth_query::DetectionRoiRelayNode)
