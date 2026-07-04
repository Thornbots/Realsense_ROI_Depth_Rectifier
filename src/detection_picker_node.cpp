// detection_picker_node.cpp
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
//   2. Subscribes to /ref_sys_status and tracks whether this robot is on blue team
//   3. Filters out allied-team detections based on team colour:
//        blue  team → exclude class IDs 0–3, keep 4–7
//        red   team → exclude class IDs 4–7, keep 0–3
//      (If no RefSysStatus has been received yet, all classes pass through with a warning.)
//   4. Ranks the surviving detections by a composite priority score (see below)
//      and picks the single best one.
//   5. Scales bbox from network space → color image space
//   6. Republishes as a singular Detection2D on /roi
//
// Priority score (higher = preferred target):
//      score = confidence
//            + center_weight        * centrality
//            + priority_class_bonus  (only if the detection's class is a
//                                     priority class)
//   where:
//     - confidence  is the detection's highest hypothesis score (0–1).
//     - centrality  is 1 at the image centre and 0 at the corners — it favours
//       whatever the robot is already aimed at (the middle of the field of view).
//     - priority classes are the higher-value targets to feed the gimbal first.
//       Default {2, 6}: the "3rd" target in each team group (0–3 / 4–7).
//   min_score still gates on raw confidence, so low-confidence noise never wins
//   purely on centrality or class bonus.
//
// Parameters:
//   detections_topic     (string,    default "/detections_output")
//   roi_topic            (string,    default "/roi")
//   ref_sys_topic        (string,    default "/ref_sys_status")
//   network_width        (int,       default 640)  — TensorRT input width
//   network_height       (int,       default 640)  — TensorRT input height
//   color_width          (int,       default 640)  — color stream width
//   color_height         (int,       default 480)  — color stream height
//   min_score            (double,    default 0.0)  — ignore detections below this score
//   center_weight        (double,    default 1.0)  — weight of centrality in score
//   priority_class_bonus (double,    default 0.5)  — score added for a priority class
//   priority_class_ids   (int array, default [2,6])— class IDs to prioritise
//   max_output_rate_hz   (double,    default 10.0) — cap on /roi publish rate;
//                                     extra detection frames are dropped. <=0
//                                     disables the cap (publish every frame).
//
// Class-ID layout (8-class model):
//   0–3  first  four classes  — excluded when on BLUE  team
//   4–7  second four classes  — excluded when on RED   team
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "dji_serial_bridge/msg/ref_sys_status.hpp"

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
        ref_sys_topic_    = declare_parameter<std::string>("ref_sys_topic",     "/ref_sys_status");
        network_w_        = declare_parameter<int>("network_width",   640);
        network_h_        = declare_parameter<int>("network_height",  640);
        color_w_          = declare_parameter<int>("color_width",     640);
        color_h_          = declare_parameter<int>("color_height",    480);
        min_score_        = declare_parameter<double>("min_score",    0.0);
        center_weight_       = declare_parameter<double>("center_weight",        1.0);
        priority_class_bonus_= declare_parameter<double>("priority_class_bonus", 0.5);
        priority_class_ids_  = declare_parameter<std::vector<int64_t>>(
                                   "priority_class_ids", std::vector<int64_t>{2, 6});
        max_output_rate_hz_  = declare_parameter<double>("max_output_rate_hz", 10.0);

        // Cap /roi publishing to at most max_output_rate_hz. <=0 disables the cap.
        min_output_period_ = (max_output_rate_hz_ > 0.0)
            ? rclcpp::Duration::from_seconds(1.0 / max_output_rate_hz_)
            : rclcpp::Duration(0, 0);

        scale_x_ = static_cast<double>(color_w_) / static_cast<double>(network_w_);
        scale_y_ = static_cast<double>(color_h_) / static_cast<double>(network_h_);

        // Half the network-image diagonal: the maximum possible distance from the
        // image centre, used to normalise centrality into [0, 1].
        half_diag_ = 0.5 * std::sqrt(
            static_cast<double>(network_w_) * network_w_ +
            static_cast<double>(network_h_) * network_h_);

        sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
            detections_topic_, 10,
            [this](vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) {
                onDetections(msg);
            });

        // Match dji_serial_bridge_node's publisher QoS exactly. It publishes
        // ~/ref_sys with rclcpp::SensorDataQoS() (best-effort, volatile,
        // KeepLast(5)). A reliable/transient-local subscriber would be QoS-
        // incompatible and silently never connect, so the team filter would
        // never learn the team colour. We accept that a late joiner may wait up
        // to one referee update (~5 Hz) for the first status instead.
        ref_sys_sub_ = create_subscription<dji_serial_bridge::msg::RefSysStatus>(
            ref_sys_topic_, rclcpp::SensorDataQoS(),
            [this](dji_serial_bridge::msg::RefSysStatus::ConstSharedPtr msg) {
                onRefSysStatus(msg);
            });

        pub_ = create_publisher<vision_msgs::msg::Detection2D>(roi_topic_, 10);

        std::string priority_ids_str;
        for (std::size_t i = 0; i < priority_class_ids_.size(); ++i) {
            priority_ids_str += std::to_string(priority_class_ids_[i]);
            if (i + 1 < priority_class_ids_.size()) priority_ids_str += ",";
        }

        RCLCPP_INFO(get_logger(),
            "DetectionRoiRelayNode ready\n"
            "  %s (Detection2DArray) -> %s (Detection2D)\n"
            "  network %dx%d -> color %dx%d  (scale x=%.4f y=%.4f)\n"
            "  scoring: confidence + %.2f*centrality + %.2f if class in {%s}  (min_score=%.3f)\n"
            "  max output rate: %.1f Hz\n"
            "  team colour source: %s  (waiting for first status msg...)",
            detections_topic_.c_str(), roi_topic_.c_str(),
            network_w_, network_h_, color_w_, color_h_,
            scale_x_, scale_y_,
            center_weight_, priority_class_bonus_, priority_ids_str.c_str(), min_score_,
            max_output_rate_hz_,
            ref_sys_topic_.c_str());
    }

private:
    // ── referee status callback ──────────────────────────────────────────────
    void onRefSysStatus(const dji_serial_bridge::msg::RefSysStatus::ConstSharedPtr & msg)
    {
        const bool was_known = is_blue_team_.has_value();
        const bool new_val   = msg->is_on_blue_team;

        if (!was_known || *is_blue_team_ != new_val) {
            RCLCPP_INFO(get_logger(),
                "Team colour %s: %s  (excluding class IDs %s)",
                was_known ? "changed to" : "set to",
                new_val ? "BLUE" : "RED",
                new_val ? "0-3" : "4-7");
        }

        is_blue_team_ = new_val;
    }

    // ── helper: numeric class ID from a detection's highest-score hypothesis ─
    //
    // isaac_ros_yolov8 serialises the YOLO integer class as a decimal string
    // in ObjectHypothesis::class_id.  Returns -1 on parse failure.
    static int topClassId(const vision_msgs::msg::Detection2D & det)
    {
        double best_score = -1.0;
        int    best_id    = -1;

        for (const auto & hyp : det.results) {
            if (hyp.hypothesis.score > best_score) {
                best_score = hyp.hypothesis.score;
                try {
                    best_id = std::stoi(hyp.hypothesis.class_id);
                } catch (const std::exception &) {
                    best_id = -1;
                }
            }
        }
        return best_id;
    }

    // ── helper: true if this class_id belongs to our own team ───────────────
    //
    // Class-ID layout (8-class model):
    //   0–3  → excluded when BLUE  (blue-team robots  — do not target allies)
    //   4–7  → excluded when RED   (red-team  robots  — do not target allies)
    //
    // Returns false (do not exclude) when team colour is not yet known so that
    // the pipeline keeps running until the first RefSysStatus arrives.
    bool isExcludedByTeam(int class_id) const
    {
        if (!is_blue_team_.has_value()) {
            return false;
        }
        if (*is_blue_team_) {
            return (class_id >= 0 && class_id <= 3);
        } else {
            return (class_id >= 4 && class_id <= 7);
        }
    }

    // ── helper: true if this class_id is a high-priority target ─────────────
    bool isPriorityClass(int class_id) const
    {
        return std::find(priority_class_ids_.begin(), priority_class_ids_.end(),
                         static_cast<int64_t>(class_id)) != priority_class_ids_.end();
    }

    // ── helper: centrality of a bbox, 1 at image centre → 0 at the corners ──
    //
    // Operates in network space (the bbox coordinates as received). Favours the
    // target the robot is already pointed at — the middle of the field of view.
    double centrality(const vision_msgs::msg::Detection2D & det) const
    {
        const double dx = det.bbox.center.position.x - 0.5 * network_w_;
        const double dy = det.bbox.center.position.y - 0.5 * network_h_;
        const double dist = std::sqrt(dx * dx + dy * dy);
        const double c = 1.0 - dist / half_diag_;
        return std::clamp(c, 0.0, 1.0);
    }

    // ── main detections callback ─────────────────────────────────────────────
    void onDetections(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg)
    {
        if (msg->detections.empty()) {
            return;
        }

        if (!is_blue_team_.has_value()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000 /*ms*/,
                "No RefSysStatus received yet on '%s' — team colour unknown, "
                "passing all detections through",
                ref_sys_topic_.c_str());
        }

        // Rank surviving detections by a composite priority score and keep the
        // best. A detection must clear min_score on raw *confidence* to be
        // eligible — centrality and the class bonus only re-order detections
        // that are already confident enough, they never resurrect noise.
        const vision_msgs::msg::Detection2D * best = nullptr;
        double best_priority = -1.0;   // composite score of the current best
        std::size_t n_filtered = 0;

        for (const auto & det : msg->detections) {
            const int class_id = topClassId(det);

            if (isExcludedByTeam(class_id)) {
                ++n_filtered;
                continue;
            }

            double confidence = 0.0;
            for (const auto & hyp : det.results) {
                confidence = std::max(confidence, hyp.hypothesis.score);
            }
            if (confidence < min_score_) {
                continue;
            }

            const double priority =
                confidence
                + center_weight_ * centrality(det)
                + (isPriorityClass(class_id) ? priority_class_bonus_ : 0.0);

            if (priority > best_priority) {
                best_priority = priority;
                best = &det;
            }
        }

        if (!best) {
            RCLCPP_DEBUG(get_logger(),
                "No detection passed filter "
                "(candidates=%zu  filtered_by_team=%zu  min_score=%.3f)",
                msg->detections.size(), n_filtered, min_score_);
            return;
        }

        // Rate-limit the output to at most max_output_rate_hz. The inference /
        // detection stream can arrive well above 10 Hz, but downstream (the
        // gimbal via the serial bridge) only needs ~10 Hz, so drop the extra
        // frames here instead of flooding /roi. Frames with no valid target
        // return above, so they never consume the rate budget.
        if (min_output_period_ > rclcpp::Duration(0, 0)) {
            const rclcpp::Time now = get_clock()->now();
            if (last_output_time_.has_value() &&
                (now - *last_output_time_) < min_output_period_) {
                return;
            }
            last_output_time_ = now;
        }

        // Scale bbox from network space → color image space
        vision_msgs::msg::Detection2D out;
        out.header  = best->header;
        out.results = best->results;

        out.bbox.center.position.x = best->bbox.center.position.x * scale_x_;
        out.bbox.center.position.y = best->bbox.center.position.y * scale_y_;
        out.bbox.size_x = best->bbox.size_x * scale_x_;
        out.bbox.size_y = best->bbox.size_y * scale_y_;

        pub_->publish(out);

        const int best_class = topClassId(*best);
        RCLCPP_DEBUG(get_logger(),
            "Relay: class=%d priority=%.3f (centrality=%.2f%s)  "
            "net(cx=%.1f cy=%.1f w=%.1f h=%.1f) -> color(cx=%.1f cy=%.1f w=%.1f h=%.1f)"
            "  [team-filtered %zu/%zu]",
            best_class, best_priority, centrality(*best),
            isPriorityClass(best_class) ? " +priority-class" : "",
            best->bbox.center.position.x, best->bbox.center.position.y,
            best->bbox.size_x, best->bbox.size_y,
            out.bbox.center.position.x, out.bbox.center.position.y,
            out.bbox.size_x, out.bbox.size_y,
            n_filtered, msg->detections.size());
    }

    // ── subscriptions / publisher ────────────────────────────────────────────
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr   sub_;
    rclcpp::Subscription<dji_serial_bridge::msg::RefSysStatus>::SharedPtr ref_sys_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2D>::SharedPtr           pub_;

    // ── parameters ───────────────────────────────────────────────────────────
    std::string detections_topic_, roi_topic_, ref_sys_topic_;
    int    network_w_, network_h_, color_w_, color_h_;
    double min_score_;
    double center_weight_, priority_class_bonus_;
    std::vector<int64_t> priority_class_ids_;
    double max_output_rate_hz_;
    double scale_x_, scale_y_;
    double half_diag_;

    // ── runtime state ────────────────────────────────────────────────────────
    // nullopt until the first RefSysStatus message arrives.
    std::optional<bool> is_blue_team_;
    // Output rate limiter: minimum wall-clock gap between /roi publishes, and
    // the time of the last publish (nullopt until the first output).
    rclcpp::Duration min_output_period_{0, 0};
    std::optional<rclcpp::Time> last_output_time_;
};

}  // namespace roi_depth_query

RCLCPP_COMPONENTS_REGISTER_NODE(roi_depth_query::DetectionRoiRelayNode)
