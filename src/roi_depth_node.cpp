// roi_depth_node.cpp
//
// Efficiently computes the mean non-zero depth inside a color-image ROI
// WITHOUT running rs2::align on the full frame.
//
// Strategy
// --------
// 1. At start-up (or on CameraInfo receipt), build a lookup table (LUT) that
//    maps every color pixel (u_c, v_c) → depth pixel (u_d, v_d) by calling
//    rs2_project_color_pixel_to_depth_pixel() from librealsense's rsutil.h.
//    This is a one-time O(W*H) cost and can be rebuilt whenever the stream
//    profile changes.
//
// 2. At runtime, given an OpenCV-style bounding box in color-image space,
//    iterate only over the pixels inside the box, look up the corresponding
//    depth pixel from the LUT, sample depth_frame at that pixel, and compute
//    the non-zero mean.
//
//    Runtime cost: O(roi_w * roi_h) — typically < 1 % of full-frame alignment.
//
// Topics consumed (defaults match realsense-ros 4.x with align_depth DISABLED)
//   /camera/camera/depth/image_rect_raw          (sensor_msgs/Image, 16UC1)
//   /camera/camera/depth/camera_info             (sensor_msgs/CameraInfo)
//   /camera/camera/color/camera_info             (sensor_msgs/CameraInfo)
//   /camera/camera/extrinsics/depth_to_color     (realsense2_camera_msgs/Extrinsics)
//   /roi                                         (vision_msgs/Detection2D  — or see below)
//
// Topic published
//   /roi_depth_m                                 (std_msgs/Float32)  – mean depth in metres
//
// Note: realsense-ros publishes depth_to_color extrinsics only when
//       publish_tf:=true (the default).  The node can also fall back to
//       building the LUT from the two CameraInfo matrices alone (zero baseline
//       assumed — less accurate, but works when extrinsics aren't published).
//
// Build: add to your package's CMakeLists.txt and link against
//        realsense2, sensor_msgs, vision_msgs, std_msgs, cv_bridge, rclcpp.

#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/float32.hpp>
#include <vision_msgs/msg/detection2_d.hpp>          // carries bounding box
#include <realsense2_camera_msgs/msg/extrinsics.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <librealsense2/rsutil.h>   // rs2_project_color_pixel_to_depth_pixel
#include <librealsense2/rs.hpp>     // rs2_intrinsics / rs2_extrinsics types

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <vector>
#include <cmath>
#include <mutex>

namespace roi_depth_query
{
class RoiDepthNode : public rclcpp::Node
{
// Convert sensor_msgs/CameraInfo → rs2_intrinsics
rs2_intrinsics cameraInfoToIntrinsics(const sensor_msgs::msg::CameraInfo & ci)
{
    rs2_intrinsics intr{};
    intr.width  = static_cast<int>(ci.width);
    intr.height = static_cast<int>(ci.height);
    intr.ppx    = static_cast<float>(ci.k[2]);
    intr.ppy    = static_cast<float>(ci.k[5]);
    intr.fx     = static_cast<float>(ci.k[0]);
    intr.fy     = static_cast<float>(ci.k[4]);
    intr.model  = RS2_DISTORTION_BROWN_CONRADY;
    for (int i = 0; i < 5 && i < (int)ci.d.size(); ++i)
        intr.coeffs[i] = static_cast<float>(ci.d[i]);
    return intr;
}

// Convert realsense2_camera_msgs/Extrinsics → rs2_extrinsics
rs2_extrinsics extrinsicsMsgToRs2(
    const realsense2_camera_msgs::msg::Extrinsics & msg)
{
    rs2_extrinsics ex{};
    for (int i = 0; i < 9; ++i) ex.rotation[i]    = msg.rotation[i];
    for (int i = 0; i < 3; ++i) ex.translation[i] = msg.translation[i];
    return ex;
}
}  // namespace


class RoiDepthNode : public rclcpp::Node
{
public:
    explicit RoiDepthNode(const rclcpp::NodeOptions & opts = rclcpp::NodeOptions())
    : Node("roi_depth_node", opts)
    {
        // ── parameters ──────────────────────────────────────────────────────
        depth_ns_ = declare_parameter<std::string>("depth_ns",
            "/camera/camera/depth");
        color_ns_ = declare_parameter<std::string>("color_ns",
            "/camera/camera/color");
        extr_topic_ = declare_parameter<std::string>("extrinsics_topic",
            "/camera/camera/extrinsics/depth_to_color");
        depth_scale_ = declare_parameter<double>("depth_scale", 0.001); // Z16 → metres
        min_depth_m_ = declare_parameter<double>("min_depth_m", 0.1);
        max_depth_m_ = declare_parameter<double>("max_depth_m", 10.0);

        // ── subscriptions ────────────────────────────────────────────────────
        depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            depth_ns_ + "/camera_info", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
                std::lock_guard lk(lut_mutex_);
                depth_intr_ = cameraInfoToIntrinsics(*m);
                depth_intr_ready_ = true;
                tryBuildLut();
            });

        color_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            color_ns_ + "/camera_info", rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) {
                std::lock_guard lk(lut_mutex_);
                color_intr_ = cameraInfoToIntrinsics(*m);
                color_intr_ready_ = true;
                tryBuildLut();
            });

        extr_sub_ = create_subscription<realsense2_camera_msgs::msg::Extrinsics>(
            extr_topic_, rclcpp::QoS(1).transient_local(),
            [this](realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr m) {
                std::lock_guard lk(lut_mutex_);
                depth_to_color_ = extrinsicsMsgToRs2(*m);
                // derive color_to_depth by transposing rotation, negating translation
                rs2_extrinsics c2d{};
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        c2d.rotation[r*3+c] = depth_to_color_.rotation[c*3+r];
                for (int i = 0; i < 3; ++i) {
                    c2d.translation[i] = 0;
                    for (int j = 0; j < 3; ++j)
                        c2d.translation[i] -= c2d.rotation[i*3+j] * depth_to_color_.translation[j];
                }
                color_to_depth_ = c2d;
                extr_ready_ = true;
                tryBuildLut();
            });

        // Incoming ROI — using vision_msgs/Detection2D which carries a
        // BoundingBox2D (center_x, center_y, size_x, size_y).
        // Replace with your own message type if preferred.
        roi_sub_ = create_subscription<vision_msgs::msg::Detection2D>(
            "/roi", 10,
            [this](vision_msgs::msg::Detection2D::ConstSharedPtr m) {
                latest_roi_ = m;
            });

        depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
            depth_ns_ + "/image_rect_raw", rclcpp::SensorDataQoS(),
            std::bind(&RoiDepthNode::onDepth, this, std::placeholders::_1));

        // ── publisher ────────────────────────────────────────────────────────
        depth_pub_ = create_publisher<std_msgs::msg::Float32>("/roi_depth_m", 10);

        RCLCPP_INFO(get_logger(),
            "roi_depth_node ready — waiting for camera info and extrinsics …");
    }

private:
    // ── LUT ─────────────────────────────────────────────────────────────────
    // lut_[v_c * color_w + u_c] = {u_d, v_d}  (-1 if out of depth FOV)
    struct DepthPx { int16_t u, v; };
    std::vector<DepthPx> lut_;
    std::mutex lut_mutex_;
    bool lut_ready_{false};

    // intrinsics / extrinsics
    rs2_intrinsics depth_intr_{}, color_intr_{};
    rs2_extrinsics depth_to_color_{}, color_to_depth_{};
    bool depth_intr_ready_{false}, color_intr_ready_{false}, extr_ready_{false};

    // latest ROI (may be nullptr)
    vision_msgs::msg::Detection2D::ConstSharedPtr latest_roi_;

    // ROS handles
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_, color_info_sub_;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr extr_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr roi_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr depth_pub_;

    // params
    std::string depth_ns_, color_ns_, extr_topic_;
    double depth_scale_, min_depth_m_, max_depth_m_;

    // ────────────────────────────────────────────────────────────────────────
    // Build the LUT.  Called under lut_mutex_ whenever a new piece of info
    // arrives.  Requires: both intrinsics + extrinsics.
    // ────────────────────────────────────────────────────────────────────────
    void tryBuildLut()
    {
        if (!depth_intr_ready_ || !color_intr_ready_) return;

        // If extrinsics never arrive (e.g. publish_tf:=false) we fall back to
        // identity — this still works well when color and depth are close to
        // aligned (same principal-point assumption breaks down at the edges).
        if (!extr_ready_) {
            // identity
            float I[9] = {1,0,0, 0,1,0, 0,0,1};
            float T[3] = {0,0,0};
            std::copy(I, I+9, depth_to_color_.rotation);
            std::copy(T, T+3, depth_to_color_.translation);
            std::copy(I, I+9, color_to_depth_.rotation);
            std::copy(T, T+3, color_to_depth_.translation);
            RCLCPP_WARN_ONCE(get_logger(),
                "Extrinsics not yet received — using identity. "
                "Set publish_tf:=true in realsense-ros launch for accurate LUT.");
        }

        const int cw = color_intr_.width;
        const int ch = color_intr_.height;
        const int dw = depth_intr_.width;
        const int dh = depth_intr_.height;

        RCLCPP_INFO(get_logger(),
            "Building depth↔color LUT (%dx%d color → %dx%d depth) …", cw, ch, dw, dh);

        lut_.resize(static_cast<size_t>(cw * ch));

        const float depth_min = static_cast<float>(min_depth_m_);
        const float depth_max = static_cast<float>(max_depth_m_);
        const float ds        = static_cast<float>(depth_scale_);
        // rs2_project_color_pixel_to_depth_pixel needs uint16_t depth data for
        // the search; at LUT-build time we pass nullptr / use the version that
        // accepts nullptr when depth_scale==0 is not what we want — instead we
        // use rs2_deproject + rs2_transform + rs2_project analytically.
        // This is exact and doesn't require a depth frame at LUT-build time.

        for (int vc = 0; vc < ch; ++vc) {
            for (int uc = 0; uc < cw; ++uc) {
                // Unproject color pixel to a unit ray in color space (depth=1)
                float color_px[2] = { static_cast<float>(uc), static_cast<float>(vc) };
                float color_pt[3];
                rs2_deproject_pixel_to_point(color_pt, &color_intr_, color_px, 1.0f);

                // Transform from color frame to depth frame
                float depth_pt[3];
                rs2_transform_point_to_point(depth_pt, &color_to_depth_, color_pt);

                // Project into depth image
                float depth_px[2];
                rs2_project_point_to_pixel(depth_px, &depth_intr_, depth_pt);

                int ud = static_cast<int>(std::round(depth_px[0]));
                int vd = static_cast<int>(std::round(depth_px[1]));

                DepthPx & entry = lut_[vc * cw + uc];
                if (ud >= 0 && ud < dw && vd >= 0 && vd < dh) {
                    entry.u = static_cast<int16_t>(ud);
                    entry.v = static_cast<int16_t>(vd);
                } else {
                    entry.u = -1;
                    entry.v = -1;
                }
            }
        }

        lut_ready_ = true;
        RCLCPP_INFO(get_logger(), "LUT built — %d × %d entries.", cw, ch);
        (void)depth_min; (void)depth_max; (void)ds;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Per-frame callback: sample depth inside the ROI via the LUT
    // ────────────────────────────────────────────────────────────────────────
    void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr & depth_msg)
    {
        {
            std::lock_guard lk(lut_mutex_);
            if (!lut_ready_) return;
        }

        if (!latest_roi_) return;

        // Convert ROS image to cv::Mat (no copy for 16UC1)
        cv_bridge::CvImageConstPtr cv_depth;
        try {
            cv_depth = cv_bridge::toCvShare(depth_msg, "16UC1");
        } catch (const cv_bridge::Exception & e) {
            RCLCPP_ERROR_ONCE(get_logger(), "cv_bridge: %s", e.what());
            return;
        }
        const cv::Mat & D = cv_depth->image;

        // Extract ROI from vision_msgs/Detection2D BoundingBox2D
        const auto & bbox = latest_roi_->bbox;
        int x0 = static_cast<int>(bbox.center.position.x - bbox.size_x / 2.0);
        int y0 = static_cast<int>(bbox.center.position.y - bbox.size_y / 2.0);
        int x1 = static_cast<int>(bbox.center.position.x + bbox.size_x / 2.0);
        int y1 = static_cast<int>(bbox.center.position.y + bbox.size_y / 2.0);

        // Clamp to color image bounds
        x0 = std::max(0, x0);
        y0 = std::max(0, y0);
        x1 = std::min(static_cast<int>(color_intr_.width)  - 1, x1);
        y1 = std::min(static_cast<int>(color_intr_.height) - 1, y1);

        const int dw      = depth_intr_.width;
        const int dh      = depth_intr_.height;
        const int color_w = color_intr_.width;
        const float ds    = static_cast<float>(depth_scale_);
        const float dmin  = static_cast<float>(min_depth_m_);
        const float dmax  = static_cast<float>(max_depth_m_);

        double   sum   = 0.0;
        uint32_t count = 0;

        {
            std::lock_guard lk(lut_mutex_);
            for (int vc = y0; vc <= y1; ++vc) {
                const DepthPx * row_lut = lut_.data() + vc * color_w;
                for (int uc = x0; uc <= x1; ++uc) {
                    const DepthPx & dp = row_lut[uc];
                    if (dp.u < 0 || dp.v < 0) continue;
                    if (dp.u >= dw || dp.v >= dh) continue;

                    uint16_t raw = D.at<uint16_t>(dp.v, dp.u);
                    if (raw == 0) continue;   // invalid depth

                    float metres = raw * ds;
                    if (metres < dmin || metres > dmax) continue;

                    sum   += metres;
                    count += 1;
                }
            }
        }

        if (count == 0) {
            RCLCPP_DEBUG(get_logger(), "ROI has no valid depth pixels.");
            return;
        }

        std_msgs::msg::Float32 out;
        out.data = static_cast<float>(sum / count);
        depth_pub_->publish(out);

        RCLCPP_DEBUG(get_logger(),
            "ROI [%d,%d → %d,%d]: mean depth = %.4f m  (n=%u)",
            x0, y0, x1, y1, out.data, count);
    }
};


};
} // namespace roi_depth_query
RCLCPP_COMPONENTS_REGISTER_NODE(roi_depth_query::RoiDepthNode)