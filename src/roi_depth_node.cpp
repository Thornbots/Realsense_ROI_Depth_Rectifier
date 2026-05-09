// roi_depth_node.cpp  (updated)
//
// Efficiently computes the non-zero mean depth inside a COLOR-space ROI
// WITHOUT running rs2::align on the full frame.
//
// IMPORTANT: /roi must arrive in color image space (NOT network/640x640 space).
// Use detection_roi_relay_node to bridge from yolov8's /detections_output.
//
// New parameter:
//   center_sample_fraction (double, default 0.25):
//     Only the inner (fraction) of each bbox dimension is sampled.
//     0.25 → inner 25% per axis → 6.25% of the bbox area.
//     Set to 1.0 to sample the full bbox.
//
// Topics consumed:
//   /camera/depth/image_rect_raw   (sensor_msgs/Image, 16UC1)
//   /camera/depth/camera_info      (sensor_msgs/CameraInfo)
//   /camera/color/camera_info      (sensor_msgs/CameraInfo)
//   /roi                           (vision_msgs/Detection2D — color image space)
//   extrinsics arrive via parameter push from extrinsics_relay_node
//
// Topic published:
//   /roi_depth_m   (std_msgs/Float32) — mean depth in metres

#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/float32.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <realsense2_camera_msgs/msg/extrinsics.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <librealsense2/rsutil.h>
#include <librealsense2/rs.hpp>

#include <vector>
#include <cmath>
#include <mutex>
#include <algorithm>

namespace roi_depth_query
{
    class RoiDepthNode : public rclcpp::Node
    {
    public:
        explicit RoiDepthNode(const rclcpp::NodeOptions &opts = rclcpp::NodeOptions())
            : Node("roi_depth_node", opts)
        {
            // ── parameters ──────────────────────────────────────────────────────
            depth_ns_   = declare_parameter<std::string>("depth_ns",   "/camera/depth");
            color_ns_   = declare_parameter<std::string>("color_ns",   "/camera/color");
            extr_topic_ = declare_parameter<std::string>("extrinsics_topic",
                              "/camera/camera/extrinsics/depth_to_color");
            depth_scale_   = declare_parameter<double>("depth_scale",   0.001);
            min_depth_m_   = declare_parameter<double>("min_depth_m",   0.1);
            max_depth_m_   = declare_parameter<double>("max_depth_m",   10.0);
            center_sample_fraction_ = declare_parameter<double>("center_sample_fraction", 0.25);

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

            // Extrinsics arrive via parameter push from extrinsics_relay_node
            declare_parameter("extrinsics.rotation",    std::vector<double>(9, 0.0));
            declare_parameter("extrinsics.translation", std::vector<double>(3, 0.0));

            extr_param_timer_ = create_wall_timer(
                std::chrono::milliseconds(100),
                [this]() {
                    if (extr_ready_) { extr_param_timer_->cancel(); return; }
                    auto rot   = get_parameter("extrinsics.rotation").as_double_array();
                    auto trans = get_parameter("extrinsics.translation").as_double_array();
                    // Non-identity check: at least one diagonal element non-zero
                    if (rot.size() == 9 && (rot[0] != 0.0 || rot[4] != 0.0 || rot[8] != 0.0)) {
                        std::lock_guard lk(lut_mutex_);
                        rs2_extrinsics d2c{};
                        for (int i = 0; i < 9; ++i) d2c.rotation[i]    = float(rot[i]);
                        for (int i = 0; i < 3; ++i) d2c.translation[i] = float(trans[i]);
                        depth_to_color_ = d2c;
                        // Invert: c2d.R = d2c.R^T, c2d.t = -R^T * t
                        rs2_extrinsics c2d{};
                        for (int r = 0; r < 3; ++r)
                            for (int c = 0; c < 3; ++c)
                                c2d.rotation[r*3+c] = d2c.rotation[c*3+r];
                        for (int i = 0; i < 3; ++i) {
                            c2d.translation[i] = 0;
                            for (int j = 0; j < 3; ++j)
                                c2d.translation[i] -= c2d.rotation[i*3+j] * d2c.translation[j];
                        }
                        color_to_depth_ = c2d;
                        extr_ready_ = true;
                        extr_param_timer_->cancel();
                        tryBuildLut();
                        RCLCPP_INFO(get_logger(), "Extrinsics received via parameter.");
                    }
                });

            // /roi is in color image space — relay node handles the scaling
            roi_sub_ = create_subscription<vision_msgs::msg::Detection2D>(
                "/roi", 10,
                [this](vision_msgs::msg::Detection2D::ConstSharedPtr m) {
                    latest_roi_ = m;
                });

            depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
                depth_ns_ + "/image_rect_raw", rclcpp::SensorDataQoS(),
                std::bind(&RoiDepthNode::onDepth, this, std::placeholders::_1));

            depth_pub_ = create_publisher<std_msgs::msg::Float32>("/roi_depth_m", 10);

            RCLCPP_INFO(get_logger(),
                "roi_depth_node ready | depth_ns=%s color_ns=%s | center_sample_fraction=%.2f",
                depth_ns_.c_str(), color_ns_.c_str(), center_sample_fraction_);
        }

    private:
        rs2_intrinsics cameraInfoToIntrinsics(const sensor_msgs::msg::CameraInfo &ci)
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

        rclcpp::TimerBase::SharedPtr extr_param_timer_;

        struct DepthPx { int16_t u, v; };
        std::vector<DepthPx> lut_;
        std::mutex lut_mutex_;
        bool lut_ready_{false};

        rs2_intrinsics depth_intr_{}, color_intr_{};
        rs2_extrinsics depth_to_color_{}, color_to_depth_{};
        bool depth_intr_ready_{false}, color_intr_ready_{false}, extr_ready_{false};

        vision_msgs::msg::Detection2D::ConstSharedPtr latest_roi_;

        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_sub_, color_info_sub_;
        rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr extr_sub_;
        rclcpp::Subscription<vision_msgs::msg::Detection2D>::SharedPtr roi_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr depth_pub_;

        std::string depth_ns_, color_ns_, extr_topic_;
        double depth_scale_, min_depth_m_, max_depth_m_, center_sample_fraction_;

        void tryBuildLut()
        {
            if (!depth_intr_ready_ || !color_intr_ready_) return;

            if (!extr_ready_) {
                float I[9] = {1,0,0, 0,1,0, 0,0,1};
                float T[3] = {0,0,0};
                std::copy(I, I+9, depth_to_color_.rotation);
                std::copy(T, T+3, depth_to_color_.translation);
                std::copy(I, I+9, color_to_depth_.rotation);
                std::copy(T, T+3, color_to_depth_.translation);
                RCLCPP_WARN_ONCE(get_logger(),
                    "Extrinsics not yet received — using identity. "
                    "Ensure extrinsics_relay_node is running.");
            }

            const int cw = color_intr_.width;
            const int ch = color_intr_.height;
            const int dw = depth_intr_.width;
            const int dh = depth_intr_.height;

            RCLCPP_INFO(get_logger(),
                "Building LUT: color %dx%d → depth %dx%d …", cw, ch, dw, dh);
            lut_.resize(static_cast<size_t>(cw * ch));

            for (int vc = 0; vc < ch; ++vc) {
                for (int uc = 0; uc < cw; ++uc) {
                    float cpx[2] = {float(uc), float(vc)};
                    float cpt[3], dpt[3], dpx[2];
                    rs2_deproject_pixel_to_point(cpt, &color_intr_, cpx, 1.0f);
                    rs2_transform_point_to_point(dpt, &color_to_depth_, cpt);
                    rs2_project_point_to_pixel(dpx, &depth_intr_, dpt);
                    int ud = static_cast<int>(std::round(dpx[0]));
                    int vd = static_cast<int>(std::round(dpx[1]));
                    DepthPx &e = lut_[vc * cw + uc];
                    if (ud >= 0 && ud < dw && vd >= 0 && vd < dh) {
                        e.u = int16_t(ud); e.v = int16_t(vd);
                    } else {
                        e.u = -1; e.v = -1;
                    }
                }
            }
            lut_ready_ = true;
            RCLCPP_INFO(get_logger(), "LUT built: %d×%d entries.", cw, ch);
        }

        void onDepth(const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg)
        {
            { std::lock_guard lk(lut_mutex_); if (!lut_ready_) return; }
            if (!latest_roi_) return;

            cv_bridge::CvImageConstPtr cv_depth;
            try { cv_depth = cv_bridge::toCvShare(depth_msg, "16UC1"); }
            catch (const cv_bridge::Exception &e) {
                RCLCPP_ERROR_ONCE(get_logger(), "cv_bridge: %s", e.what());
                return;
            }
            const cv::Mat &D = cv_depth->image;

            const auto &bbox = latest_roi_->bbox;
            double half_w = bbox.size_x / 2.0;
            double half_h = bbox.size_y / 2.0;

            // Shrink to center fraction
            double frac = std::clamp(center_sample_fraction_, 0.05, 1.0);
            int x0 = static_cast<int>(std::round(bbox.center.position.x - half_w * frac));
            int y0 = static_cast<int>(std::round(bbox.center.position.y - half_h * frac));
            int x1 = static_cast<int>(std::round(bbox.center.position.x + half_w * frac));
            int y1 = static_cast<int>(std::round(bbox.center.position.y + half_h * frac));

            x0 = std::max(0, x0);
            y0 = std::max(0, y0);
            x1 = std::min(int(color_intr_.width)  - 1, x1);
            y1 = std::min(int(color_intr_.height) - 1, y1);

            if (x0 > x1 || y0 > y1) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                    "Sampled ROI is empty [%d,%d→%d,%d]", x0, y0, x1, y1);
                return;
            }

            const int dw      = depth_intr_.width;
            const int dh      = depth_intr_.height;
            const int color_w = color_intr_.width;
            const float ds    = float(depth_scale_);
            const float dmin  = float(min_depth_m_);
            const float dmax  = float(max_depth_m_);

            double sum = 0.0; uint32_t count = 0;

            {
                std::lock_guard lk(lut_mutex_);
                for (int vc = y0; vc <= y1; ++vc) {
                    const DepthPx *row = lut_.data() + vc * color_w;
                    for (int uc = x0; uc <= x1; ++uc) {
                        const DepthPx &dp = row[uc];
                        if (dp.u < 0 || dp.v < 0 || dp.u >= dw || dp.v >= dh) continue;
                        uint16_t raw = D.at<uint16_t>(dp.v, dp.u);
                        if (raw == 0) continue;
                        float m = raw * ds;
                        if (m < dmin || m > dmax) continue;
                        sum += m; count++;
                    }
                }
            }

            if (count == 0) {
                RCLCPP_DEBUG(get_logger(), "No valid depth in center ROI.");
                return;
            }

            std_msgs::msg::Float32 out;
            out.data = float(sum / count);
            depth_pub_->publish(out);

            RCLCPP_DEBUG(get_logger(),
                "Center ROI [%d,%d→%d,%d] frac=%.2f: %.4f m (n=%u)",
                x0, y0, x1, y1, frac, out.data, count);
        }
    };

}  // namespace roi_depth_query

RCLCPP_COMPONENTS_REGISTER_NODE(roi_depth_query::RoiDepthNode)
