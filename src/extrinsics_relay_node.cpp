// extrinsics_relay_node.cpp
//
// One-shot node: subscribes to the realsense extrinsics topic (transient_local),
// and on first receipt sets extrinsics.rotation / extrinsics.translation
// parameters on roi_depth_node via the ROS 2 parameter service, then exits.

#include <rclcpp/rclcpp.hpp>
#include <realsense2_camera_msgs/msg/extrinsics.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("extrinsics_relay");

    std::string extr_topic = node->declare_parameter<std::string>(
        "extrinsics_topic", "/camera/camera/extrinsics/depth_to_color");
    std::string target_node = node->declare_parameter<std::string>(
        "target_node", "/roi_depth_node");

    auto client = node->create_client<rcl_interfaces::srv::SetParameters>(
        target_node + "/set_parameters");

    rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr sub;
    sub = node->create_subscription<realsense2_camera_msgs::msg::Extrinsics>(
        extr_topic, rclcpp::QoS(1).transient_local(),
        [&](realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr msg)
        {
            RCLCPP_INFO(node->get_logger(),
                        "Extrinsics received — forwarding to %s", target_node.c_str());

            // Wait for the parameter service to be available
            if (!client->wait_for_service(std::chrono::seconds(5)))
            {
                RCLCPP_ERROR(node->get_logger(),
                             "Parameter service on %s not available", target_node.c_str());
                return;
            }

            auto req = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();

            // rotation (9 doubles)
            rcl_interfaces::msg::Parameter rot_param;
            rot_param.name = "extrinsics.rotation";
            rot_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE_ARRAY;
            for (float v : msg->rotation)
                rot_param.value.double_array_value.push_back(static_cast<double>(v));
            req->parameters.push_back(rot_param);

            // translation (3 doubles)
            rcl_interfaces::msg::Parameter trans_param;
            trans_param.name = "extrinsics.translation";
            trans_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE_ARRAY;
            for (float v : msg->translation)
                trans_param.value.double_array_value.push_back(static_cast<double>(v));
            req->parameters.push_back(trans_param);

            client->async_send_request(req,
                                       [node](rclcpp::Client<rcl_interfaces::srv::SetParameters>::SharedFuture)
                                       {
                                           RCLCPP_INFO(node->get_logger(), "Extrinsics forwarded — shutting down.");
                                           rclcpp::shutdown();
                                       });
        });

    rclcpp::spin(node);
    return 0;
}