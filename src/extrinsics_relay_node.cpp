// extrinsics_relay_node.cpp
//
// One-shot node: subscribes to the realsense extrinsics topic and on first
// receipt sets extrinsics.rotation / extrinsics.translation parameters on
// roi_depth_node via the ROS 2 parameter service, then exits.
//
// QoS note:
//   realsense-ros publishes the extrinsics topic with VOLATILE durability
//   (the ROS default). A TRANSIENT_LOCAL subscriber is incompatible with a
//   VOLATILE publisher in DDS — no messages will ever be delivered across
//   that pairing regardless of history depth. This node therefore uses the
//   default (volatile) QoS to match the publisher.
//
// Topic note:
//   With ComposableNode(name='camera', namespace=''), realsense-ros resolves
//   all relative topic names against namespace '' (root). The extrinsics
//   topic is therefore at /extrinsics/depth_to_color, NOT at
//   /camera/camera/extrinsics/depth_to_color.

#include <rclcpp/rclcpp.hpp>
#include <realsense2_camera_msgs/msg/extrinsics.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("extrinsics_relay");

    // Default topic matches observed realsense-ros behaviour with namespace=''
    std::string extr_topic = node->declare_parameter<std::string>(
        "extrinsics_topic", "/extrinsics/depth_to_color");
    std::string target_node = node->declare_parameter<std::string>(
        "target_node", "/roi_depth_node");

    RCLCPP_INFO(node->get_logger(),
        "Waiting for extrinsics on '%s' (volatile QoS) …", extr_topic.c_str());

    auto client = node->create_client<rcl_interfaces::srv::SetParameters>(
        target_node + "/set_parameters");

    bool done = false;
    rclcpp::Subscription<realsense2_camera_msgs::msg::Extrinsics>::SharedPtr sub;

    // Use default (volatile) QoS — must match the realsense publisher.
    // DO NOT use .transient_local() here: realsense-ros uses volatile durability,
    // and a transient_local subscriber paired with a volatile publisher will
    // never receive a single message.
    sub = node->create_subscription<realsense2_camera_msgs::msg::Extrinsics>(
        extr_topic, rclcpp::QoS(1),
        [&](realsense2_camera_msgs::msg::Extrinsics::ConstSharedPtr msg)
        {
            if (done) return;
            done = true;

            RCLCPP_INFO(node->get_logger(),
                "Extrinsics received — forwarding to %s", target_node.c_str());

            if (!client->wait_for_service(std::chrono::seconds(5)))
            {
                RCLCPP_ERROR(node->get_logger(),
                    "Parameter service on %s not available after 5 s — "
                    "is roi_depth_node running?", target_node.c_str());
                rclcpp::shutdown();
                return;
            }

            auto req = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();

            rcl_interfaces::msg::Parameter rot_param;
            rot_param.name = "extrinsics.rotation";
            rot_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_DOUBLE_ARRAY;
            for (float v : msg->rotation)
                rot_param.value.double_array_value.push_back(static_cast<double>(v));
            req->parameters.push_back(rot_param);

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
