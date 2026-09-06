#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 ARC Robotics.  Modifications under the same license.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
detection_picker_visualizer.py

Diagnostic overlay for detection_picker_node (roi_depth_query::DetectionRoiRelayNode).

Derived from NVIDIA's isaac_ros_yolov8_visualizer.py, with three changes:

  1. COCO class names replaced with our 8-class model:
       0 blue_hero  1 blue_std  2 blue_stry  3 blue_na
       4 red_hero   5 red_std   6 red_stry   7 red_na

  2. Instead of drawing only "<name> <conf>", it re-computes the *same*
     composite priority score the picker uses and draws every factor that
     went into the pick decision, then highlights the detection the picker
     would have chosen. Factors shown per box:
         name, confidence, centrality, +priority-class bonus, team exclusion,
         and the final score. The winner is boxed in green and tagged PICK.

  3. QoS is set to match the rest of the pipeline (best-effort SensorDataQoS),
     so it connects to both the best-effort RealSense / NITROS image streams
     and the reliable Isaac ROS decoder, and reads the referee team colour the
     same way the picker does.

Coordinate space:
  /detections_output bboxes are in NETWORK space (network_w x network_h, e.g.
  640x640). The only published frame in that space is the DNN encoder's resize
  output (/yolov8_encoder/resize/image), so we overlay on that. This is the
  same space the picker scores in, BEFORE it scales the winner to color space
  for /roi -- which is exactly the decision we want to inspect here.

The node mirrors detection_picker_node's parameters so the overlay reflects the
live picker configuration. Keep these in sync with the launch file.
"""

import cv2
import cv_bridge
import math
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

# Optional: referee team colour. Imported lazily so the visualizer still runs
# (with team filtering disabled) if dji_serial_bridge is not on the path.
try:
    from dji_serial_bridge.msg import RefSysStatus
    HAVE_REF_SYS = True
except ImportError:
    RefSysStatus = None
    HAVE_REF_SYS = False

# Custom 8-class model. Order matters: index == YOLO class id.
NAMES = {
    0: 'blue_hero',
    1: 'blue_std',
    2: 'blue_stry',
    3: 'blue_na',
    4: 'red_hero',
    5: 'red_std',
    6: 'red_stry',
    7: 'red_na',
}


def _sensor_data_qos(depth=5):
    """Best-effort / volatile / keep-last QoS.

    Equivalent to rclcpp::SensorDataQoS(), which is what the RealSense driver,
    the NITROS image streams, and dji_serial_bridge_node's ~/ref_sys all use.
    A best-effort subscriber is compatible with both best-effort AND reliable
    publishers, so this connects to the Isaac decoder's /detections_output too.
    """
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


class DetectionPickerVisualizer(Node):
    # colours are RGB tuples (the resize image is rgb8, so viewers read them
    # as RGB -- no BGR swap needed).
    COLOR_PICK = (0, 255, 0)       # green  : the detection the picker chooses
    COLOR_ELIGIBLE = (255, 215, 0)  # amber  : passed filters, but not the best
    COLOR_EXCLUDED = (255, 60, 60)  # red    : dropped (allied team)
    COLOR_LOWCONF = (160, 160, 160)  # grey   : below min_score
    TXT_COLOR = (255, 255, 255)     # white  : label text
    TXT_BG = (0, 0, 0)              # black  : label background

    def __init__(self):
        super().__init__('detection_picker_visualizer')

        # ── parameters (mirror detection_picker_node) ────────────────────────
        self.declare_parameter('detections_topic', '/detections_output')
        self.declare_parameter('image_topic', '/yolov8_encoder/resize/image')
        self.declare_parameter('output_topic', 'yolov8_processed_image')
        self.declare_parameter('ref_sys_topic', '/dji_serial_bridge/ref_sys')
        self.declare_parameter('network_width', 640)
        self.declare_parameter('network_height', 640)
        self.declare_parameter('min_score', 0.0)
        self.declare_parameter('center_weight', 1.0)
        self.declare_parameter('priority_class_bonus', 0.5)
        self.declare_parameter('priority_class_ids', [2, 6])
        self.declare_parameter('is_blue_fallback', True)
        # exact (default, matches the original / Isaac timestamps line up) or
        # approximate sync as a fallback if stamps drift.
        self.declare_parameter('use_approx_sync', False)
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('queue_size', 10)

        gp = self.get_parameter
        detections_topic = gp('detections_topic').value
        image_topic = gp('image_topic').value
        output_topic = gp('output_topic').value
        self.ref_sys_topic = gp('ref_sys_topic').value
        self.network_w = int(gp('network_width').value)
        self.network_h = int(gp('network_height').value)
        self.min_score = float(gp('min_score').value)
        self.center_weight = float(gp('center_weight').value)
        self.priority_class_bonus = float(gp('priority_class_bonus').value)
        self.priority_class_ids = set(int(c) for c in gp('priority_class_ids').value)
        self.is_blue_team = bool(gp('is_blue_fallback').value)
        use_approx = bool(gp('use_approx_sync').value)
        slop = float(gp('sync_slop').value)
        queue_size = int(gp('queue_size').value)

        # Half the network-image diagonal: max distance from centre, used to
        # normalise centrality into [0, 1] -- identical to the C++ node.
        self.half_diag = 0.5 * math.sqrt(
            self.network_w * self.network_w + self.network_h * self.network_h)

        self._bridge = cv_bridge.CvBridge()
        self._pub = self.create_publisher(Image, output_topic, queue_size)

        qos = _sensor_data_qos(queue_size)
        self._det_sub = message_filters.Subscriber(
            self, Detection2DArray, detections_topic, qos_profile=qos)
        self._img_sub = message_filters.Subscriber(
            self, Image, image_topic, qos_profile=qos)

        if use_approx:
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._det_sub, self._img_sub], queue_size, slop)
        else:
            self._sync = message_filters.TimeSynchronizer(
                [self._det_sub, self._img_sub], queue_size)
        self._sync.registerCallback(self.detections_callback)

        # Team colour: same source & QoS as the picker (SensorDataQoS). A
        # reliable/transient-local sub would be QoS-incompatible and silently
        # never connect.
        if HAVE_REF_SYS:
            self._ref_sub = self.create_subscription(
                RefSysStatus, self.ref_sys_topic, self.ref_sys_callback,
                _sensor_data_qos())
        else:
            self.get_logger().warn(
                "dji_serial_bridge.msg.RefSysStatus not importable -- team "
                "exclusion will not be shown (all classes treated as eligible).")

        self.get_logger().info(
            f"detection_picker_visualizer ready\n"
            f"  detections: {detections_topic}  image: {image_topic}\n"
            f"  -> {output_topic}\n"
            f"  network {self.network_w}x{self.network_h}  min_score={self.min_score}\n"
            f"  score = conf + {self.center_weight}*centrality + "
            f"{self.priority_class_bonus} if class in "
            f"{sorted(self.priority_class_ids)}\n"
            f"  team source: {self.ref_sys_topic}  "
            f"sync: {'approx(%.3fs)' % slop if use_approx else 'exact'}")

    # ── referee status callback (mirror of C++ onRefSysStatus) ───────────────
    def ref_sys_callback(self, msg):
        new_val = bool(msg.is_on_blue_team)
        if self.is_blue_team is None or self.is_blue_team != new_val:
            self.get_logger().info(
                f"Team colour set to {'BLUE' if new_val else 'RED'} "
                f"(excluding class IDs {'0-3' if new_val else '4-7'})")
        self.is_blue_team = new_val

    # ── picker logic, replicated 1:1 from detection_picker_node.cpp ──────────
    def _top_hypothesis(self, detection):
        """Return (class_id:int, confidence:float) of the highest-score hyp.

        Mirrors topClassId(): the class id is read from the highest-scoring
        hypothesis. class_id is a decimal string in Isaac ROS; -1 on failure.
        """
        best_score = -1.0
        best_id = -1
        for hyp in detection.results:
            if hyp.hypothesis.score > best_score:
                best_score = hyp.hypothesis.score
                try:
                    best_id = int(hyp.hypothesis.class_id)
                except (ValueError, TypeError):
                    best_id = -1
        return best_id, max(best_score, 0.0)

    def _is_excluded_by_team(self, class_id):
        if self.is_blue_team is None:
            return False
        if self.is_blue_team:
            return 0 <= class_id <= 3
        return 4 <= class_id <= 7

    def _centrality(self, detection):
        dx = detection.bbox.center.position.x - 0.5 * self.network_w
        dy = detection.bbox.center.position.y - 0.5 * self.network_h
        dist = math.hypot(dx, dy)
        c = 1.0 - dist / self.half_diag
        return min(max(c, 0.0), 1.0)

    # ── main callback ────────────────────────────────────────────────────────
    def detections_callback(self, detections_msg, img_msg):
        cv2_img = self._bridge.imgmsg_to_cv2(img_msg)

        lw = max(round((img_msg.height + img_msg.width) / 2 * 0.003), 2)
        tf = max(lw - 1, 1)
        font_scale = lw / 3.0

        # First pass: compute every factor for every detection, exactly as the
        # picker would, and find the winner (highest score among eligibles).
        infos = []
        best_idx = -1
        best_score = -1.0
        for i, det in enumerate(detections_msg.detections):
            class_id, conf = self._top_hypothesis(det)
            excluded = self._is_excluded_by_team(class_id)
            centrality = self._centrality(det)
            is_prio = class_id in self.priority_class_ids
            score = (conf
                     + self.center_weight * centrality
                     + (self.priority_class_bonus if is_prio else 0.0))
            # eligibility gate: not allied AND clears min_score on raw conf
            eligible = (not excluded) and (conf >= self.min_score)
            infos.append({
                'class_id': class_id, 'conf': conf, 'centrality': centrality,
                'is_prio': is_prio, 'score': score, 'excluded': excluded,
                'eligible': eligible,
            })
            if eligible and score > best_score:
                best_score = score
                best_idx = i

        for i, det in enumerate(detections_msg.detections):
            info = infos[i]
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            min_pt = (round(cx - w / 2.0), round(cy - h / 2.0))
            max_pt = (round(cx + w / 2.0), round(cy + h / 2.0))

            if i == best_idx:
                color = self.COLOR_PICK
                box_thick = lw + 1
            elif info['excluded']:
                color = self.COLOR_EXCLUDED
                box_thick = max(lw - 1, 1)
            elif not info['eligible']:
                color = self.COLOR_LOWCONF
                box_thick = max(lw - 1, 1)
            else:
                color = self.COLOR_ELIGIBLE
                box_thick = lw

            cv2.rectangle(cv2_img, min_pt, max_pt, color, box_thick)

            name = NAMES.get(info['class_id'], f"id{info['class_id']}")
            lines = [
                f"#{i} {name}" + ("  <PICK>" if i == best_idx else ""),
                f"conf {info['conf']:.2f}  cen {info['centrality']:.2f}",
                f"score {info['score']:.2f}"
                + ("  +prio" if info['is_prio'] else ""),
            ]
            if info['excluded']:
                lines.append("EXCLUDED: ally")
            elif not info['eligible']:
                lines.append(f"< min_score {self.min_score:.2f}")

            self._draw_label(cv2_img, lines, min_pt, max_pt,
                             color, font_scale, tf)

        processed = self._bridge.cv2_to_imgmsg(cv2_img, encoding=img_msg.encoding)
        processed.header = img_msg.header
        self._pub.publish(processed)

    def _draw_label(self, img, lines, min_pt, max_pt, accent, font_scale, tf):
        """Draw a multi-line label block with a filled background for legibility.

        Placed above the box if there is room, otherwise below it.
        """
        sizes = [cv2.getTextSize(t, 0, fontScale=font_scale, thickness=tf)[0]
                 for t in lines]
        line_h = max(s[1] for s in sizes) + 4
        block_w = max(s[0] for s in sizes) + 6
        block_h = line_h * len(lines) + 2

        x0 = min_pt[0]
        if min_pt[1] - block_h >= 0:        # room above the box
            y0 = min_pt[1] - block_h
        else:                                # otherwise drop below
            y0 = max_pt[1]
        # keep the block inside the image horizontally
        img_w = img.shape[1]
        x0 = max(0, min(x0, img_w - block_w))

        cv2.rectangle(img, (x0, y0), (x0 + block_w, y0 + block_h),
                      self.TXT_BG, -1)
        # accent stripe so the label colour-matches its box
        cv2.rectangle(img, (x0, y0), (x0 + 3, y0 + block_h), accent, -1)

        y = y0
        for text, (_, th) in zip(lines, sizes):
            y += line_h
            cv2.putText(img, text, (x0 + 5, y - 3), 0, font_scale,
                        self.TXT_COLOR, thickness=tf, lineType=cv2.LINE_AA)


def main():
    rclpy.init()
    node = DetectionPickerVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
