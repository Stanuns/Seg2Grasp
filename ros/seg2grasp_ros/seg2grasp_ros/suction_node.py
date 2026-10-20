"""Seg2Grasp ROS 2 node: live suction-point detection from RealSense / Orbbec topics.

Subscribes to the camera color image, depth image and both camera_infos,
registers the depth into the color camera (skipped when it already is, e.g. with
RealSense ``align_depth.enable:=true`` or Orbbec ``depth_registration:=true``),
runs segmentation -> target selection -> suction planning, and publishes the result.
Two-stage pipeline: a worker thread segments the newest frame (GPU, float16 by
default) while the main thread plans the suction point on the previous one. Frames
that arrive while busy are dropped, and the window always shows the latest result.

Publishes:
    ~/suction_pose  geometry_msgs/PoseStamped  suction point (m) in the color optical
                    frame; the pose's z axis is the approach direction (into the surface)
    ~/result_image  sensor_msgs/Image (bgr8)   annotated result

    source venvs/seg/bin/activate && source install/setup.bash
    ros2 run seg2grasp_ros suction_node
    ros2 launch seg2grasp_ros seg2grasp.launch.py   # also starts the camera

Quit with q/Esc in the window, by closing the window, or Ctrl+C.
"""
import array
import json
import threading
import time

import cv2
import message_filters
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

from seg2grasp.camera.depth_registration import backproject, register_depth_to_color
from seg2grasp.pipeline import Seg2GraspPipeline
from seg2grasp.segmentation.segmenter import Segmenter
from seg2grasp.segmentation.visualize import annotate_result

WINDOW = "Seg2Grasp (q=quit)"

# Reliable, newest-only. Best-effort image subscriptions can receive nothing at all
# when large images are fragmented over UDP (observed with CycloneDDS here).
IMAGE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=2)
INFO_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      history=HistoryPolicy.KEEP_LAST, depth=1)

_ENCODINGS = {  # encoding -> (dtype, channels)
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3),
    "16UC1": (np.uint16, 1), "mono16": (np.uint16, 1), "32FC1": (np.float32, 1),
}


def image_to_numpy(msg):
    """sensor_msgs/Image -> np.ndarray (no cv_bridge, so it works in the venv)."""
    dtype, ch = _ENCODINGS[msg.encoding]
    row = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // np.dtype(dtype).itemsize)
    img = row[:, :msg.width * ch]
    return img.reshape(msg.height, msg.width, ch) if ch > 1 else img


def numpy_to_image(img, header):
    msg = Image(header=header, height=img.shape[0], width=img.shape[1], encoding="bgr8",
                is_bigendian=0, step=img.shape[1] * 3)
    data = array.array("B")
    data.frombytes(np.ascontiguousarray(img).tobytes())
    msg.data = data       # array.array is taken as-is; bytes are converted per element (~50 ms)
    return msg


def approach_quaternion(normal):
    """Quaternion (x, y, z, w) of a frame whose z axis is the approach direction -normal."""
    z = -np.asarray(normal, float)
    z /= np.linalg.norm(z)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    return Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat()


class SuctionNode(Node):
    def __init__(self):
        super().__init__("seg2grasp")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("color_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("depth_info_topic", "/camera/camera/depth/camera_info")
        self.declare_parameter("roi", "")                 # "y,x,h,w" crop of the color image
        self.declare_parameter("vacuum_radius", 30.0)     # suction cup radius (mm)
        self.declare_parameter("seal_min", 0.98)          # min sealing fraction of the cup disc; 0 = off
        self.declare_parameter("show", True)              # display window
        self.declare_parameter("sync_slop", 0.02)         # color/depth stamp tolerance (s)
        self.declare_parameter("fp16", True)              # float16 inference (~3x faster segmentation)
        self.declare_parameter("seg_min_size", 0)         # segmentation input short edge; 0 = model default (800)
        self.declare_parameter("z_near", 300.0)           # depth band (mm) of object points used for grasping
        self.declare_parameter("z_far", 1050.0)

        p = lambda name: self.get_parameter(name).value
        roi = p("roi").strip()
        self.roi = tuple(int(v) for v in roi.split(",")) if roi else None
        self.show = bool(p("show"))

        self.get_logger().info("Loading segmentation model ...")
        z_near, z_far = float(p("z_near")), float(p("z_far"))
        segmenter = Segmenter(fp16=bool(p("fp16")), min_size=int(p("seg_min_size")) or None, depth_min=z_near)
        self.pipe = Seg2GraspPipeline(segmenter, classifier=None, vacuum_radius=float(p("vacuum_radius")),
                                      seal_min=float(p("seal_min")), z_near=z_near, z_far=z_far)

        self.k_color = self.k_depth = None
        self.depth_frame = None
        self.extrinsics = None          # (R, t[mm]) depth -> color
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._handoff = threading.Condition()
        self._segmented = None          # newest segmented frame awaiting grasp planning
        self._last_out = self._period = None

        self.create_subscription(CameraInfo, p("color_info_topic"), self._on_color_info, INFO_QOS)
        self.create_subscription(CameraInfo, p("depth_info_topic"), self._on_depth_info, INFO_QOS)
        color_sub = message_filters.Subscriber(self, Image, p("color_topic"), qos_profile=IMAGE_QOS)
        depth_sub = message_filters.Subscriber(self, Image, p("depth_topic"), qos_profile=IMAGE_QOS)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=float(p("sync_slop")))
        self._sync.registerCallback(self._on_frames)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(PoseStamped, "~/suction_pose", 10)
        self.image_pub = self.create_publisher(Image, "~/result_image", 2)
        self.get_logger().info(f"Waiting for {p('color_topic')} + {p('depth_topic')} ...")

    # --- callbacks (executor thread): only store data ---
    def _on_color_info(self, msg):
        self.k_color = np.array(msg.k, float).reshape(3, 3)

    def _on_depth_info(self, msg):
        self.k_depth = np.array(msg.k, float).reshape(3, 3)
        self.depth_frame = msg.header.frame_id

    def _on_frames(self, color_msg, depth_msg):
        with self._lock:
            self._latest = (color_msg, depth_msg)
            self._seq += 1
        self._new_frame.set()

    def latest(self):
        with self._lock:
            return self._seq, self._latest

    # --- stage 1 (worker thread): preprocessing + segmentation of the newest frame ---
    def _depth_to_color(self, depth, color_msg, depth_msg):
        """Depth (mm) on the color image grid, registering it if it is not already."""
        if depth_msg.header.frame_id == color_msg.header.frame_id and depth.shape == (color_msg.height, color_msg.width):
            return depth                                     # already aligned to color
        if self.k_depth is None:
            return None
        if self.extrinsics is None:
            try:
                tf = self.tf_buffer.lookup_transform(color_msg.header.frame_id, depth_msg.header.frame_id,
                                                     rclpy.time.Time())
            except tf2_ros.TransformException as e:
                self.get_logger().warn(f"Waiting for TF {depth_msg.header.frame_id} -> "
                                       f"{color_msg.header.frame_id}: {e}", throttle_duration_sec=2.0)
                return None
            q, t = tf.transform.rotation, tf.transform.translation
            self.extrinsics = (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                               np.array([t.x, t.y, t.z]) * 1000.0)
            self.get_logger().info(f"Registering depth into color using TF {depth_msg.header.frame_id} -> "
                                   f"{color_msg.header.frame_id} (t = {self.extrinsics[1].round(2)} mm)")
        return register_depth_to_color(depth, self.k_depth, self.k_color, *self.extrinsics,
                                       (color_msg.height, color_msg.width))

    def _prepare(self, color_msg, depth_msg):
        """Frame pair -> (rgb BGR, depth mm, organized cloud mm) on the color grid, ROI-cropped."""
        if self.k_color is None:
            self.get_logger().warn("Waiting for camera_info ...", throttle_duration_sec=2.0)
            return None
        rgb = image_to_numpy(color_msg)
        if color_msg.encoding == "rgb8":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = image_to_numpy(depth_msg).astype(np.float32)
        if depth_msg.encoding == "32FC1":
            depth *= 1000.0                                  # m -> mm
        depth = self._depth_to_color(depth, color_msg, depth_msg)
        if depth is None:
            return None
        pc = backproject(depth, self.k_color)
        if self.roi:
            y, x, h, w = self.roi
            rgb, depth, pc = rgb[y:y + h, x:x + w], depth[y:y + h, x:x + w], pc[y:y + h, x:x + w]
        return rgb, depth, pc

    def segment_loop(self):
        """Keep segmenting the newest frame and hand the result to the main thread,
        so segmentation of frame N+1 (GPU) overlaps grasp planning of frame N (CPU)."""
        last_seq = 0
        while not self._stop.is_set():
            self._new_frame.wait(timeout=0.1)
            self._new_frame.clear()
            seq, frames = self.latest()
            if seq == last_seq or frames is None:
                continue
            last_seq = seq
            try:
                t0 = time.perf_counter()
                prepared = self._prepare(*frames)
                if prepared is None:
                    continue
                rgb, depth, pc = prepared
                masks, bboxes, _ = self.pipe.segmenter.segment(rgb, depth, return_vis=False)
            except Exception as e:                          # keep the loop alive; report and go on
                self.get_logger().error(f"segmentation failed: {e!r}", throttle_duration_sec=2.0)
                continue
            with self._handoff:
                self._segmented = (frames[0].header, rgb, pc, masks, bboxes, t0)
                self._handoff.notify()

    def take_segmented(self, timeout):
        """Newest segmented frame not yet planned, or None after ``timeout`` s."""
        with self._handoff:
            if self._segmented is None:
                self._handoff.wait(timeout)
            item, self._segmented = self._segmented, None
            return item

    # --- stage 2 (main thread): grasp planning, publishing ---
    def finish(self, item):
        """Plan the suction point on a segmented frame, publish it; returns the annotated image."""
        header, rgb, pc, masks, bboxes, t0 = item
        result = self.pipe.grasp(rgb, pc, masks, bboxes, classify=False)
        now = time.perf_counter()
        latency = (now - t0) * 1000                          # from start of preprocessing
        if self._last_out is not None:
            dt = now - self._last_out
            self._period = dt if self._period is None else 0.8 * self._period + 0.2 * dt
        self._last_out = now
        rate = f"{1.0 / self._period:.1f} Hz, " if self._period else ""
        timing = f"  ({rate}latency {latency:.0f} ms)"

        if result is None:
            self.get_logger().info("no graspable object" + timing)
            vis = rgb.copy()
            cv2.putText(vis, "no graspable object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            p, n = result.suction.point.astype(float), result.suction.normal.astype(float)
            u = int(round(p[0] / p[2] * self.k_color[0, 0] + self.k_color[0, 2]))
            v = int(round(p[1] / p[2] * self.k_color[1, 1] + self.k_color[1, 2]))
            self.get_logger().info(json.dumps({"point_mm": p.round(1).tolist(), "normal": n.round(4).tolist(),
                                               "pixel": [u, v], "target": result.target.index}) + timing)
            pose = PoseStamped(header=header)
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (p / 1000.0).tolist()
            qx, qy, qz, qw = approach_quaternion(n)
            pose.pose.orientation.x, pose.pose.orientation.y = float(qx), float(qy)
            pose.pose.orientation.z, pose.pose.orientation.w = float(qz), float(qw)
            self.pose_pub.publish(pose)
            vis = annotate_result(rgb, result, pc)

        cv2.putText(vis, timing.strip(" ()"), (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.image_pub.publish(numpy_to_image(vis, header))
        return vis


def _spin(executor):
    try:
        executor.spin()
    except ExternalShutdownException:   # Ctrl+C: rclpy's signal handler shut the context down
        pass


def main():
    rclpy.init()
    node = SuctionNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=_spin, args=(executor,), daemon=True).start()
    worker = threading.Thread(target=node.segment_loop, daemon=True)
    worker.start()

    shown = False
    try:
        while rclpy.ok():
            item = node.take_segmented(timeout=0.01)
            if item is not None:
                vis = node.finish(item)
                if node.show:
                    cv2.imshow(WINDOW, vis)
                    shown = True
            if node.show:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if shown and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node._stop.set()
        worker.join(timeout=3.0)
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
