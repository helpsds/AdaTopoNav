#!/usr/bin/env python3
"""Record a real ROS camera stream during one navigation trial as MP4."""

import argparse
import os
import time

import cv2
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


class NavigationVideoRecorder:
    def __init__(self, args):
        self.args = args
        self.writer = None
        self.start_time = time.monotonic()
        self.last_frame_time = float("-inf")
        self.position = None
        self.frames = 0
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        rospy.Subscriber(args.odom_topic, Odometry, self.on_odom, queue_size=1)
        rospy.Subscriber(args.image_topic, Image, self.on_image, queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self.close)
        rospy.loginfo("Recording %s to %s", args.image_topic, args.output)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.position = (p.x, p.y)

    @staticmethod
    def decode(msg):
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
        if msg.encoding not in channels:
            raise ValueError("unsupported camera encoding: %s" % msg.encoding)
        n = channels[msg.encoding]
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        pixels = rows[:, :msg.width * n].reshape(msg.height, msg.width, n)
        if msg.encoding == "rgb8":
            return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        if msg.encoding == "rgba8":
            return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
        if msg.encoding == "bgra8":
            return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
        if msg.encoding == "mono8":
            return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        return pixels.copy()

    def on_image(self, msg):
        now = time.monotonic()
        if now - self.last_frame_time < 1.0 / self.args.fps:
            return
        self.last_frame_time = now
        try:
            frame = self.decode(msg)
            width = self.args.width
            height = int(round(frame.shape[0] * width / frame.shape[1] / 2) * 2)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if self.writer is None:
                self.writer = cv2.VideoWriter(
                    self.args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                    self.args.fps, (width, height),
                )
                if not self.writer.isOpened():
                    raise RuntimeError("OpenCV could not open MP4 writer")
            elapsed = now - self.start_time
            cv2.rectangle(frame, (0, 0), (width, 62), (0, 0, 0), -1)
            cv2.putText(frame, self.args.label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            details = "%s | %.1f s" % (self.args.scene_label, elapsed)
            if self.position is not None:
                details += " | odom %.2f, %.2f m" % self.position
            cv2.putText(frame, details, (12, 51), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            self.writer.write(frame)
            self.frames += 1
        except Exception as exc:
            rospy.logerr("Video recording failed: %s", exc)
            rospy.signal_shutdown(str(exc))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        rospy.loginfo("Saved %d frames to %s", self.frames, self.args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scene-label", default="Navigation scene")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args(rospy.myargv()[1:])
    if args.fps <= 0 or args.width <= 0 or args.width % 2:
        parser.error("fps must be positive and width must be a positive even number")
    rospy.init_node("navigation_video_recorder", anonymous=True, disable_signals=False)
    NavigationVideoRecorder(args)
    rospy.spin()


if __name__ == "__main__":
    main()
