#!/usr/bin/env python3
"""Record one navigation trial using explicit metric and collision criteria."""

import argparse
import csv
import math
import os
import time

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String


class TrialEvaluator:
    def __init__(self, args):
        self.args = args
        self.start_wall = time.monotonic()
        self.last_pose = None
        self.last_odom_wall = None
        self.travelled_distance = 0.0
        self.integrated_abs_angular_velocity = 0.0
        self.rotation_in_place_seconds = 0.0
        self.final_distance = math.inf
        self.collision_count = 0
        self.last_collision_time = -math.inf
        self.collision_active = False
        self.planner_reached = False
        self.termination_reason = "ROS_SHUTDOWN"
        self.done = False

        rospy.Subscriber(args.odom_topic, Odometry, self.odom_callback, queue_size=20)
        rospy.Subscriber(
            args.goal_topic, Bool, self.goal_callback, queue_size=1
        )
        if args.collision_topic:
            collision_type = Bool if args.collision_mode == "bool" else String
            rospy.Subscriber(
                args.collision_topic,
                collision_type,
                self.collision_callback,
                queue_size=20,
            )
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        rospy.on_shutdown(self.write_result)

    def odom_callback(self, msg):
        now = time.monotonic()
        pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.last_pose is not None:
            self.travelled_distance += math.hypot(
                pose[0] - self.last_pose[0], pose[1] - self.last_pose[1]
            )
        if self.last_odom_wall is not None:
            dt = min(max(now - self.last_odom_wall, 0.0), 0.5)
            linear_speed = math.hypot(
                msg.twist.twist.linear.x, msg.twist.twist.linear.y
            )
            angular_speed = abs(msg.twist.twist.angular.z)
            self.integrated_abs_angular_velocity += angular_speed * dt
            if (
                linear_speed <= self.args.rotation_linear_threshold
                and angular_speed >= self.args.rotation_angular_threshold
            ):
                self.rotation_in_place_seconds += dt
        self.last_pose = pose
        self.last_odom_wall = now
        self.final_distance = math.hypot(
            pose[0] - self.args.goal_x, pose[1] - self.args.goal_y
        )

    def goal_callback(self, msg):
        self.planner_reached = bool(msg.data)

    def collision_callback(self, msg):
        if self.args.collision_mode == "bool":
            active = bool(msg.data)
            if not active:
                self.collision_active = False
                return
            if self.collision_active:
                return
            self.collision_active = True

        now = time.monotonic()
        if now - self.last_collision_time >= self.args.collision_debounce:
            self.collision_count += 1
            self.last_collision_time = now
            if self.args.terminate_on_collision:
                self.finish("COLLISION")

    def timer_callback(self, _event):
        elapsed = time.monotonic() - self.start_wall
        if self.planner_reached and self.final_distance <= self.args.goal_distance:
            if self.args.require_collision_free and self.collision_count:
                self.finish("COLLISION_FAILURE")
            else:
                self.finish("SUCCESS")
        elif elapsed >= self.args.timeout:
            self.finish("TIMEOUT")

    def finish(self, reason):
        if self.done:
            return
        self.done = True
        self.termination_reason = reason
        rospy.signal_shutdown(reason)

    def write_result(self):
        if getattr(self, "_written", False):
            return
        self._written = True
        elapsed = time.monotonic() - self.start_wall
        success = int(self.termination_reason == "SUCCESS")
        collision_free_success = int(success and self.collision_count == 0)
        spl = ""
        if success and self.args.shortest_path_distance is not None:
            shortest = self.args.shortest_path_distance
            denominator = max(shortest, self.travelled_distance)
            spl = f"{(shortest / denominator if denominator > 0 else 0.0):.6f}"
        os.makedirs(os.path.dirname(os.path.abspath(self.args.output)), exist_ok=True)
        new_file = not os.path.exists(self.args.output)
        fields = [
            "trial_id", "environment", "method", "seed", "success",
            "collision_free_success",
            "collisions", "elapsed_seconds", "travelled_distance",
            "shortest_path_distance", "spl", "final_goal_distance",
            "integrated_abs_angular_velocity", "rotation_in_place_seconds",
            "termination_reason",
        ]
        row = {
            "trial_id": self.args.trial_id,
            "environment": self.args.environment,
            "method": self.args.method,
            "seed": self.args.seed,
            "success": success,
            "collision_free_success": collision_free_success,
            "collisions": self.collision_count,
            "elapsed_seconds": f"{elapsed:.3f}",
            "travelled_distance": f"{self.travelled_distance:.3f}",
            "shortest_path_distance": (
                "" if self.args.shortest_path_distance is None
                else f"{self.args.shortest_path_distance:.3f}"
            ),
            "spl": spl,
            "final_goal_distance": (
                "" if math.isinf(self.final_distance)
                else f"{self.final_distance:.3f}"
            ),
            "integrated_abs_angular_velocity": (
                f"{self.integrated_abs_angular_velocity:.3f}"
            ),
            "rotation_in_place_seconds": (
                f"{self.rotation_in_place_seconds:.3f}"
            ),
            "termination_reason": self.termination_reason,
        }
        with open(self.args.output, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--collision-topic", default="")
    parser.add_argument(
        "--collision-mode",
        choices=("bool", "event"),
        default="event",
        help="bool counts False-to-True edges; event debounces event messages",
    )
    parser.add_argument("--collision-debounce", type=float, default=0.5)
    parser.add_argument(
        "--terminate-on-collision",
        action="store_true",
        help="end the trial immediately after the first debounced collision",
    )
    parser.add_argument("--rotation-linear-threshold", type=float, default=0.03)
    parser.add_argument("--rotation-angular-threshold", type=float, default=0.05)
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument(
        "--goal-topic", default="/topoplan/planner_goal_reached"
    )
    parser.add_argument("--shortest-path-distance", type=float)
    parser.add_argument("--require-collision-free", action="store_true")
    args = parser.parse_known_args()[0]
    if args.require_collision_free and not args.collision_topic:
        parser.error("--require-collision-free requires --collision-topic")
    return args


if __name__ == "__main__":
    rospy.init_node("navigation_trial_evaluator", anonymous=True)
    TrialEvaluator(parse_args())
    rospy.spin()
