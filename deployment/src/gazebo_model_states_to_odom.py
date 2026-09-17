#!/usr/bin/env python3
"""Publish one Gazebo model state as nav_msgs/Odometry for simulation.

This is a ground-truth pose bridge, not wheel odometry. It is intended for
repeatable Gazebo experiments when the robot plugin does not publish /odom.
"""

import argparse
import sys

import rospy
import tf2_ros
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry


class GazeboModelStatesToOdom:
    def __init__(
        self,
        model_name,
        model_prefix,
        input_topic,
        output_topic,
        frame_id,
        child_frame_id,
        publish_tf,
    ):
        self.model_name = model_name
        self.model_prefix = model_prefix
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self.model_index = None
        self.missing_logged = False

        self.odom_pub = rospy.Publisher(output_topic, Odometry, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() if publish_tf else None
        self.subscriber = rospy.Subscriber(
            input_topic, ModelStates, self.callback, queue_size=1
        )
        rospy.loginfo(
            "Gazebo ground-truth odometry bridge: input=%s output=%s "
            "model=%s prefix=%s",
            input_topic,
            output_topic,
            model_name or "<auto>",
            model_prefix,
        )

    def resolve_model_index(self, names):
        if self.model_name:
            if self.model_name in names:
                return names.index(self.model_name)
            normalized = self.model_name.rstrip("/")
            for index, name in enumerate(names):
                if name.rstrip("/") == normalized:
                    return index

        candidates = [
            index
            for index, name in enumerate(names)
            if name.rstrip("/").startswith(self.model_prefix.rstrip("/"))
        ]
        if len(candidates) == 1:
            return candidates[0]

        if not self.missing_logged:
            self.missing_logged = True
            rospy.logerr(
                "无法唯一确定 Scout 模型。候选=%s；Gazebo 模型列表=%s。"
                "请使用 --model-name 指定准确名称。",
                [names[index] for index in candidates],
                list(names),
            )
        return None

    def callback(self, msg):
        if self.model_index is None:
            self.model_index = self.resolve_model_index(msg.name)
            if self.model_index is None:
                return
            rospy.loginfo("使用 Gazebo 模型: %s", msg.name[self.model_index])

        if self.model_index >= len(msg.pose) or self.model_index >= len(msg.twist):
            rospy.logerr_throttle(2.0, "ModelStates pose/twist 数组长度不一致")
            return

        stamp = rospy.Time.now()
        pose = msg.pose[self.model_index]
        twist = msg.twist[self.model_index]

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = pose
        odom.twist.twist = twist
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.frame_id
            transform.child_frame_id = self.child_frame_id
            transform.transform.translation.x = pose.position.x
            transform.transform.translation.y = pose.position.y
            transform.transform.translation.z = pose.position.z
            transform.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(transform)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Gazebo model's ground-truth state to /odom"
    )
    parser.add_argument("--model-name", default="")
    parser.add_argument("--model-prefix", default="scout")
    parser.add_argument("--input-topic", default="/gazebo/model_states")
    parser.add_argument("--output-topic", default="/odom")
    parser.add_argument("--frame-id", default="odom")
    parser.add_argument("--child-frame-id", default="base_link")
    parser.add_argument("--no-tf", action="store_true")
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


if __name__ == "__main__":
    rospy.init_node("gazebo_model_states_to_odom")
    args = parse_args()
    GazeboModelStatesToOdom(
        model_name=args.model_name,
        model_prefix=args.model_prefix,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        frame_id=args.frame_id,
        child_frame_id=args.child_frame_id,
        publish_tf=not args.no_tf,
    )
    rospy.spin()
