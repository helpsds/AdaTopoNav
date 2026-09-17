
import argparse
import os
import shutil

# Add the repository-local train and diffusion_policy packages to sys.path
# before importing deployment utilities. This keeps the script runnable
# directly from deployment/src without requiring a manual PYTHONPATH export.
from deployment_bootstrap import VISUALNAV_ROOT  # noqa: F401

from utils import msg_to_pil
import time

# ROS
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy

# IMAGE_TOPIC = "/camera_node/camera/image"

DEFAULT_IMAGE_TOPIC = "/camera/color/image_raw"
#仿真相机话题：/camera/color/image_raw
TOPOMAP_IMAGES_DIR = "../topomaps/images"
obs_img = None


def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def callback_obs(msg: Image):
    global obs_img
    obs_img = msg_to_pil(msg)


def callback_joy(msg: Joy):
    if msg.buttons[0]:
        rospy.signal_shutdown("shutdown")


def main(args: argparse.Namespace):
    global obs_img
    rospy.init_node("CREATE_TOPOMAP", anonymous=False)
    image_curr_msg = rospy.Subscriber(
        args.image_topic, Image, callback_obs, queue_size=1)
    subgoals_pub = rospy.Publisher(
        "/subgoals", Image, queue_size=1)
    joy_sub = rospy.Subscriber("joy", Joy, callback_joy)

    topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    if not os.path.isdir(topomap_name_dir):
        os.makedirs(topomap_name_dir)
    else:
        print(f"{topomap_name_dir} already exists. Removing previous images...")
        remove_files_in_dir(topomap_name_dir)


    assert args.dt > 0, "dt must be positive"
    rate = rospy.Rate(1/args.dt)
    print(f"Registered with master node. Waiting for images on {args.image_topic}...")
    i = 0
    start_time = float("inf")
    while not rospy.is_shutdown():
        if obs_img is not None:
            obs_img.save(os.path.join(topomap_name_dir, f"{i}.png"))
            print("published image", i)
            i += 1
            rate.sleep()
            start_time = time.time()
            obs_img = None
        if time.time() - start_time > 2 * args.dt:
            print(f"Topic {args.image_topic} not publishing anymore. Shutting down...")
            rospy.signal_shutdown("shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to generate a topomap from a ROS image topic"
    )
    parser.add_argument(
        "--image-topic",
        default=DEFAULT_IMAGE_TOPIC,
        help=(
            "image topic to sample; use a remapped bag-only topic when Gazebo "
            "is publishing the live camera at the same time"
        ),
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.,
        type=float,
        help="time between sampled images in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    main(args)
