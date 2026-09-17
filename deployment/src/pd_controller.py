import argparse
import time
import numpy as np
import yaml
from typing import Tuple

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1/robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None
collision_active = False
recovery_started_at = None
last_recovery_end = float("-inf")
recovery_turn_direction = 1.0

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      

def waypoint_controller(waypoint: np.ndarray) -> Tuple[float, float]:
	"""Convert a local waypoint to clipped linear and angular velocities.

	This is a kinematic proportional mapping, not a PD controller: no
	derivative of tracking error is estimated.
	"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/DT		
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*DT)
	else:
		v = dx / DT
		w = np.arctan(dy/dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	print("seting waypoint")
	waypoint.set(waypoint_msg.data)
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	reached_goal = reached_goal_msg.data


def callback_collision(collision_msg: Bool):
	"""Start one recovery maneuver on each False-to-True contact edge."""
	global collision_active, recovery_started_at
	global last_recovery_end, recovery_turn_direction

	active = bool(collision_msg.data)
	if not active:
		collision_active = False
		return
	if collision_active:
		return
	collision_active = True

	now = time.monotonic()
	if recovery_started_at is not None:
		return
	if now - last_recovery_end < main.args.recovery_cooldown:
		return
	recovery_turn_direction *= -1.0
	recovery_started_at = now
	rospy.logwarn(
		"COLLISION_RECOVERY start: reverse %.2fs, turn %.2fs",
		main.args.recovery_reverse_seconds,
		main.args.recovery_turn_seconds,
	)


def main(args):
	global vel_msg, reverse_mode, recovery_started_at, last_recovery_end
	main.args = args
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	if args.collision_topic:
		rospy.Subscriber(args.collision_topic, Bool, callback_collision, queue_size=10)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	print("Registered with master node. Waiting for waypoints...")
	while not rospy.is_shutdown():
		vel_msg = Twist()
		if reached_goal:
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			return
		elif recovery_started_at is not None:
			elapsed = time.monotonic() - recovery_started_at
			if elapsed < args.recovery_reverse_seconds:
				vel_msg.linear.x = -abs(args.recovery_linear_speed)
				vel_msg.angular.z = 0.0
			elif elapsed < args.recovery_reverse_seconds + args.recovery_turn_seconds:
				vel_msg.linear.x = 0.0
				vel_msg.angular.z = (
					recovery_turn_direction * abs(args.recovery_angular_speed)
				)
			else:
				recovery_started_at = None
				last_recovery_end = time.monotonic()
				rospy.loginfo("COLLISION_RECOVERY complete; resuming waypoint tracking")
			vel_out.publish(vel_msg)
		elif waypoint.is_valid(verbose=True):
			v, w = waypoint_controller(waypoint.get())
			if reverse_mode:
				v *= -1
			vel_msg.linear.x = v
			vel_msg.angular.z = w
			print(f"publishing new vel: {v}, {w}")
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="Waypoint controller with optional collision recovery"
	)
	parser.add_argument(
		"--collision-topic", default="/scout/has_obstacle_contact"
	)
	parser.add_argument("--recovery-reverse-seconds", type=float, default=0.8)
	parser.add_argument("--recovery-turn-seconds", type=float, default=1.2)
	parser.add_argument("--recovery-linear-speed", type=float, default=0.12)
	parser.add_argument("--recovery-angular-speed", type=float, default=0.35)
	parser.add_argument("--recovery-cooldown", type=float, default=1.0)
	main(parser.parse_args(rospy.myargv()[1:]))
