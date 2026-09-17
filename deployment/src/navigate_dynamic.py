import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import yaml
import argparse
import time
import math
# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action
from PIL import Image as PILImage

# UTILS
from topic_names import (IMAGE_TOPIC,
                         WAYPOINT_TOPIC,
                         SAMPLED_ACTIONS_TOPIC)

# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
context_queue = []
context_size = None  
subgoal = []
obs_img = None
dynamic_goal_img = None  
planner_goal_reached = False

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def callback_obs(msg):
    global obs_img 
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

# 🔴 新增：接收舰长目标的耳朵
def callback_dynamic_goal(msg):
    global dynamic_goal_img
    try:
        # 硬解二进制图像数据，绕过 cv_bridge
        img_1d = np.frombuffer(msg.data, dtype=np.uint8)
        img_array = img_1d.reshape(msg.height, msg.width, -1)
        if msg.encoding == "bgr8":
            img_array = img_array[:, :, ::-1]
        dynamic_goal_img = PILImage.fromarray(img_array)
    except Exception as e:
        rospy.logerr_throttle(2, f"解析舰长目标图像失败: {e}")

def callback_planner_goal(msg):
    global planner_goal_reached
    # A previous planner process may still have a latched True while a new
    # trial is starting. A success result is only meaningful after this
    # navigation node has received a target image for the current trial.
    if bool(msg.data) and dynamic_goal_img is None:
        rospy.logwarn("忽略本轮目标图像就绪前收到的旧终点成功信号")
        return
    planner_goal_reached = bool(msg.data)

def main(args: argparse.Namespace):
    global context_size, dynamic_goal_img, planner_goal_reached

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    
    model = load_model(
        ckpth_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()
    
    # load topomap (依然加载本地地图作为备用或静态导航使用)
    topomap_filenames = sorted(os.listdir(os.path.join(
        TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

    # ROS 节点初始化
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
        
    # 🔴 新增：挂载监听通道
    target_image_sub = rospy.Subscriber(
        "/topoplan/target_image", Image, callback_dynamic_goal, queue_size=1)
    planner_goal_sub = rospy.Subscriber(
        "/topoplan/planner_goal_reached", Bool, callback_planner_goal, queue_size=1)
        
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(
        SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher(
        "/topoplan/reached_goal", Bool, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    # navigation loop
    while not rospy.is_shutdown():
        if planner_goal_reached:
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = np.zeros(4)
            waypoint_pub.publish(waypoint_msg)
            goal_pub.publish(True)
            rospy.signal_shutdown("Goal reached with metric confirmation")
            break
        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                # 🔴 拦截逻辑：如果有动态目标，就无视本地文件夹的图
                if dynamic_goal_img is not None:
                    current_goals = [dynamic_goal_img]
                    start = closest_node # 维持坐标系不变
                else:
                    start = max(closest_node - args.radius, 0)
                    end = min(closest_node + args.radius + 1, goal_node)
                    current_goals = topomap[start:end + 1]

                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in current_goals]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                
                # 只有在使用静态地图时才打印 closest node，避免动态图模式下干扰日志
                if dynamic_goal_img is None:
                    print("closest node:", closest_node)
                    
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
                    for k in noise_scheduler.timesteps[:]:
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    # print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                sampled_actions_pub.publish(sampled_actions_msg)
                
                naction = naction[0] 
                chosen_waypoint = naction[args.waypoint]
                
            else:
                # GNM 分支逻辑
                # 🔴 拦截逻辑 (GNM分支)
                if dynamic_goal_img is not None:
                    current_goals = [dynamic_goal_img]
                    start = closest_node
                else:
                    start = max(closest_node - args.radius, 0)
                    end = min(closest_node + args.radius + 1, goal_node)
                    current_goals = topomap[start:end + 1]

                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                
                # 🔴 注意这里遍历的是 current_goals
                for i, sg_img in enumerate(current_goals): 
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    goal_data = transform_images(sg_img, model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                
                min_dist_idx = np.argmin(distances)
                
                if distances[min_dist_idx] > args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                    closest_node = min(start + min_dist_idx + 1, goal_node)
                    
        # RECOVERY MODE / OUTPUT
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  

        if not args.disable_waypoint_modulation:
            angle_to_goal = math.atan2(chosen_waypoint[1], chosen_waypoint[0])
            if abs(angle_to_goal) > args.heading_threshold:
                chosen_waypoint[0] = 0.0
                chosen_waypoint[1] = (
                    np.sign(chosen_waypoint[1]) * args.turn_waypoint
                )
            else:
                speed_factor = max(0.0, math.cos(angle_to_goal))
                chosen_waypoint[0] *= speed_factor ** 2
                chosen_waypoint[1] *= args.lateral_gain
            
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)
        
        # 🔴 动态目标模式下，小车永不主动停车，由全局舰长控制
        if dynamic_goal_img is not None:
            reached_goal = False 
        else:
            reached_goal = closest_node == goal_node
            
        goal_pub.publish(reached_goal)
        
        if reached_goal:
            rospy.loginfo("🏁 已检测到抵达终点，正在停机...")
            # 发送一个零速度确保不滑行
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = np.zeros(4)
            waypoint_pub.publish(waypoint_msg)
            
            # 停止 ROS 节点并退出 Python
            rospy.signal_shutdown("Goal reached")
            break # 跳出 while 循环
            
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, 
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed used for diffusion action sampling",
    )
    parser.add_argument(
        "--disable-waypoint-modulation",
        action="store_true",
        help="Publish the raw model waypoint; intended for controlled ablation.",
    )
    parser.add_argument("--heading-threshold", type=float, default=0.44)
    parser.add_argument("--turn-waypoint", type=float, default=0.6)
    parser.add_argument("--lateral-gain", type=float, default=1.5)
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
