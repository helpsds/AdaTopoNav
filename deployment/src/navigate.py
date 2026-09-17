# """Topological NoMaD navigation with a stateful DA3 safety shield."""

# from __future__ import annotations

# import argparse
# import os
# import time
# from pathlib import Path

# import numpy as np
# import rospy
# import torch
# import yaml
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from PIL import Image as PILImage
# from sensor_msgs.msg import Image
# from std_msgs.msg import Bool, Float32MultiArray
# from visualization_msgs.msg import MarkerArray

# from deployment_bootstrap import VISUALNAV_ROOT
# from navigation_state import ClosestNodeTracker
# from nomad_inference import sample_nomad_actions
# from remote_safety_client import RemoteSafetyRuntime
# from safety_runtime import SafetyRuntime, add_safety_arguments
# from safety_selection_policy import SafetyMode
# from topic_names import (
#     IMAGE_TOPIC,
#     NOMAD_TRAJECTORIES_TOPIC,
#     SAMPLED_ACTIONS_TOPIC,
#     SELECTED_TRAJECTORY_TOPIC,
#     WAYPOINT_TOPIC,
# )
# from trajectory_candidates import build_candidate_batch, selected_controller_waypoint
# from trajectory_visualization import make_trajectory_marker_arrays
# from utils import load_model, msg_to_pil, to_numpy, transform_images


# DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
# TOPOMAP_IMAGES_DIR = DEPLOYMENT_ROOT / "topomaps/images"
# ROBOT_CONFIG_PATH = DEPLOYMENT_ROOT / "config/robot.yaml"
# MODEL_CONFIG_PATH = DEPLOYMENT_ROOT / "config/models.yaml"
# with open(ROBOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
#     robot_config = yaml.safe_load(handle)
# MAX_V = float(robot_config["max_v"])
# MAX_W = float(robot_config["max_w"])
# RATE = float(robot_config["frame_rate"])

# context_queue: list[PILImage.Image] = []
# context_size: int | None = None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


# def callback_obs(msg: Image) -> None:
#     obs_img = msg_to_pil(msg)
#     if context_size is None:
#         return
#     if len(context_queue) < context_size + 1:
#         context_queue.append(obs_img)
#     else:
#         context_queue.pop(0)
#         context_queue.append(obs_img)


# def publish_visualization(
#     args: argparse.Namespace,
#     all_pub,
#     selected_pub,
#     actions: np.ndarray,
#     metadata,
#     selected_idx: int,
#     prediction=None,
#     result=None,
# ) -> None:
#     if args.disable_trajectory_viz:
#         return
#     all_markers, selected_markers = make_trajectory_marker_arrays(
#         actions,
#         metadata,
#         selected_idx,
#         args.waypoint,
#         args.trajectory_viz_frame,
#         prediction=prediction,
#         result=result,
#         nomad_safe_threshold=args.nomad_safe_threshold,
#         manual_safe_threshold=args.manual_safe_threshold,
#     )
#     all_pub.publish(all_markers)
#     selected_pub.publish(selected_markers)


# def main(args: argparse.Namespace) -> None:
#     global context_size

#     with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as handle:
#         model_paths = yaml.safe_load(handle)
#     configured_model_config = Path(model_paths[args.model]["config_path"])
#     copied_config = (Path(__file__).resolve().parent / configured_model_config).resolve()
#     model_config_path = Path(args.nomad_config).expanduser().resolve() if args.nomad_config else copied_config
#     if not model_config_path.exists() and args.model == "nomad":
#         model_config_path = VISUALNAV_ROOT / "train/config/nomad.yaml"
#     with model_config_path.open("r", encoding="utf-8") as handle:
#         model_params = yaml.safe_load(handle)
#     context_size = int(model_params["context_size"])
#     configured_checkpoint = Path(model_paths[args.model]["ckpt_path"])
#     copied_checkpoint = (Path(__file__).resolve().parent / configured_checkpoint).resolve()
#     checkpoint_path = Path(args.nomad_checkpoint).expanduser().resolve() if args.nomad_checkpoint else copied_checkpoint
#     if not checkpoint_path.exists() and args.model == "nomad":
#         checkpoint_path = VISUALNAV_ROOT / "deployment/model_weights/nomad.pth"
#     if not checkpoint_path.exists():
#         raise FileNotFoundError(f"Model weights not found at {checkpoint_path}")
#     print(f"Loading NoMaD from {checkpoint_path}")
#     model = load_model(checkpoint_path, model_params, device).to(device).eval()

#     topomap_dir = os.path.join(str(TOPOMAP_IMAGES_DIR), args.dir)
#     topomap_filenames = sorted(os.listdir(topomap_dir), key=lambda value: int(value.split(".")[0]))
#     topomap = [PILImage.open(os.path.join(topomap_dir, name)).convert("RGB") for name in topomap_filenames]
#     if not topomap:
#         raise ValueError(f"empty topomap: {topomap_dir}")
#     if not -1 <= args.goal_node < len(topomap):
#         raise ValueError("invalid goal index")
#     goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node
#     node_tracker = ClosestNodeTracker(
#         confirmed=0,
#         confirm_frames=args.closest_node_confirm_frames,
#         max_backtrack=args.max_node_backtrack,
#     )
#     locked_subgoal: int | None = None

#     if model_params["model_type"] != "nomad":
#         if args.safety_checkpoint:
#             raise ValueError("DA3 safety deployment currently supports NoMaD only")
#         raise ValueError("this safety deployment navigate.py supports model_type=nomad only")
#     num_diffusion_iters = int(model_params["num_diffusion_iters"])
#     noise_scheduler = DDPMScheduler(
#         num_train_timesteps=num_diffusion_iters,
#         beta_schedule="squaredcos_cap_v2",
#         clip_sample=True,
#         prediction_type="epsilon",
#     )
#     safety = RemoteSafetyRuntime(args) if args.safety_server else SafetyRuntime(args, device)

#     rospy.init_node("EXPLORATION", anonymous=False)
#     rate = rospy.Rate(RATE)
#     rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
#     waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
#     sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
#     goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
#     trajectories_pub = rospy.Publisher(NOMAD_TRAJECTORIES_TOPIC, MarkerArray, queue_size=1)
#     selected_pub = rospy.Publisher(SELECTED_TRAJECTORY_TOPIC, MarkerArray, queue_size=1)
#     print("Registered with ROS master. Waiting for image observations...")

#     while not rospy.is_shutdown():
#         chosen_waypoint = np.zeros((2,), dtype=np.float32)
#         if len(context_queue) > context_size:
#             obs_images = transform_images(
#                 context_queue, model_params["image_size"], center_crop=False
#             ).to(device)
#             mask = torch.zeros(1, dtype=torch.long, device=device)

#             start = max(node_tracker.confirmed - args.radius, 0)
#             end = min(node_tracker.confirmed + args.radius + 1, goal_node)
#             localization_nodes = list(range(start, end + 1))
#             goal_nodes = list(localization_nodes)
#             policy_mode_before = (
#                 safety.policy.mode if safety.enabled and safety.policy is not None else SafetyMode.NORMAL
#             )
#             if policy_mode_before != SafetyMode.NORMAL and locked_subgoal is not None:
#                 if locked_subgoal not in goal_nodes:
#                     goal_nodes.append(locked_subgoal)

#             goal_images = torch.cat(
#                 [
#                     transform_images(topomap[index], model_params["image_size"], center_crop=False)
#                     .to(device)
#                     for index in goal_nodes
#                 ],
#                 dim=0,
#             )
#             with torch.inference_mode():
#                 obsgoal_cond = model(
#                     "vision_encoder",
#                     obs_img=obs_images.repeat(len(goal_images), 1, 1, 1),
#                     goal_img=goal_images,
#                     input_goal_mask=mask.repeat(len(goal_images)),
#                 )
#                 dists = np.asarray(
#                     to_numpy(model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())
#                 )
#             local_min = int(np.argmin(dists[: len(localization_nodes)]))
#             closest_node = node_tracker.update(localization_nodes[local_min])
#             closest_cond_index = goal_nodes.index(closest_node)
#             advance = int(float(dists[closest_cond_index]) < float(args.close_threshold))
#             normal_subgoal = min(closest_node + advance, goal_node)
#             if policy_mode_before != SafetyMode.NORMAL and locked_subgoal is not None:
#                 subgoal_node = locked_subgoal
#             else:
#                 subgoal_node = normal_subgoal
#             if subgoal_node not in goal_nodes:
#                 goal_nodes.append(subgoal_node)
#                 extra_goal = transform_images(
#                     topomap[subgoal_node], model_params["image_size"], center_crop=False
#                 ).to(device)
#                 with torch.inference_mode():
#                     extra_cond = model(
#                         "vision_encoder",
#                         obs_img=obs_images,
#                         goal_img=extra_goal,
#                         input_goal_mask=mask,
#                     )
#                 obsgoal_cond = torch.cat([obsgoal_cond, extra_cond], dim=0)
#             obs_cond = obsgoal_cond[goal_nodes.index(subgoal_node)].unsqueeze(0)

#             started = time.time()
#             nomad_actions = sample_nomad_actions(
#                 model,
#                 obs_cond,
#                 noise_scheduler,
#                 num_diffusion_iters,
#                 args.num_samples,
#                 int(model_params["len_traj_pred"]),
#             )
#             selected_idx = 0
#             selected_actions = nomad_actions
#             selected_metadata = build_candidate_batch(
#                 nomad_actions,
#                 num_waypoints=args.student_num_waypoints,
#                 append_manual=False,
#                 platform_config=args.platform_config,
#             ).metadata
#             prediction = None
#             result = None

#             if safety.enabled:
#                 try:
#                     step = safety.step(
#                         context_queue,
#                         nomad_actions,
#                         max_v_mps=MAX_V,
#                         frame_rate_hz=RATE,
#                         normalize_controller_waypoint=bool(model_params["normalize"]),
#                     )
#                     selected_actions = step.candidates.actions
#                     selected_metadata = step.candidates.metadata
#                     selected_idx = step.result.selected_index
#                     prediction = step.prediction
#                     result = step.result
#                     chosen_waypoint = safety.controller_waypoint(
#                         step,
#                         args.waypoint,
#                         bool(model_params["normalize"]),
#                         MAX_V,
#                         RATE,
#                     )
#                     if policy_mode_before == SafetyMode.NORMAL and result.mode != SafetyMode.NORMAL:
#                         locked_subgoal = subgoal_node
#                     elif result.mode == SafetyMode.NORMAL:
#                         locked_subgoal = None
#                     safety.log(
#                         step,
#                         closest_node=int(closest_node),
#                         subgoal=int(subgoal_node),
#                         locked_subgoal=(int(locked_subgoal) if locked_subgoal is not None else None),
#                         inference_seconds=float(time.time() - started),
#                     )
#                     print(
#                         f"closest={closest_node} subgoal={subgoal_node} mode={result.mode.value} "
#                         f"idx0={result.idx0_raw_risk:.3f}/{result.idx0_filtered_risk:.3f} "
#                         f"selected={selected_idx} reason={result.reason}"
#                     )
#                 except Exception as error:
#                     safety.fail_closed()
#                     failure = safety.failure_candidates(nomad_actions)
#                     selected_actions = failure.actions
#                     selected_metadata = failure.metadata
#                     selected_idx = -1
#                     chosen_waypoint = np.zeros((2,), dtype=np.float32)
#                     locked_subgoal = subgoal_node
#                     safety.log_error(
#                         error,
#                         closest_node=int(closest_node),
#                         subgoal=int(subgoal_node),
#                     )
#                     rospy.logerr(f"Safety inference failed closed: {type(error).__name__}: {error}")
#             else:
#                 chosen_waypoint = selected_controller_waypoint(
#                     nomad_actions[0],
#                     args.waypoint,
#                     bool(model_params["normalize"]),
#                     MAX_V,
#                     RATE,
#                 )
#                 print(f"closest={closest_node} subgoal={subgoal_node} selected=0 safety=disabled")

#             sampled = Float32MultiArray()
#             sampled.data = np.concatenate(
#                 [np.asarray([selected_idx], dtype=np.float32), selected_actions.reshape(-1)]
#             )
#             sampled_actions_pub.publish(sampled)
#             publish_visualization(
#                 args,
#                 trajectories_pub,
#                 selected_pub,
#                 selected_actions,
#                 selected_metadata,
#                 selected_idx,
#                 prediction=prediction,
#                 result=result,
#             )

#         mode_is_normal = not safety.enabled or (
#             safety.policy is not None and safety.policy.mode == SafetyMode.NORMAL
#         )
#         reached_goal = bool(node_tracker.confirmed == goal_node and mode_is_normal)
#         if reached_goal:
#             chosen_waypoint = np.zeros_like(chosen_waypoint)
#         waypoint_msg = Float32MultiArray()
#         waypoint_msg.data = np.asarray(chosen_waypoint, dtype=np.float32)
#         waypoint_pub.publish(waypoint_msg)
#         goal_pub.publish(reached_goal)
#         if reached_goal:
#             print("Reached goal! Publishing zero waypoint.")
#         rate.sleep()


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="NoMaD topological navigation with DA3 safety")
#     parser.add_argument("--model", "-m", default="nomad")
#     parser.add_argument("--waypoint", "-w", type=int, default=2)
#     parser.add_argument("--dir", "-d", default="topomap")
#     parser.add_argument("--goal-node", "-g", type=int, default=-1)
#     parser.add_argument("--close-threshold", "-t", type=float, default=3.0)
#     parser.add_argument("--radius", "-r", type=int, default=4)
#     parser.add_argument("--num-samples", "-n", type=int, default=16)
#     parser.add_argument("--nomad-config", default=None, help="Override NoMaD model YAML.")
#     parser.add_argument("--nomad-checkpoint", default=None, help="Override NoMaD .pth checkpoint.")
#     parser.add_argument("--closest-node-confirm-frames", type=int, default=2)
#     parser.add_argument("--max-node-backtrack", type=int, default=1)
#     add_safety_arguments(parser)
#     return parser.parse_args()


# if __name__ == "__main__":
#     arguments = parse_args()
#     print(f"Using {device}")
#     main(arguments)


import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

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


# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    # global obs_img 
    # print(f"DEBUG: 收到图像了！序列号: {msg.header.seq}")
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


def main(args: argparse.Namespace):
    global context_size

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

    
     # load topomap
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

     # ROS
    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    image_curr_msg = rospy.Subscriber(
        IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(
        WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)  
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)

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

        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                goal_image = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) for g_img in topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
                    for k in noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
                print("published sampled actions")
                sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] 
                chosen_waypoint = naction[args.waypoint]
            else:
                start = max(closest_node - args.radius, 0)
                end = min(closest_node + args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    goal_data = transform_images(sg_img, model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                if distances[min_dist_idx] > args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                    closest_node = min(start + min_dist_idx + 1, goal_node)
        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)
        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("Reached goal! Stopping...")
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
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
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
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)

