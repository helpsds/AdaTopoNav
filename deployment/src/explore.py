"""Goal-masked NoMaD exploration with the shared DA3 safety shield."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rospy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray

from deployment_bootstrap import VISUALNAV_ROOT
from nomad_inference import sample_nomad_actions
from remote_safety_client import RemoteSafetyRuntime
from safety_runtime import SafetyRuntime, add_safety_arguments
from topic_names import (
    IMAGE_TOPIC,
    NOMAD_TRAJECTORIES_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    SELECTED_TRAJECTORY_TOPIC,
    WAYPOINT_TOPIC,
)
from trajectory_candidates import build_candidate_batch, selected_controller_waypoint
from trajectory_visualization import make_trajectory_marker_arrays
from utils import load_model, msg_to_pil, transform_images


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG_PATH = DEPLOYMENT_ROOT / "config/robot.yaml"
MODEL_CONFIG_PATH = DEPLOYMENT_ROOT / "config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
    robot_config = yaml.safe_load(handle)
MAX_V = float(robot_config["max_v"])
MAX_W = float(robot_config["max_w"])
RATE = float(robot_config["frame_rate"])

context_queue: list[PILImage.Image] = []
context_size: int | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg: Image) -> None:
    obs_img = msg_to_pil(msg)
    if context_size is None:
        return
    if len(context_queue) < context_size + 1:
        context_queue.append(obs_img)
    else:
        context_queue.pop(0)
        context_queue.append(obs_img)


def main(args: argparse.Namespace) -> None:
    global context_size

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as handle:
        model_paths = yaml.safe_load(handle)
    configured_model_config = Path(model_paths[args.model]["config_path"])
    copied_config = (Path(__file__).resolve().parent / configured_model_config).resolve()
    model_config_path = Path(args.nomad_config).expanduser().resolve() if args.nomad_config else copied_config
    if not model_config_path.exists() and args.model == "nomad":
        model_config_path = VISUALNAV_ROOT / "train/config/nomad.yaml"
    with model_config_path.open("r", encoding="utf-8") as handle:
        model_params = yaml.safe_load(handle)
    if model_params["model_type"] != "nomad":
        raise ValueError("this safety deployment explore.py supports model_type=nomad only")
    context_size = int(model_params["context_size"])
    configured_checkpoint = Path(model_paths[args.model]["ckpt_path"])
    copied_checkpoint = (Path(__file__).resolve().parent / configured_checkpoint).resolve()
    checkpoint_path = Path(args.nomad_checkpoint).expanduser().resolve() if args.nomad_checkpoint else copied_checkpoint
    if not checkpoint_path.exists() and args.model == "nomad":
        checkpoint_path = VISUALNAV_ROOT / "deployment/model_weights/nomad.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model weights not found at {checkpoint_path}")
    print(f"Loading NoMaD from {checkpoint_path}")
    model = load_model(checkpoint_path, model_params, device).to(device).eval()
    num_diffusion_iters = int(model_params["num_diffusion_iters"])
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    safety = RemoteSafetyRuntime(args) if args.safety_server else SafetyRuntime(args, device)

    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    trajectories_pub = rospy.Publisher(NOMAD_TRAJECTORIES_TOPIC, MarkerArray, queue_size=1)
    selected_pub = rospy.Publisher(SELECTED_TRAJECTORY_TOPIC, MarkerArray, queue_size=1)
    print("Registered with ROS master. Waiting for image observations...")

    while not rospy.is_shutdown():
        if len(context_queue) > context_size:
            obs_images = transform_images(
                context_queue, model_params["image_size"], center_crop=False
            ).to(device)
            fake_goal = torch.randn((1, 3, *model_params["image_size"]), device=device)
            mask = torch.ones(1, dtype=torch.long, device=device)
            with torch.inference_mode():
                obs_cond = model(
                    "vision_encoder",
                    obs_img=obs_images,
                    goal_img=fake_goal,
                    input_goal_mask=mask,
                )
            started = time.time()
            nomad_actions = sample_nomad_actions(
                model,
                obs_cond,
                noise_scheduler,
                num_diffusion_iters,
                args.num_samples,
                int(model_params["len_traj_pred"]),
            )
            selected_idx = 0
            selected_actions = nomad_actions
            selected_metadata = build_candidate_batch(
                nomad_actions,
                num_waypoints=args.student_num_waypoints,
                append_manual=False,
                platform_config=args.platform_config,
            ).metadata
            prediction = None
            result = None

            if safety.enabled:
                try:
                    step = safety.step(
                        context_queue,
                        nomad_actions,
                        max_v_mps=MAX_V,
                        frame_rate_hz=RATE,
                        normalize_controller_waypoint=bool(model_params["normalize"]),
                    )
                    selected_actions = step.candidates.actions
                    selected_metadata = step.candidates.metadata
                    selected_idx = step.result.selected_index
                    prediction = step.prediction
                    result = step.result
                    chosen_waypoint = safety.controller_waypoint(
                        step,
                        args.waypoint,
                        bool(model_params["normalize"]),
                        MAX_V,
                        RATE,
                    )
                    safety.log(step, inference_seconds=float(time.time() - started), node="explore")
                    print(
                        f"mode={result.mode.value} "
                        f"idx0={result.idx0_raw_risk:.3f}/{result.idx0_filtered_risk:.3f} "
                        f"selected={selected_idx} reason={result.reason}"
                    )
                except Exception as error:
                    safety.fail_closed()
                    failure = safety.failure_candidates(nomad_actions)
                    selected_actions = failure.actions
                    selected_metadata = failure.metadata
                    selected_idx = -1
                    chosen_waypoint = np.zeros((2,), dtype=np.float32)
                    safety.log_error(error, node="explore")
                    rospy.logerr(f"Safety inference failed closed: {type(error).__name__}: {error}")
            else:
                chosen_waypoint = selected_controller_waypoint(
                    nomad_actions[0],
                    args.waypoint,
                    bool(model_params["normalize"]),
                    MAX_V,
                    RATE,
                )

            sampled = Float32MultiArray()
            sampled.data = np.concatenate(
                [np.asarray([selected_idx], dtype=np.float32), selected_actions.reshape(-1)]
            )
            sampled_actions_pub.publish(sampled)
            if not args.disable_trajectory_viz:
                all_markers, selected_markers = make_trajectory_marker_arrays(
                    selected_actions,
                    selected_metadata,
                    selected_idx,
                    args.waypoint,
                    args.trajectory_viz_frame,
                    prediction=prediction,
                    result=result,
                    nomad_safe_threshold=args.nomad_safe_threshold,
                    manual_safe_threshold=args.manual_safe_threshold,
                )
                trajectories_pub.publish(all_markers)
                selected_pub.publish(selected_markers)

            waypoint = Float32MultiArray()
            waypoint.data = np.asarray(chosen_waypoint, dtype=np.float32)
            waypoint_pub.publish(waypoint)
        rate.sleep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NoMaD exploration with DA3 safety")
    parser.add_argument("--model", "-m", default="nomad")
    parser.add_argument("--waypoint", "-w", type=int, default=2)
    parser.add_argument("--num-samples", "-n", type=int, default=16)
    parser.add_argument("--nomad-config", default=None, help="Override NoMaD model YAML.")
    parser.add_argument("--nomad-checkpoint", default=None, help="Override NoMaD .pth checkpoint.")
    add_safety_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(f"Using {device}")
    main(arguments)
