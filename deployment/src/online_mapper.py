#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import os
import argparse
import re
import sys
from PIL import Image as PILImage
import message_filters

class OnlineTopologicalMapper:
    def __init__(
        self,
        map_name="L_map",
        snap_thresh=0.65,
        dense_dist=0.65,
        sparse_dist=1.20,
        parent_radius=1.50,
        sync_slop=0.10,
        map_root=None,
        image_topic="/camera/color/image_raw",
        odom_topic="/odom",
        startup_timeout=8.0,
    ):
        rospy.init_node('online_topomap_builder', anonymous=True)
        self.map_name = map_name 
        
        rospy.loginfo(f"正在加载 DINOv2 模型... 当前建图名称: {self.map_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.saved_features = []
        self.saved_images = []
        self.saved_poses = []
        self.edges = [] 

        self.current_pose = None   
        self.active_node = None  
        
        # Empirical mapping parameters; expose them as CLI options so they can
        # be reported and swept in experiments.
        self.sparse_dist = sparse_dist
        self.dense_dist = dense_dist
        self.snap_thresh = snap_thresh
        self.parent_radius = parent_radius
        self.image_topic = image_topic
        self.odom_topic = odom_topic
        self.received_synchronized_frame = False
        self.map_root = os.path.abspath(
            os.path.expanduser(
                map_root or os.path.join(os.path.dirname(__file__), "..", "topomaps")
            )
        )

        # Mapping decisions must use a pose associated with the same camera
        # timestamp. ApproximateTimeSynchronizer is used because most RGB and
        # odometry drivers do not publish at identical rates.
        self.image_sub = message_filters.Subscriber(
            self.image_topic, Image, queue_size=5
        )
        self.odom_sub = message_filters.Subscriber(
            self.odom_topic, Odometry, queue_size=20
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.odom_sub],
            queue_size=20,
            slop=sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synchronized_callback)
        self.startup_timer = rospy.Timer(
            rospy.Duration(startup_timeout),
            self.check_input_topics,
            oneshot=True,
        )
        rospy.on_shutdown(self.save_map)

    def check_input_topics(self, _event):
        if self.received_synchronized_frame:
            return
        published = dict(rospy.get_published_topics())
        missing = [
            topic
            for topic in (self.image_topic, self.odom_topic)
            if topic not in published
        ]
        if missing:
            rospy.logerr(
                "无法开始建图：缺少 ROS topic: %s。当前算法需要带时间戳的"
                " RGB 与 odometry；请先启动里程计发布节点，或通过"
                " --odom-topic 指定实际 nav_msgs/Odometry topic。",
                ", ".join(missing),
            )
        else:
            rospy.logerr(
                "已发现图像和里程计 topic，但 %.2f 秒内没有同步帧。"
                "请检查两者 header.stamp，并适当增大 --sync-slop。",
                self.sync.slop,
            )

    def extract_feature(self, img_array):
        img_tensor = self.transform(img_array).unsqueeze(0).to(self.device)
        with torch.no_grad(): return self.feature_extractor(img_tensor).flatten()

    def synchronized_callback(self, image_msg, odom_msg):
        self.received_synchronized_frame = True
        pos = odom_msg.pose.pose.position
        self.current_pose = np.array([pos.x, pos.y], dtype=np.float64)
        
        try:
            img_1d = np.frombuffer(image_msg.data, dtype=np.uint8)
            img_array = img_1d.reshape(image_msg.height, image_msg.width, -1)
            if image_msg.encoding == "bgr8":
                img_array = img_array[:, :, ::-1]
        except (ValueError, TypeError) as exc:
            rospy.logwarn_throttle(2.0, f"无法解析相机图像: {exc}")
            return

        if self.active_node is None:
            feature = self.extract_feature(img_array)
            self.add_node(feature, img_array, self.current_pose, reason="起点")
            self.active_node = 0
            return

        poses_array = np.array(self.saved_poses)
        dists = np.linalg.norm(poses_array - self.current_pose, axis=1)
        min_dist = np.min(dists)
        closest_idx = np.argmin(dists)

        # Spatial re-entry lock: update the active node without map growth.
        if min_dist < self.snap_thresh:
            if self.active_node != closest_idx:
                self.active_node = closest_idx
                rospy.logwarn(f"🔙 回溯吸附：定位于 [{closest_idx}]，暂停建图！")
            return 

        # State-dependent spacing. This distinguishes extension from the
        # newest node from departure at a historical node; it does not
        # explicitly classify straight corridors.
        active_pose = self.saved_poses[self.active_node]
        dist_to_active = np.linalg.norm(self.current_pose - active_pose)
        new_idx = len(self.saved_features)

        is_branching = (self.active_node != new_idx - 1)
        
        required_dist = self.dense_dist if is_branching else self.sparse_dist

        if dist_to_active >= required_dist:
            feature = self.extract_feature(img_array)
            
            if is_branching:
                valid_parents = np.where(dists < self.parent_radius)[0]
                # Prefer the physically nearest eligible historical node.
                # Selecting the smallest node ID can create a non-traversable
                # shortcut when several old branches fall inside the radius.
                best_parent = (
                    int(valid_parents[np.argmin(dists[valid_parents])])
                    if len(valid_parents) > 0
                    else self.active_node
                )
                action_str = f"🛣️ 开辟新岔路 (锚定 {best_parent})"
            else:
                best_parent = self.active_node
                action_str = "顺延直行"
                
            self.edges.append((best_parent, new_idx))
            self.add_node(feature, img_array, self.current_pose, reason=action_str)
            self.active_node = new_idx

    def add_node(self, feature, image, pose, reason):
        node_id = len(self.saved_features)
        self.saved_features.append(feature.cpu())
        self.saved_images.append(image)
        self.saved_poses.append(pose)
        rospy.loginfo(f"📸 节点 [{node_id}] | {reason}")

    def save_map(self):
        if not self.saved_features: return
        base_dir = self.map_root
        os.makedirs(base_dir, exist_ok=True)
        torch.save(torch.stack(self.saved_features), f"{base_dir}/{self.map_name}_vectors.pt")
        
        # 🔴 消除警告：先转 numpy.array，再转 tensor
        poses_array = np.array(self.saved_poses)
        torch.save(torch.tensor(poses_array), f"{base_dir}/{self.map_name}_poses.pt")
        
        edges_tensor = torch.tensor(self.edges, dtype=torch.long).reshape(-1, 2)
        torch.save(edges_tensor, f"{base_dir}/{self.map_name}_edges.pt")
        img_dir = f"{base_dir}/images/{self.map_name}/"
        os.makedirs(img_dir, exist_ok=True)
        for i, img_array in enumerate(self.saved_images):
            PILImage.fromarray(img_array).save(os.path.join(img_dir, f"{i}.png"))
        rospy.loginfo(f"🎉 终极纯净版地图 [{self.map_name}] 构建完毕！")

def valid_map_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(
            "map name may contain only letters, digits, '.', '_' and '-'"
        )
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Online Topological Mapper")
    parser.add_argument(
        "--map",
        type=valid_map_name,
        default="y_graph",
        help="Name of the map to save, for example: --map L_map",
    )
    parser.add_argument("--map-root", default=None, help="Topomap output directory")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--snap-threshold", type=float, default=0.65)
    parser.add_argument("--dense-distance", type=float, default=0.65)
    parser.add_argument("--sparse-distance", type=float, default=1.20)
    parser.add_argument("--parent-radius", type=float, default=1.50)
    parser.add_argument("--sync-slop", type=float, default=0.10)
    parser.add_argument("--startup-timeout", type=float, default=8.0)
    # rospy.myargv removes ROS remapping arguments while argparse still rejects
    # misspelled application options such as "--L_map".
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    
    # 🔴 将解析到的名字传给类
    mapper = OnlineTopologicalMapper(
        map_name=args.map,
        map_root=args.map_root,
        snap_thresh=args.snap_threshold,
        dense_dist=args.dense_distance,
        sparse_dist=args.sparse_distance,
        parent_radius=args.parent_radius,
        sync_slop=args.sync_slop,
        image_topic=args.image_topic,
        odom_topic=args.odom_topic,
        startup_timeout=args.startup_timeout,
    )
    rospy.spin()
