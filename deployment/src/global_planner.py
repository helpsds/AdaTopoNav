#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32
from nav_msgs.msg import Odometry
import os
import time
from PIL import Image as PILImage
import networkx as nx

class GlobalPlanner:
    def __init__(
        self,
        map_name,
        goal_node=-1,
        map_root=None,
        goal_distance=0.45,
        goal_confirmations=3,
        timeout=0.0,
        lookahead=3,
        vlos_threshold=0.85,
        enable_vlos=True,
    ):
        rospy.init_node('global_topological_planner', anonymous=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.abspath(
            os.path.expanduser(
                map_root or os.path.join(os.path.dirname(__file__), "..", "topomaps")
            )
        )
        self.map_vectors = torch.load(f"{base_dir}/{map_name}_vectors.pt", map_location=self.device)
        self.map_poses = torch.load(f"{base_dir}/{map_name}_poses.pt").to(self.device)
        
        edges_path = f"{base_dir}/{map_name}_edges.pt"
        if os.path.exists(edges_path):
            self.map_edges = torch.load(edges_path).numpy()
        else:
            rospy.logerr("找不到 edges 文件！")
            return

        self.total_nodes = self.map_vectors.shape[0]
        self.img_dir = f"{base_dir}/images/{map_name}/"
        
        self.goal_node = self.total_nodes - 1 if goal_node == -1 else goal_node
        self.goal_node = min(max(self.goal_node, 0), self.total_nodes - 1)
        self.goal_distance = goal_distance
        self.goal_confirmations = max(1, goal_confirmations)
        self.goal_confirmation_count = 0
        self.timeout = timeout
        # Start the navigation timeout only after heavyweight model loading
        # and ROS setup complete. Use wall time so Gazebo real-time factor
        # cannot make a 240-second trial expire during DINOv2 loading.
        self.start_wall_time = None
        self.lookahead = max(1, lookahead)
        self.vlos_threshold = vlos_threshold
        self.enable_vlos = enable_vlos
        self.finished = False

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.total_nodes))
        
        for u, v in self.map_edges:
            dist = torch.norm(self.map_poses[u] - self.map_poses[v]).item()
            self.graph.add_edge(u, v, weight=dist)

        rospy.loginfo(f"🗺️ 拓扑网构建完成！共 {self.total_nodes} 节点, {len(self.map_edges)} 条边。")
        
        rospy.loginfo("正在加载 DINOv2 模型...")
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.feature_extractor.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.current_node = 0
        self.current_pose = None

        self.goal_image_pub = rospy.Publisher("/topoplan/target_image", Image, queue_size=1)
        self.node_pub = rospy.Publisher("/topoplan/current_node", Int32, queue_size=1)
        self.goal_reached_pub = rospy.Publisher(
            "/topoplan/planner_goal_reached", Bool, queue_size=1, latch=True
        )
        # Reset the latched result at the beginning of every trial. Without
        # this, a navigation node can consume True left by the preceding
        # trial and shut down before publishing its first waypoint.
        self.goal_reached_pub.publish(False)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback) 
        self.start_wall_time = time.monotonic()

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_pose = (pos.x, pos.y)

    def publish_goal_image(self, target_idx):
        img_path = os.path.join(self.img_dir, f"{target_idx}.png")
        if not os.path.exists(img_path): return
        pil_img = PILImage.open(img_path).convert('RGB')
        img_array = np.array(pil_img)
        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now()
        img_msg.height, img_msg.width, _ = img_array.shape
        img_msg.encoding = "rgb8"
        img_msg.step = img_msg.width * 3
        img_msg.data = img_array.tobytes()
        self.goal_image_pub.publish(img_msg)

    def image_callback(self, msg):
        if self.current_pose is None or self.finished:
            return
        if (
            self.timeout > 0
            and self.start_wall_time is not None
            and time.monotonic() - self.start_wall_time >= self.timeout
        ):
            rospy.logwarn("导航超时；规划器未声明成功。")
            self.finished = True
            self.goal_reached_pub.publish(False)
            rospy.signal_shutdown("Navigation timeout")
            return

        try:
            img_1d = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_1d.reshape(msg.height, msg.width, -1)
            if msg.encoding == "bgr8": img_array = img_array[:, :, ::-1]
        except Exception: return

        img_tensor = self.transform(img_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            current_feature = self.feature_extractor(img_tensor).flatten()

        # ---------------------------------------------------------
        # 📍 第一步：【绝杀修复】基于图论的动态搜索池
        # ---------------------------------------------------------
        try:
            temp_path = nx.shortest_path(self.graph, source=self.current_node, target=self.goal_node, weight='weight')
        except nx.NetworkXNoPath:
            temp_path = [self.current_node]

        # 构造搜索池：当前节点 + 它的连通邻居(包括捷径) + 前方即将走的3个节点
        search_pool = set()
        search_pool.add(self.current_node)
        for neighbor in self.graph.neighbors(self.current_node):
            search_pool.add(neighbor)
        for node in temp_path[:4]: 
            search_pool.add(node)
            
        search_pool = sorted(search_pool)

        # 从张量中提取这些指定节点的特征和坐标
        window_vectors = self.map_vectors[search_pool]
        expected_poses = self.map_poses[search_pool]

        similarities = torch.nn.functional.cosine_similarity(current_feature.unsqueeze(0), window_vectors)
        current_pose_tensor = torch.tensor([self.current_pose[0], self.current_pose[1]], device=self.device)
        dist_errors = torch.norm(expected_poses - current_pose_tensor, dim=1)

        total_score = (1.0 * similarities) - (0.5 * dist_errors)
        best_idx_in_pool = torch.argmax(total_score).item()

        # 更新真实的全局节点 ID
        self.current_node = search_pool[best_idx_in_pool]
        self.node_pub.publish(self.current_node)

        # ---------------------------------------------------------
        # 🛑 终点刹车逻辑
        # ---------------------------------------------------------
        goal_pose = self.map_poses[self.goal_node]
        goal_distance = torch.norm(goal_pose - current_pose_tensor).item()
        if self.current_node == self.goal_node and goal_distance <= self.goal_distance:
            self.goal_confirmation_count += 1
        else:
            self.goal_confirmation_count = 0

        if self.goal_confirmation_count >= self.goal_confirmations:
            rospy.loginfo(
                "GOAL_REACHED node=%d distance=%.3f confirmations=%d",
                self.goal_node,
                goal_distance,
                self.goal_confirmation_count,
            )
            self.finished = True
            self.goal_reached_pub.publish(True)
            return

        # ---------------------------------------------------------
        # 🚀 第二步：基于更新后的位置，下发真实导航目标
        # ---------------------------------------------------------
        try:
            shortest_path = nx.shortest_path(self.graph, source=self.current_node, target=self.goal_node, weight='weight')
        except nx.NetworkXNoPath:
            return

        path_length_remaining = len(shortest_path) - 1
        
        # Appearance-continuity lookahead adjustment. DINOv2 similarity is a
        # heuristic and must not be interpreted as geometric visibility.
        if path_length_remaining > 0:
            # 默认最大看前方 3 步
            dynamic_lookahead = min(self.lookahead, path_length_remaining)
            target_node = shortest_path[dynamic_lookahead]

            sim_to_target = torch.nn.functional.cosine_similarity(
                current_feature.unsqueeze(0), self.map_vectors[target_node].unsqueeze(0)
            ).item()

            if self.enable_vlos and sim_to_target < self.vlos_threshold:
                target_node = shortest_path[1]
                rospy.loginfo_throttle(1.0, f"低外观连续性 (Sim: {sim_to_target:.2f})；缩短至节点 [{target_node}]")
            else:
                mode = "固定前瞻" if not self.enable_vlos else "外观连续"
                rospy.loginfo_throttle(
                    1.0,
                    f"{mode} (Sim: {sim_to_target:.2f})，目标 [{target_node}]",
                )
        else:
            target_node = self.goal_node

        rospy.loginfo_throttle(1.0, f"📍 定位: {self.current_node} | 🎯 路线目标: {target_node} | 余剩步数: {path_length_remaining}")
        self.publish_goal_image(target_node)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="y_graph_map")
    parser.add_argument("--goal", type=int, default=-1)
    parser.add_argument("--map-root", default=None)
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--goal-confirmations", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--vlos-threshold", type=float, default=0.85)
    parser.add_argument(
        "--disable-vlos",
        action="store_true",
        help="Use a fixed lookahead; intended for controlled ablation.",
    )
    args = parser.parse_args()

    try:
        GlobalPlanner(
            map_name=args.dir,
            goal_node=args.goal,
            map_root=args.map_root,
            goal_distance=args.goal_distance,
            goal_confirmations=args.goal_confirmations,
            timeout=args.timeout,
            lookahead=args.lookahead,
            vlos_threshold=args.vlos_threshold,
            enable_vlos=not args.disable_vlos,
        )
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
