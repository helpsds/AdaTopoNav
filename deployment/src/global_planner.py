#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
import os
from PIL import Image as PILImage
import networkx as nx

class GlobalPlanner:
    def __init__(self, map_name, goal_node=-1):
        rospy.init_node('global_topological_planner', anonymous=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.expanduser(f"~/visualnav-transformer/deployment/topomaps")
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
        if self.goal_node >= self.total_nodes: self.goal_node = self.total_nodes - 1

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
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback) 

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
        if self.current_pose is None: return

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
            
        search_pool = list(search_pool) # 比如 [7, 8, 9, 24, 25]

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
        if self.current_node == self.goal_node:
            rospy.loginfo("🎯 成功抵达最终节点！正在紧急制动...")
            
            # 1. 🔪 第一步：立刻切断小脑和肌肉的电源，防止它们继续抢夺方向盘
            os.system("pkill -9 -f navigate_dynamic.py")
            os.system("pkill -9 -f pd_controller.py")
            
            # 给系统一点点时间彻底回收进程
            rospy.sleep(0.2) 
            
            # 2. 🛑 第二步：此时世界安静了，我们来踩死刹车
            from geometry_msgs.msg import Twist
            vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
            zero_vel = Twist()
            zero_vel.linear.x = 0.0
            zero_vel.angular.z = 0.0
            
            # 连踩 20 脚刹车，确保底层绝对收到
            for _ in range(20):
                vel_pub.publish(zero_vel)
                rospy.sleep(0.05)
            
            rospy.loginfo("🏁 车辆已彻底停稳。完美收工！")
            rospy.signal_shutdown("Goal Reached")
            return

        # ---------------------------------------------------------
        # 🚀 第二步：基于更新后的位置，下发真实导航目标
        # ---------------------------------------------------------
        try:
            shortest_path = nx.shortest_path(self.graph, source=self.current_node, target=self.goal_node, weight='weight')
        except nx.NetworkXNoPath:
            return

        path_length_remaining = len(shortest_path) - 1
        
        # ---------------------------------------------------------
        # 🧠 第三步：视距动态回弹 (V-LOS 遮挡检测)
        # ---------------------------------------------------------
        if path_length_remaining > 0:
            # 默认最大看前方 3 步
            dynamic_lookahead = min(3, path_length_remaining)
            target_node = shortest_path[dynamic_lookahead]

            # 🔴 终极遮挡检测：拿现在的画面，去和预定目标节点的照片比对
            sim_to_target = torch.nn.functional.cosine_similarity(
                current_feature.unsqueeze(0), self.map_vectors[target_node].unsqueeze(0)
            ).item()

            # 如果 DINOv2 判定相似度跌破 0.85，说明目标在墙后面（视野盲区）！
            if sim_to_target < 0.85:
                # 强行把视距拉回眼前（只看下一步的拐角），防止小脑抓瞎
                target_node = shortest_path[1]
                rospy.loginfo_throttle(1.0, f"⚠️ 弯道盲区 (Sim: {sim_to_target:.2f})！视距紧急缩短至节点 [{target_node}]")
            else:
                rospy.loginfo_throttle(1.0, f"🛣️ 视野开阔 (Sim: {sim_to_target:.2f})，巡航目标 [{target_node}]")
        else:
            target_node = self.goal_node

        rospy.loginfo_throttle(1.0, f"📍 定位: {self.current_node} | 🎯 路线目标: {target_node} | 余剩步数: {path_length_remaining}")
        self.publish_goal_image(target_node)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="y_graph_map")
    parser.add_argument("--goal", type=int, default=-1)
    args = parser.parse_args()

    try:
        GlobalPlanner(map_name=args.dir, goal_node=args.goal) 
        rospy.spin()
    except rospy.ROSInterruptException:
        pass