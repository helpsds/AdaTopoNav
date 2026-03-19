#!/usr/bin/env python3
import os
import rospy
import numpy as np
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
import networkx as nx

class GlobalTopologicalPlanner:
    def __init__(self, map_name, goal_node=-1):
        rospy.init_node('global_topological_planner', anonymous=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 基础配置与参数化 (去除硬编码)
        self.base_dir = os.path.expanduser("~/visualnav-transformer/deployment/topomaps")
        self.map_name = map_name
        self.img_dir = f"{self.base_dir}/images/{map_name}/"
        self.lookahead_dist = rospy.get_param("~lookahead_dist", 2) # 从ROS参数服务器读取预瞄距离

        # 2. 加载与构建地图
        self._load_map_data()

        # 3. 目标与状态管理
        self.goal_node = self.total_nodes - 1 if goal_node == -1 else goal_node
        self.goal_node = min(self.goal_node, self.total_nodes - 1)
        self.is_goal_reached = False
        self.current_node = 0
        self.current_pose = None

        # 4. 加载视觉模型
        self._init_vision_model()

        # 5. ROS 通信接口 (规范化)
        self.goal_image_pub = rospy.Publisher("/topoplan/target_image", Image, queue_size=1)
        self.node_pub = rospy.Publisher("/topoplan/current_node", Int32, queue_size=1)
        # 替代 pkill 的优雅方式：发布导航状态让底层主动停止
        self.nav_status_pub = rospy.Publisher("/topoplan/goal_reached", Bool, queue_size=1, latch=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # 增加 buff_size 防止图像延迟累积导致规划器滞后
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"🚀 全局拓扑规划器已启动，目标节点: {self.goal_node}")

    def _load_map_data(self):
        """模块 1：地图数据加载与图构建"""
        rospy.loginfo("正在加载拓扑地图数据...")
        self.map_vectors = torch.load(f"{self.base_dir}/{self.map_name}_vectors.pt", map_location=self.device)
        self.map_poses = torch.load(f"{self.base_dir}/{self.map_name}_poses.pt").to(self.device)
        
        edges_path = f"{self.base_dir}/{self.map_name}_edges.pt"
        if not os.path.exists(edges_path):
            rospy.logerr(f"找不到边缘文件: {edges_path}")
            raise FileNotFoundError("Missing edges file")
        
        self.map_edges = torch.load(edges_path).numpy()
        self.total_nodes = self.map_vectors.shape[0]

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.total_nodes))
        for u, v in self.map_edges:
            dist = torch.norm(self.map_poses[u] - self.map_poses[v]).item()
            self.graph.add_edge(u, v, weight=dist)

        rospy.loginfo(f"🗺️ 拓扑网构建完成！共 {self.total_nodes} 节点, {len(self.map_edges)} 条边。")

    def _init_vision_model(self):
        """模块 2：视觉模型初始化"""
        rospy.loginfo("正在加载 DINOv2 模型...")
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.feature_extractor.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_pose = np.array([pos.x, pos.y])

    def _stop_robot(self):
        """模块 3：优雅的生命周期与控制管理"""
        self.nav_status_pub.publish(Bool(True)) # 通知小脑节点自我关闭或挂起
        zero_vel = Twist()
        for _ in range(5): 
            self.cmd_vel_pub.publish(zero_vel)
            rospy.sleep(0.1)
        rospy.loginfo("🏁 车辆已彻底停稳。")

    def _localize_robot(self, current_feature):
        """模块 4：拓扑定位逻辑 (解耦与数学修正)"""
        # 1. 构建合理的候选池（当前节点 + 1阶和2阶邻居），防止全局搜索跳变
        search_pool = set([self.current_node])
        for neighbor in self.graph.neighbors(self.current_node):
            search_pool.add(neighbor)
            for second_neighbor in self.graph.neighbors(neighbor):
                search_pool.add(second_neighbor)
        
        search_pool = list(search_pool)
        window_vectors = self.map_vectors[search_pool]
        
        # 2. 计算视觉相似度
        similarities = torch.nn.functional.cosine_similarity(current_feature.unsqueeze(0), window_vectors)
        
        # 3. 里程计与视觉的数学融合修正：使用高斯衰减而非线性相减
        if self.current_pose is not None:
            expected_poses = self.map_poses[search_pool].cpu().numpy()
            dist_errors = np.linalg.norm(expected_poses - self.current_pose, axis=1)
            # 距离越远，权重呈指数衰减
            dist_weights = torch.tensor(np.exp(-0.5 * dist_errors), device=self.device)
            total_score = similarities * dist_weights
        else:
            total_score = similarities

        best_idx = torch.argmax(total_score).item()
        return search_pool[best_idx]

    def _get_lookahead_target(self, path):
        """模块 5：动态前瞻策略 (替代生硬的 0.85 盲区判定)"""
        path_len = len(path)
        if path_len == 1:
            return path[0]
        # 根据当前路径长度和设定的前瞻距离提取目标节点
        lookahead_idx = min(self.lookahead_dist, path_len - 1)
        return path[lookahead_idx]

    def image_callback(self, msg):
        """主循环：基于图像回调驱动状态机"""
        if self.is_goal_reached:
            return

        # [步骤 A] 图像处理
        try:
            img_1d = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_1d.reshape(msg.height, msg.width, -1)
            if msg.encoding == "bgr8": img_array = img_array[:, :, ::-1]
        except Exception:
            return

        img_tensor = self.transform(img_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            current_feature = self.feature_extractor(img_tensor).flatten()

        # [步骤 B] 拓扑定位
        self.current_node = self._localize_robot(current_feature)
        self.node_pub.publish(self.current_node)

        # [步骤 C] 终点判定
        if self.current_node == self.goal_node:
            rospy.loginfo("🎯 成功抵达最终节点！")
            self.is_goal_reached = True
            self._stop_robot()
            return

        # [步骤 D] 全局规划
        try:
            shortest_path = nx.shortest_path(self.graph, source=self.current_node, target=self.goal_node, weight='weight')
        except nx.NetworkXNoPath:
            rospy.logwarn_throttle(2.0, "⚠️ 找不到通往目标的路径！")
            return

        # [步骤 E] 下发局部目标
        target_node = self._get_lookahead_target(shortest_path)
        rospy.loginfo_throttle(1.0, f"📍 定位: {self.current_node} | 🎯 预瞄目标: {target_node} | 余剩步数: {len(shortest_path)-1}")
        self._publish_goal_image(target_node)

    def _publish_goal_image(self, target_idx):
        img_path = os.path.join(self.img_dir, f"{target_idx}.png")
        if not os.path.exists(img_path): return
        try:
            pil_img = PILImage.open(img_path).convert('RGB')
            img_array = np.array(pil_img)
            img_msg = Image()
            img_msg.header.stamp = rospy.Time.now()
            img_msg.height, img_msg.width, _ = img_array.shape
            img_msg.encoding = "rgb8"
            img_msg.step = img_msg.width * 3
            img_msg.data = img_array.tobytes()
            self.goal_image_pub.publish(img_msg)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"加载目标图像失败: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="y_graph_map")
    parser.add_argument("--goal", type=int, default=-1)
    # 使用 parse_known_args 防止 roslaunch 传入的额外参数导致解析崩溃
    args, unknown = parser.parse_known_args() 

    try:
        GlobalTopologicalPlanner(map_name=args.dir, goal_node=args.goal) 
        rospy.spin()
    except rospy.ROSInterruptException:
        pass