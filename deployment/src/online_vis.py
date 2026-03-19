#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torchvision.transforms as transforms
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import os
from PIL import Image as PILImage

# 🔴 新增：引入画图库，并设置为后台模式以防卡死
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class OnlineTopologicalMapper:
    def __init__(self, map_name="y_graph"):
        rospy.init_node('online_topomap_builder', anonymous=True)
        self.map_name = map_name 
        
        rospy.loginfo("正在加载 DINOv2 模型...")
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
        
        # 🔴 核心参数精调
        self.dist_thresh = 1.0   # 直道上的正常节点间距
        self.snap_thresh = 0.65  # 路口吸附半径 (稍微放大一点，包容倒车时的左右摇摆)

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.on_shutdown(self.save_map)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_pose = np.array([pos.x, pos.y])

    def extract_feature(self, img_array):
        img_tensor = self.transform(img_array).unsqueeze(0).to(self.device)
        with torch.no_grad(): return self.feature_extractor(img_tensor).flatten()

    def image_callback(self, msg):
        if self.current_pose is None: return
        
        try:
            img_1d = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_1d.reshape(msg.height, msg.width, -1)
            if msg.encoding == "bgr8": img_array = img_array[:, :, ::-1]
        except: return

        if self.active_node is None:
            feature = self.extract_feature(img_array)
            self.add_node(feature, img_array, self.current_pose, reason="起点")
            self.active_node = 0
            return

        poses_array = np.array(self.saved_poses)
        dists = np.linalg.norm(poses_array - self.current_pose, axis=1)
        min_dist = np.min(dists)
        closest_idx = np.argmin(dists)

        # 🎯 动作 1：绝对空间吸附 (霸道总裁版)
        if min_dist < self.snap_thresh:
            if self.active_node != closest_idx:
                self.active_node = closest_idx
                rospy.logwarn(f"🔙 回溯吸附：定位于 [{closest_idx}]，暂停建图！")
            return 

        # 🎯 动作 2：开疆拓土 (智能间距版)
        active_pose = self.saved_poses[self.active_node]
        dist_to_active = np.linalg.norm(self.current_pose - active_pose)
        new_idx = len(self.saved_features)

        is_branching = (self.active_node != new_idx - 1)
        
        required_dist = 0.65 if is_branching else self.dist_thresh

        if dist_to_active >= required_dist:
            feature = self.extract_feature(img_array)
            
            if is_branching:
                search_radius = 1.5 # 放宽找老祖宗的视野
                valid_parents = np.where(dists < search_radius)[0]
                best_parent = np.min(valid_parents) if len(valid_parents) > 0 else self.active_node
                action_str = f"🛣️ 开辟新岔路 (锚定老祖先 {best_parent}，已投放 0.65m 过渡面包屑)"
            else:
                best_parent = self.active_node
                action_str = "顺延直行 (1.0m)"
                
            self.edges.append((best_parent, new_idx))
            self.add_node(feature, img_array, self.current_pose, reason=action_str)
            self.active_node = new_idx

    def add_node(self, feature, image, pose, reason):
        node_id = len(self.saved_features)
        self.saved_features.append(feature.cpu())
        self.saved_images.append(image)
        self.saved_poses.append(pose)
        rospy.loginfo(f"📸 节点 [{node_id}] | {reason}")

    # 🔴 新增：自动绘制并保存拓扑图的方法
    def save_visualization(self, base_dir):
        if not self.saved_poses or not self.edges:
            return
            
        poses = np.array(self.saved_poses)
        plt.figure(figsize=(10, 8))
        
        # 画连线
        for u, v in self.edges:
            x_coords = [poses[u, 0], poses[v, 0]]
            y_coords = [poses[u, 1], poses[v, 1]]
            if abs(u - v) == 1:
                plt.plot(x_coords, y_coords, 'b-', alpha=0.4, linewidth=2, label='Sequential Path' if u==0 else "")
            else:
                plt.plot(x_coords, y_coords, 'r--', alpha=0.9, linewidth=3, label='Branch/Shortcut' if u!=0 else "")
                
        # 画节点
        plt.scatter(poses[:, 0], poses[:, 1], c='black', s=40, zorder=5, label='Nodes')
        for i, (x, y) in enumerate(poses):
            plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,6), ha='center', fontsize=9)
            
        plt.title(f"Topological Graph: {self.map_name}\nTotal Nodes: {len(poses)}", fontsize=15, fontweight='bold')
        plt.xlabel("X Coordinate (meters)")
        plt.ylabel("Y Coordinate (meters)")
        plt.axis('equal') 
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 去除重复图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        # 存为 PNG 图片
        vis_path = os.path.join(base_dir, f"{self.map_name}_vis.png")
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()
        rospy.loginfo(f"📊 拓扑图可视化已自动保存至: {vis_path}")

    def save_map(self):
        if not self.saved_features: return
        base_dir = os.path.expanduser(f"~/visualnav-transformer/deployment/topomaps")
        os.makedirs(base_dir, exist_ok=True)
        
        torch.save(torch.stack(self.saved_features), f"{base_dir}/{self.map_name}_vectors.pt")
        torch.save(torch.tensor(self.saved_poses), f"{base_dir}/{self.map_name}_poses.pt")
        torch.save(torch.tensor(self.edges), f"{base_dir}/{self.map_name}_edges.pt")
        
        img_dir = f"{base_dir}/images/{self.map_name}/"
        os.makedirs(img_dir, exist_ok=True)
        for i, img_array in enumerate(self.saved_images):
            PILImage.fromarray(img_array).save(os.path.join(img_dir, f"{i}.png"))
            
        # 🔴 在保存数据后，立刻触发画图保存
        self.save_visualization(base_dir)
        rospy.loginfo(f"🎉 终极纯净版 Y 图构建完毕！")

if __name__ == '__main__':
    mapper = OnlineTopologicalMapper()
    rospy.spin()