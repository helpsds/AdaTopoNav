#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torchvision.transforms as transforms
# 🔴 删掉了 resnet，改为准备加载 DINOv2
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math
import os
from PIL import Image as PILImage

class OnlineTopologicalMapper:
    def __init__(self, map_name="online_map"):
        rospy.init_node('online_topomap_builder', anonymous=True)
        self.map_name = map_name 
        
        # ---------------------------------------------------------
        # 🚀 升级 1：换装 DINOv2 (几何特征之王)
        # ---------------------------------------------------------
        rospy.loginfo("正在加载 DINOv2 模型 (vits14)... 首次运行可能较慢...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载 Facebook 的 DINOv2。vits14 比较轻量，适合 RTX 5060 跑满帧
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.feature_extractor.eval()
        
        # DINOv2 的输入必须是 14 的倍数，224x224 (14*16) 正好合适
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.saved_features = []
        self.saved_images = []
        self.saved_poses = []       # 🔴 升级 2：新增坐标存储，用于“方法三”导航
        
        self.last_feature = None
        self.last_pose = None      
        self.current_pose = None   
        
        # 🔴 优化参数：针对 DINOv2 的高灵敏度，相似度阈值设为 0.90
        self.dist_thresh = 1.0            # 每1米一个点
        self.yaw_thresh = math.radians(25) # 转过25度拍一张
        self.sim_thresh = 0.90            # DINOv2 向量非常紧凑，0.90 比较合适
        
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        
        rospy.on_shutdown(self.save_map)
        rospy.loginfo("✅ [DINOv2版] 运动学+视觉联合建图启动！")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # 四元数转 Yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_pose = (pos.x, pos.y, yaw)

    def extract_feature(self, img_array):
        img_tensor = self.transform(img_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # DINOv2 直接输出的就是高度浓缩的特征向量
            feature = self.feature_extractor(img_tensor)
        return feature.flatten()

    def image_callback(self, msg):
        if self.current_pose is None: return
        
        try:
            img_1d = np.frombuffer(msg.data, dtype=np.uint8)
            img_array = img_1d.reshape(msg.height, msg.width, -1)
            if msg.encoding == "bgr8":
                img_array = img_array[:, :, ::-1]
        except: return

        current_feature = self.extract_feature(img_array)
        
        if self.last_feature is None:
            self.add_node(current_feature, img_array, "起点")
            return

        # 核心判定逻辑
        distance = math.hypot(self.current_pose[0] - self.last_pose[0], self.current_pose[1] - self.last_pose[1])
        dyaw = self.current_pose[2] - self.last_pose[2]
        yaw_diff = abs(math.atan2(math.sin(dyaw), math.cos(dyaw)))
        similarity = torch.nn.functional.cosine_similarity(current_feature.unsqueeze(0), self.last_feature.unsqueeze(0)).item()

        if distance > self.dist_thresh:
            self.add_node(current_feature, img_array, f"距离触发({distance:.1f}m)")
        elif yaw_diff > self.yaw_thresh:
            self.add_node(current_feature, img_array, f"转弯触发({math.degrees(yaw_diff):.0f}°)")
        elif similarity < self.sim_thresh and distance > 0.3:
            self.add_node(current_feature, img_array, f"视觉突变(Sim:{similarity:.2f})")

    def add_node(self, feature, image, reason):
        self.saved_features.append(feature.cpu())
        self.saved_images.append(image)
        # 🔴 同时保存当前的物理坐标
        self.saved_poses.append(np.array([self.current_pose[0], self.current_pose[1]]))
        
        self.last_feature = feature
        self.last_pose = self.current_pose
        rospy.loginfo(f"📸 节点 [{len(self.saved_features)-1}] 记录成功 | 原因: {reason}")

    def save_map(self):
        if not self.saved_features: return
            
        rospy.loginfo(f"💾 正在导出 DINOv2 高维地图...")
        # 保存高维向量
        map_tensor = torch.stack(self.saved_features)
        save_path_vec = os.path.expanduser(f"~/visualnav-transformer/deployment/topomaps/{self.map_name}_vectors.pt")
        torch.save(map_tensor, save_path_vec)
        
        # 🔴 升级 3：保存对应的坐标数据 (用于方法三导航比对)
        save_path_poses = os.path.expanduser(f"~/visualnav-transformer/deployment/topomaps/{self.map_name}_poses.pt")
        torch.save(torch.tensor(self.saved_poses), save_path_poses)
        
        # 保存图片预览
        img_dir = os.path.expanduser(f"~/visualnav-transformer/deployment/topomaps/images/{self.map_name}/")
        os.makedirs(img_dir, exist_ok=True)
        for i, img_array in enumerate(self.saved_images):
            PILImage.fromarray(img_array).save(os.path.join(img_dir, f"{i}.png"))
        
        rospy.loginfo(f"🎉 地图保存完毕！向量与坐标均已就绪。")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="online_map")
    args = parser.parse_args()
    mapper = OnlineTopologicalMapper(map_name=args.map)
    rospy.spin()