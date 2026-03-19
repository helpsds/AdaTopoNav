#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt
import os
import argparse

def visualize_topomap(map_name):
    # 1. 拼接文件路径
    base_dir = os.path.expanduser("~/visualnav-transformer/deployment/topomaps")
    poses_path = os.path.join(base_dir, f"{map_name}_poses.pt")
    edges_path = os.path.join(base_dir, f"{map_name}_edges.pt")

    if not os.path.exists(poses_path):
        print(f"❌ 找不到坐标文件: {poses_path}")
        return

    # 2. 加载数据
    poses = torch.load(poses_path, map_location='cpu').numpy()
    
    if os.path.exists(edges_path):
        edges = torch.load(edges_path, map_location='cpu').numpy()
    else:
        edges = []
        print("⚠️ 未找到边(Edges)文件，可能建图时没有开启闭环检测。")

    # 3. 开始绘图
    plt.figure(figsize=(10, 8))
    loop_closures = 0

    # 🧶 画出所有的连通边 (Edges)
    for u, v in edges:
        x_coords = [poses[u, 0], poses[v, 0]]
        y_coords = [poses[u, 1], poses[v, 1]]
        
        # 判断是顺序相邻的节点，还是跨越性的闭环节点
        if abs(u - v) == 1:
            # 顺序行驶路线（蓝色实线）
            plt.plot(x_coords, y_coords, 'b-', alpha=0.4, linewidth=2, 
                     label='Sequential Path' if u==0 else "")
        else:
            # 🚀 闭环捷径（红色粗虚线）
            plt.plot(x_coords, y_coords, 'r--', alpha=0.9, linewidth=3, 
                     label='Loop Closure (Shortcut)' if loop_closures==0 else "")
            loop_closures += 1

    # 📍 画出所有的节点 (Nodes)
    plt.scatter(poses[:, 0], poses[:, 1], c='black', s=40, zorder=5, label='Nodes')
    
    # 在每个黑点旁边标上节点的序号 (0, 1, 2...)
    for i, (x, y) in enumerate(poses):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,6), ha='center', fontsize=9)

    # 4. 图表排版与美化
    plt.title(f"Topological Graph: {map_name}\nTotal Nodes: {len(poses)} | Loop Closures: {loop_closures}", fontsize=15, fontweight='bold')
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    
    # ⚠️ 关键设置：保证 X 和 Y 轴比例 1:1，这样画出来的 Y 型路口才不会变形
    plt.axis('equal') 
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 去除重复的图例标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    # ---------------------------------------------------------
    # 💾 核心修改：将弹窗改为保存为高清图片
    # ---------------------------------------------------------
    save_path = os.path.join(base_dir, f"{map_name}_visualization.png")
    
    # bbox_inches='tight' 会自动裁掉图表边缘多余的空白
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # 保存后关闭图表，释放内存，防止批量处理时卡顿

    print(f"✅ 绘图完成！共绘制了 {len(poses)} 个节点和 {len(edges)} 条边。")
    print(f"🔥 发现了 {loop_closures} 条闭环捷径！")
    print(f"🖼️ 高清拓扑图已保存至: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="拓扑图可视化工具")
    parser.add_argument("--map", type=str, default="y_graph_map", help="要显示的地图名称")
    args = parser.parse_args()
    visualize_topomap(args.map)