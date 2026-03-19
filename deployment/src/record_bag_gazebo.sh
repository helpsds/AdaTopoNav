#!/bin/bash

# 1. 自动获取当前脚本所在目录，确保路径绝对正确
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BAG_DIR="$SCRIPT_DIR/../topomaps/bags"

# 确保录制目录存在
mkdir -p "$BAG_DIR"

# 2. 清理之前的残留进程，防止 SpawnModel 报错
killall -9 rosmaster roscore gzserver gzclient 2>/dev/null

# 3. 创建一个新的 tmux 会话
session_name="record_scout_$(date +%s)"
tmux new-session -d -s $session_name

# 4. 窗格布局：左侧大窗格 (Gazebo)，右侧上下切分 (Teleop & Bag)
tmux selectp -t 0
tmux splitw -h -p 35 # 将屏幕分为左 65% 右 35%
tmux selectp -t 1
tmux splitw -v -p 50 # 将右侧上下平分

# --- 窗格 0：启动 Gazebo 仿真 (核心：必须纯净环境) ---
tmux select-pane -t 0
tmux send-keys "conda deactivate" Enter
tmux send-keys "conda deactivate" Enter # 确保彻底退出 Conda
tmux send-keys "source /opt/ros/noetic/setup.bash" Enter
tmux send-keys "source ~/scout_ws/devel/setup.bash" Enter
tmux send-keys "roslaunch scout_gazebo_sim scout_y_path_v2.launch" Enter

# 关键：Gazebo 加载模型很慢，必须等久一点
echo "等待仿真环境加载 (15秒)..."
sleep 15

# --- 窗格 1：启动键盘控制 ---
tmux select-pane -t 1
tmux send-keys "conda deactivate" Enter
tmux send-keys "source /opt/ros/noetic/setup.bash" Enter
tmux send-keys "rosrun teleop_twist_keyboard teleop_twist_keyboard.py" Enter

# --- 窗格 2：准备录制 Rosbag ---
tmux select-pane -t 2
tmux send-keys "cd $BAG_DIR" Enter
# 【重要修改】话题名改为验证成功的 /camera/color/image_raw
# 注意：这里不加 Enter，由你手动触发录制
tmux send-keys "rosbag record /camera/color/image_raw -o $1" 

# 5. 开启鼠标支持（方便你切换窗口）
tmux set-option -g mouse on

# 6. 附加到会话
tmux -2 attach-session -t $session_name