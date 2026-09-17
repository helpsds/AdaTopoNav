#!/bin/bash

# 1. 自动获取当前脚本所在目录，确保路径绝对正确
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BAG_DIR="$SCRIPT_DIR/../topomaps/bags"
COLLECTION_ROOT="/home/ljx/gazebo_models_worlds_collection"
CMU_MODELS="/home/ljx/scout_ws/src/ugv_gazebo_sim/cmu_worlds/models"

# 用法：
#   bash record_bag_gazebo.sh <bag_name> [world_name|world_file]
# 示例：
#   bash record_bag_gazebo.sh y_city city_osm_roundabout_combined
#   bash record_bag_gazebo.sh y_city /path/to/custom.world
WORLD_INPUT="${2:-${WORLD_FILE:-}}"
if [[ -z "$WORLD_INPUT" ]]; then
    WORLD_FILE="$(rospack find cmu_worlds)/worlds/indoor.world"
elif [[ "$WORLD_INPUT" == */* ]]; then
    WORLD_FILE="$WORLD_INPUT"
else
    [[ "$WORLD_INPUT" == *.world ]] || WORLD_INPUT="${WORLD_INPUT}.world"
    WORLD_FILE="$COLLECTION_ROOT/worlds/$WORLD_INPUT"
fi

if [[ ! -f "$WORLD_FILE" ]]; then
    echo "错误：找不到 Gazebo 场景：$WORLD_FILE" >&2
    exit 2
fi

# 保留原有 CMU 模型，同时加入场景库模型；不能覆盖已有搜索路径。
export GAZEBO_MODEL_PATH="$COLLECTION_ROOT/models:$CMU_MODELS${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
export GAZEBO_RESOURCE_PATH="$COLLECTION_ROOT/worlds${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"
SPAWN_X="${SPAWN_X:--12}"
SPAWN_Y="${SPAWN_Y:-14}"
SPAWN_Z="${SPAWN_Z:-0.3}"
SPAWN_YAW="${SPAWN_YAW:-1.56}"

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
tmux send-keys "ros_sim" Enter
# tmux send-keys "source /opt/ros/noetic/setup.bash" Enter
tmux send-keys "source ~/scout_ws/devel/setup.bash" Enter
tmux send-keys "export GAZEBO_MODEL_PATH='$GAZEBO_MODEL_PATH'" Enter
tmux send-keys "export GAZEBO_RESOURCE_PATH='$GAZEBO_RESOURCE_PATH'" Enter
tmux send-keys "roslaunch scout_gazebo_sim scout_mini_playpen.launch world_name:='$WORLD_FILE' spawn_x:='$SPAWN_X' spawn_y:='$SPAWN_Y' spawn_z:='$SPAWN_Z' spawn_yaw:='$SPAWN_YAW'" Enter
# tmux send-keys "roslaunch scout_gazebo_sim scout_mini_obstacle.launch" Enter

# tmux send-keys "roslaunch scout_gazebo_sim scout_mini_playpen.launch world_name:=/home/ljx/gazebo_world/Obstacle.world" Enter

# 关键：Gazebo 加载模型很慢，必须等久一点
echo "等待仿真环境加载 (10秒)..."
sleep 10

# --- 窗格 1：启动键盘控制 ---
tmux select-pane -t 1
tmux send-keys "conda deactivate" Enter
tmux send-keys "ros_sim" Enter
tmux send-keys "source /opt/ros/noetic/setup.bash" Enter
tmux send-keys "source ~/scout_ws/devel/setup.bash" Enter
tmux send-keys "rosrun teleop_twist_keyboard teleop_twist_keyboard.py" Enter

# --- 窗格 2：准备录制 Rosbag ---
tmux select-pane -t 2
tmux send-keys "ros_sim" Enter
tmux send-keys "cd $BAG_DIR" Enter
# 【重要修改】话题名改为验证成功的 /camera/color/image_raw
# 注意：这里不加 Enter，由你手动触发录制
tmux send-keys "rosbag record /camera/color/image_raw -o $1" 
# tmux send-keys "rosbag record -a -o $1"

# 5. 开启鼠标支持（方便你切换窗口）
tmux set-option -g mouse on

# 6. 附加到会话
tmux -2 attach-session -t $session_name
