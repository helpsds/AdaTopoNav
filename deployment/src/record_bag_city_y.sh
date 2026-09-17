#!/usr/bin/env bash

# 在 city_osm_roundabout_combined 场景中遥控 Scout 并录制视觉引导图。
#
# 用法：
#   bash record_bag_city_y.sh <bag_name>
#
# 示例：
#   bash record_bag_city_y.sh Y_CITY_GNM
#
# 第三个 tmux 窗格只会填入 rosbag 命令，不会自动开始录制。
# 确认场景、相机和机器人正常后，在该窗格按 Enter 开始录制。

set -u

if [[ $# -lt 1 || -z "$1" ]]; then
    echo "用法：bash $(basename "$0") <bag_name>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_DIR="$SCRIPT_DIR/../topomaps/bags"
COLLECTION_ROOT="/home/ljx/gazebo_models_worlds_collection"
CITY_WORLD="$COLLECTION_ROOT/worlds/city_osm_roundabout_combined.world"
CMU_MODELS="/home/ljx/scout_ws/src/ugv_gazebo_sim/cmu_worlds/models"
SCOUT_SETUP="/home/ljx/scout_ws/devel/setup.bash"

# 图片中给出的初始平面位姿。
SPAWN_X="-181.007352"
SPAWN_Y="-61.859492"
SPAWN_YAW="-0.040445"

# 使用较高的生成高度让 Scout 自然落地；落地后底盘高度约为 0.18 m。
SPAWN_Z="0.30"

if [[ ! -f "$CITY_WORLD" ]]; then
    echo "错误：找不到城市场景：$CITY_WORLD" >&2
    exit 2
fi

if [[ ! -f "$SCOUT_SETUP" ]]; then
    echo "错误：找不到 Scout 工作空间：$SCOUT_SETUP" >&2
    exit 2
fi

mkdir -p "$BAG_DIR"

export GAZEBO_MODEL_PATH="$COLLECTION_ROOT/models:$CMU_MODELS${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"
export GAZEBO_RESOURCE_PATH="$COLLECTION_ROOT/worlds${GAZEBO_RESOURCE_PATH:+:$GAZEBO_RESOURCE_PATH}"

# 防止上一次异常退出留下的 ROS/Gazebo 进程占用端口。
killall -9 rosmaster roscore gzserver gzclient 2>/dev/null || true

SESSION_NAME="record_city_y_$(date +%s)"
WINDOW="$SESSION_NAME:0"

tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR"
tmux set-option -t "$SESSION_NAME" mouse on

# 左侧为 Gazebo，右上为遥控，右下为 rosbag。
tmux split-window -h -p 35 -t "$WINDOW.0" -c "$SCRIPT_DIR"
tmux split-window -v -p 50 -t "$WINDOW.1" -c "$BAG_DIR"

# 窗格 0：启动城市环境和 Scout。
# 严格复用已验证可以正常启动该场景的环境和命令，不在这里切换 Conda。
tmux send-keys -t "$WINDOW.0" "export ROS_MASTER_URI=http://localhost:11311 ROS_HOSTNAME=localhost ROS_IP=127.0.0.1" Enter
tmux send-keys -t "$WINDOW.0" "source '$SCOUT_SETUP'" Enter
# gazebo_ros/spawn_model 使用 /usr/bin/env python3；确保它选中系统 Python，
# 因为 netifaces 安装在 /usr/lib/python3/dist-packages 中。
tmux send-keys -t "$WINDOW.0" "export PATH=/opt/ros/noetic/bin:/usr/bin:/bin:\$PATH; unset PYTHONHOME; hash -r" Enter
tmux send-keys -t "$WINDOW.0" "export GAZEBO_MODEL_PATH='$GAZEBO_MODEL_PATH'" Enter
tmux send-keys -t "$WINDOW.0" "export GAZEBO_RESOURCE_PATH='$GAZEBO_RESOURCE_PATH'" Enter
tmux send-keys -t "$WINDOW.0" \
    "roslaunch scout_gazebo_sim scout_mini_playpen.launch world_name:='$CITY_WORLD' spawn_x:='$SPAWN_X' spawn_y:='$SPAWN_Y' spawn_z:='$SPAWN_Z' spawn_yaw:='$SPAWN_YAW'" \
    Enter

# 窗格 1：键盘遥控。等待 Gazebo 和 ROS master 初始化。
tmux send-keys -t "$WINDOW.1" "export ROS_MASTER_URI=http://localhost:11311 ROS_HOSTNAME=localhost ROS_IP=127.0.0.1" Enter
tmux send-keys -t "$WINDOW.1" "source '$SCOUT_SETUP'" Enter
tmux send-keys -t "$WINDOW.1" "export PATH=/opt/ros/noetic/bin:/usr/bin:/bin:\$PATH; unset PYTHONHOME; hash -r" Enter
tmux send-keys -t "$WINDOW.1" "echo '等待 /cmd_vel 和相机话题...'" Enter
tmux send-keys -t "$WINDOW.1" \
    "until rostopic list 2>/dev/null | grep -qx '/camera/color/image_raw'; do sleep 1; done; rosrun teleop_twist_keyboard teleop_twist_keyboard.py" \
    Enter

# 窗格 2：预填录制命令，由用户按 Enter 正式开始。
tmux send-keys -t "$WINDOW.2" "export ROS_MASTER_URI=http://localhost:11311 ROS_HOSTNAME=localhost ROS_IP=127.0.0.1" Enter
tmux send-keys -t "$WINDOW.2" "source '$SCOUT_SETUP'" Enter
tmux send-keys -t "$WINDOW.2" "export PATH=/opt/ros/noetic/bin:/usr/bin:/bin:\$PATH; unset PYTHONHOME; hash -r" Enter
tmux send-keys -t "$WINDOW.2" "cd '$BAG_DIR'" Enter
tmux send-keys -t "$WINDOW.2" \
    "rosbag record /camera/color/image_raw -o '$1'"

echo "城市 Y 型场景：$CITY_WORLD"
echo "初始位姿：x=$SPAWN_X, y=$SPAWN_Y, spawn_z=$SPAWN_Z, yaw=$SPAWN_YAW"
echo "Bag 保存目录：$BAG_DIR"
echo "第三个窗格按 Enter 开始录制。"
echo "结束整个会话：tmux kill-session -t $SESSION_NAME"

tmux select-pane -t "$WINDOW.0"
tmux -2 attach-session -t "$SESSION_NAME"
