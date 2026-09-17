#!/bin/bash

set -u

if [ "$#" -ne 2 ]; then
    echo "用法: $0 <topomap名称> <bag文件>"
    echo "示例: $0 L_test L_test_2026-07-30-10-23-09.bag"
    exit 2
fi

# Create a new tmux session
session_name="gnm_locobot_$(date +%s)"
tmux new-session -d -s "$session_name"

# ========== 新增：开启全局鼠标支持（点击切换窗格/滚轮/拖拽调整大小） ==========
tmux set-option -g mouse on

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run roscore in the first pane
tmux select-pane -t 0
tmux send-keys "conda deactivate" Enter
tmux send-keys "ros_sim" Enter
tmux send-keys "roscore" Enter


# Run the create_topoplan.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate nomad_blackwell" Enter
tmux send-keys "ros_sim" Enter
# tmux panes start concurrently. Do not start the collector until roscore is ready.
tmux send-keys "until rostopic list >/dev/null 2>&1; do sleep 0.2; done" Enter
tmux send-keys "python create_toposim.py --dt 1 --dir '$1' --image-topic /topomap_bag/image_raw" Enter

# Change the directory to ../topomaps/bags and run the rosbag play command in the third pane
tmux select-pane -t 2
tmux send-keys "mkdir -p ../topomaps/bags" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "conda deactivate" Enter
tmux send-keys "ros_sim" Enter
# Replay only the image topic. rosbag waits until create_toposim.py has
# registered its subscriber, so the beginning of the bag cannot be lost.
tmux send-keys "until rostopic list >/dev/null 2>&1; do sleep 0.2; done" Enter
tmux send-keys "rosbag play -r 1.5 --wait-for-subscribers '$2' --topics /camera/color/image_raw /camera/color/image_raw:=/topomap_bag/image_raw" Enter

# Attach to the tmux session
tmux -2 attach-session -t "$session_name"
