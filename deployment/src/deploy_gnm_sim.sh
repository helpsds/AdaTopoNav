#!/bin/bash
export PYTHONPATH=/home/ljx/visualnav-transformer/diffusion_policy:/home/ljx/visualnav-transformer/train:$PYTHONPATH
# 创建一个新的 tmux 会话
session_name="gnm_gazebo_$(date +%s)"
tmux new-session -d -s $session_name
# 将窗口分为四个窗格
tmux selectp -t 0    # 选择第一个 (0) 窗格
tmux splitw -h -p 50 # 左右平分
tmux selectp -t 0    # 回到左侧
tmux splitw -v -p 50 # 左侧上下平分

tmux selectp -t 2    # 选择右侧窗格
tmux splitw -v -p 50 # 右侧上下平分
tmux selectp -t 0    # 回到左上角

# 窗格 0：提示窗口（替换掉原来报错的物理硬件 Launch）
tmux select-pane -t 0
tmux send-keys "echo '🟢 【仿真模式】已跳过启动真实硬件底盘'" Enter
tmux send-keys "echo '👉 请确保 Gazebo 仿真已经单独启动，且小车在起点！'" Enter

# 窗格 1：运行 navigate.py 大脑（补充了你的深度学习环境）
tmux select-pane -t 1
tmux send-keys "conda activate nomad_blackwell" Enter
tmux send-keys "python navigate.py $*" Enter

# 窗格 2：运行键盘控制（替换掉物理摇杆，方便你随时接管小车）
tmux select-pane -t 2
tmux send-keys "conda deactivate" Enter
tmux send-keys "source /opt/ros/noetic/setup.bash" Enter
tmux send-keys "rosrun teleop_twist_keyboard teleop_twist_keyboard.py" Enter

# 窗格 3：运行 pd_controller.py 神经（控制底盘运动）
tmux select-pane -t 3
tmux send-keys "conda activate nomad_blackwell" Enter
tmux send-keys "python pd_controller.py" Enter

# 附加到 tmux 会话并开启鼠标支持
tmux set-option -g mouse on
tmux -2 attach-session -t $session_name