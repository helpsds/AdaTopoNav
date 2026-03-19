#!/bin/bash

echo "🚀 正在启动真·拓扑导航系统 (图搜索版)..."

# 0. 提前保存原始参数，留给小脑使用
ORIGINAL_ARGS="$@"

# 提取 --dir 和 --goal 给舰长用 (设置默认值)
MAP_NAME="y_graph_map"
GOAL_NODE="-1"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir|-d) 
            MAP_NAME="$2"
            shift 2 
            ;;
        --goal|-g) 
            GOAL_NODE="$2"
            shift 2 
            ;;
        *) 
            # 遇到不认识的参数（比如 --model nomad）直接跳过
            shift 1 
            ;;
    esac
done

# 1. 激活环境并配置路径
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nomad_blackwell
export PYTHONPATH=/home/ljx/visualnav-transformer/diffusion_policy:/home/ljx/visualnav-transformer/train:$PYTHONPATH

# 2. 启动底盘肌肉 (后台运行)
echo "🦵 启动 PD 控制器..."
python pd_controller.py &
PID_PD=$!
sleep 2

# 3. 启动小脑舵手 (后台运行，原封不动接收你传入的所有参数)
echo "🧠 启动动态导航中枢 (NoMaD / GNM)..."
python navigate_dynamic.py $ORIGINAL_ARGS &
PID_NAV=$!
sleep 6  # 给 RTX 5060 多留一点加载双模型的时间

# 4. 捕捉 Ctrl+C 信号，确保退出时干掉所有挂起的进程
trap "echo -e '\n🛑 接收到退出指令，正在物理切断所有后台进程...'; kill -9 $PID_PD $PID_NAV; exit" SIGINT SIGTERM

# 5. 启动全局舰长 (前台运行，接管终端输出)
echo "🗺️ 启动全局拓扑规划器 (上帝视角)"
echo "   📍 载入地图: [$MAP_NAME] | 🎯 指定终点节点: [$GOAL_NODE]"
python global_planner.py --dir "$MAP_NAME" --goal "$GOAL_NODE"