#!/bin/bash

echo "🚀 正在启动自动导航系统..."

# 1. 激活环境并配置路径
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nomad_blackwell
export PYTHONPATH=/home/ljx/visualnav-transformer/diffusion_policy:/home/ljx/visualnav-transformer/train:$PYTHONPATH

# 2. 启动底盘肌肉 (后台运行)
echo "🦵 启动 PD 控制器..."
python pd_controller.py &
PID_PD=$!
sleep 2

# 3. 启动小脑舵手 (后台运行，接收你传入的所有参数)
echo "🧠 启动动态导航中枢，接收参数: $@"
python navigate_dynamic.py "$@" &
PID_NAV=$!
sleep 5  # 给 5060 一点加载模型的时间

# 4. 启动全局舰长 (解析传进来的 --dir 参数作为地图名)
# 提取 --dir 后面的值作为 map_name
MAP_NAME="online_map" # 默认值
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir|-d) MAP_NAME="$2"; shift ;;
    esac
    shift
done

echo "🗺️ 启动全局拓扑规划器，加载地图: $MAP_NAME"
python global_planner1.py --map "$MAP_NAME"
# 捕捉 Ctrl+C，一键关闭所有后台进程
trap "kill $PID_PD $PID_NAV; exit" SIGINT SIGTERM