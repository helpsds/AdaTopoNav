#!/usr/bin/env bash

set -uo pipefail

# ============================================================
# 基础配置
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="${SCRIPT_DIR}/deploy_gnm_sim.sh"

MODEL="nomad"
TOPOMAP_DIR="NoMaD-Obstacle"

TOTAL_TESTS=20
TEST_TIMEOUT=600
POLL_INTERVAL=1

ROS_SETUP="/opt/ros/noetic/setup.bash"
SCOUT_WS_SETUP="/home/ljx/scout_ws/devel/setup.bash"

# Gazebo 中机器人的模型名称
GAZEBO_MODEL_NAME="scout/"

# 底盘速度话题。需要根据你的仿真确认
CMD_VEL_TOPIC="/cmd_vel"

# 碰撞话题。
# 如果机器人有 bumper/contact sensor，在这里填写具体话题。
# 留空时，只根据 navigate.py 日志中的碰撞关键词判断。
COLLISION_TOPIC="${COLLISION_TOPIC:-}"

# navigate.py 日志关键词
SUCCESS_REGEX="reached.*goal|goal.*reached|navigation.*success"
COLLISION_REGEX="collision|collided|bumper|contact detected|碰撞|撞击"
ERROR_REGEX="Traceback|CUDA out of memory|Segmentation fault|RuntimeError|Exception"

# ============================================================
# 初始位置
# 来自 Gazebo 中记录的 scout 位姿
# ============================================================

START_X="58.000000"
START_Y="49.000000"
START_Z="-0.072852"

START_QX="-0.000598761"
START_QY="0.083348180"
START_QZ="0.022235915"
START_QW="0.996272195"

# ============================================================
# 输出目录
# ============================================================

RESULT_DIR="${SCRIPT_DIR}/test_results/nomad_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="${RESULT_DIR}/summary.csv"

mkdir -p "$RESULT_DIR"

# ============================================================
# 运行状态
# ============================================================

DEPLOY_PID=""
CURRENT_SESSION=""
NAV_PANE=""

COLLISION_WATCH_PID=""
COLLISION_FLAG=""
COLLISION_WATCHER=""

NEW_SESSION=""
MONITOR_STATUS=""

# ============================================================
# ROS 环境
# ============================================================

if [[ ! -f "$ROS_SETUP" ]]; then
    echo "错误：未找到 ROS 环境：$ROS_SETUP"
    exit 1
fi

source "$ROS_SETUP"

if [[ -f "$SCOUT_WS_SETUP" ]]; then
    source "$SCOUT_WS_SETUP"
fi

# ============================================================
# 基础检查
# ============================================================

if [[ ! -f "$DEPLOY_SCRIPT" ]]; then
    echo "错误：未找到部署脚本：$DEPLOY_SCRIPT"
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "错误：未安装 tmux"
    exit 1
fi

if ! rostopic list >/dev/null 2>&1; then
    echo "错误：无法连接 ROS Master。"
    echo "请先手动启动 Gazebo 仿真环境。"
    exit 1
fi

if ! rosservice list 2>/dev/null |
    grep -qx "/gazebo/set_model_state"; then

    echo "错误：没有找到 /gazebo/set_model_state。"
    echo "请确认 Gazebo 已经正常启动。"
    exit 1
fi

if ! timeout 5 rosservice call /gazebo/get_model_state \
    "model_name: '${GAZEBO_MODEL_NAME}'
relative_entity_name: 'world'" 2>/dev/null |
    grep -q "success: True"; then

    echo "错误：Gazebo 中不存在模型：${GAZEBO_MODEL_NAME}"
    echo
    echo "可使用下面的命令确认模型名称："
    echo "  rostopic echo -n 1 /gazebo/model_states/name"
    exit 1
fi

# ============================================================
# 终止进程组
# ============================================================

kill_process_group() {
    local pid="${1:-}"

    [[ -z "$pid" ]] && return

    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return
    fi

    kill -INT -- "-${pid}" 2>/dev/null || true

    for _ in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return
        fi
        sleep 0.2
    done

    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 1

    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi

    wait "$pid" 2>/dev/null || true
}

# ============================================================
# 停止碰撞监听器
# ============================================================

stop_collision_watcher() {
    if [[ -n "$COLLISION_WATCH_PID" ]]; then
        kill "$COLLISION_WATCH_PID" 2>/dev/null || true
        wait "$COLLISION_WATCH_PID" 2>/dev/null || true
    fi

    COLLISION_WATCH_PID=""
}

# ============================================================
# 停止本轮部署
# 不关闭 Gazebo
# ============================================================

stop_deployment() {
    stop_collision_watcher

    if [[ -n "$CURRENT_SESSION" ]] &&
       tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then

        echo "关闭部署会话：$CURRENT_SESSION"
        tmux kill-session -t "$CURRENT_SESSION" 2>/dev/null || true
    fi

    if [[ -n "$DEPLOY_PID" ]]; then
        kill_process_group "$DEPLOY_PID"
    fi

    DEPLOY_PID=""
    CURRENT_SESSION=""
    NAV_PANE=""
}

# ============================================================
# Ctrl+C / Ctrl+Z：终止全部测试
# 只关闭导航，不关闭 Gazebo
# ============================================================

cleanup_all() {
    stop_deployment
}

handle_interrupt() {
    trap - INT TERM HUP TSTP EXIT

    echo
    echo "收到终止信号，结束全部测试。"

    cleanup_all

    echo "全部测试已停止，Gazebo 保持运行。"
    exit 130
}

handle_exit() {
    local exit_code=$?

    trap - EXIT
    cleanup_all

    exit "$exit_code"
}

trap handle_interrupt INT TERM HUP TSTP
trap handle_exit EXIT

# ============================================================
# 连续发送零速度
# 用于碰撞后暂停小车
# ============================================================

hold_robot_still() {
    local duration="${1:-3}"

    echo "向 ${CMD_VEL_TOPIC} 连续发送零速度 ${duration} 秒……"

    timeout "${duration}s" rostopic pub -r 20 \
        "$CMD_VEL_TOPIC" \
        geometry_msgs/Twist \
        "linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0" \
        >/dev/null 2>&1 || true
}

# ============================================================
# 只重置小车位置
#
# 不调用：
#   /gazebo/reset_world
#   /gazebo/reset_simulation
#
# 不修改箱子、地图或其他模型
# ============================================================

reset_robot_position() {
    echo "将 ${GAZEBO_MODEL_NAME} 重置到初始位置……"

    local result

    result="$(
        rosservice call /gazebo/set_model_state "
model_state:
  model_name: '${GAZEBO_MODEL_NAME}'
  pose:
    position:
      x: ${START_X}
      y: ${START_Y}
      z: ${START_Z}
    orientation:
      x: ${START_QX}
      y: ${START_QY}
      z: ${START_QZ}
      w: ${START_QW}
  twist:
    linear:
      x: 0.0
      y: 0.0
      z: 0.0
    angular:
      x: 0.0
      y: 0.0
      z: 0.0
  reference_frame: 'world'
" 2>&1
    )"

    if ! grep -qi "success: True" <<< "$result"; then
        echo "错误：小车位置重置失败。"
        echo "$result"
        return 1
    fi

    # 防止重置后残留速度
    hold_robot_still 1

    # 等待 Gazebo 物理状态稳定
    sleep 2

    return 0
}

# ============================================================
# 创建通用碰撞话题监听器
#
# 支持常见消息：
#   gazebo_msgs/ContactsState
#   std_msgs/Bool
#   带 states、contacts、data 字段的碰撞消息
# ============================================================

create_collision_watcher() {
    COLLISION_WATCHER="${RESULT_DIR}/collision_watcher.py"

    cat > "$COLLISION_WATCHER" <<'PYTHON'
#!/usr/bin/env python3

import os
import sys

import rospy
import rostopic


def is_collision(message) -> bool:
    if hasattr(message, "states"):
        return len(message.states) > 0

    if hasattr(message, "contacts"):
        return len(message.contacts) > 0

    if hasattr(message, "data"):
        return bool(message.data)

    if hasattr(message, "collision"):
        return bool(message.collision)

    return False


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: collision_watcher.py TOPIC FLAG_FILE")
        return 1

    topic = sys.argv[1]
    flag_file = sys.argv[2]

    rospy.init_node(
        "nomad_collision_watcher",
        anonymous=True,
        disable_signals=True,
    )

    message_class, real_topic, _ = rostopic.get_topic_class(
        topic,
        blocking=True,
    )

    if message_class is None:
        print(f"Cannot determine message type for {topic}")
        return 2

    def callback(message):
        if not is_collision(message):
            return

        try:
            with open(flag_file, "w", encoding="utf-8") as file:
                file.write("collision\n")
        finally:
            rospy.signal_shutdown("collision detected")

    rospy.Subscriber(
        real_topic,
        message_class,
        callback,
        queue_size=1,
    )

    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYTHON

    chmod +x "$COLLISION_WATCHER"
}

start_collision_watcher() {
    local flag_file="$1"

    stop_collision_watcher

    rm -f "$flag_file"

    if [[ -z "$COLLISION_TOPIC" ]]; then
        return 0
    fi

    if ! rostopic list 2>/dev/null |
        grep -Fxq "$COLLISION_TOPIC"; then

        echo "警告：碰撞话题不存在：$COLLISION_TOPIC"
        echo "本轮只根据 navigate.py 日志判断碰撞。"
        return 0
    fi

    echo "监听碰撞话题：$COLLISION_TOPIC"

    /usr/bin/python3 "$COLLISION_WATCHER" \
        "$COLLISION_TOPIC" \
        "$flag_file" \
        >"${flag_file}.log" 2>&1 &

    COLLISION_WATCH_PID=$!
}

# ============================================================
# 查找新的部署 tmux 会话
# ============================================================

find_new_session() {
    local before_sessions="$1"
    local deadline=$((SECONDS + 20))
    local session

    NEW_SESSION=""

    while (( SECONDS < deadline )); do
        while IFS= read -r session; do
            [[ -z "$session" ]] && continue

            if [[ "$session" == gnm_gazebo_* ]] &&
               ! grep -Fxq "$session" <<< "$before_sessions"; then

                NEW_SESSION="$session"
                return 0
            fi
        done < <(
            tmux list-sessions -F '#S' 2>/dev/null || true
        )

        sleep 1
    done

    return 1
}

# ============================================================
# 查找 navigate.py 所在窗格
# ============================================================

find_navigation_pane() {
    local pane
    local content

    NAV_PANE=""

    for _ in {1..20}; do
        if ! tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then
            return 1
        fi

        while IFS= read -r pane; do
            [[ -z "$pane" ]] && continue

            content="$(
                tmux capture-pane \
                    -p \
                    -S -100 \
                    -t "$pane" \
                    2>/dev/null || true
            )"

            if grep -q "navigate.py" <<< "$content"; then
                NAV_PANE="$pane"
                return 0
            fi
        done < <(
            tmux list-panes \
                -t "$CURRENT_SESSION" \
                -F '#{pane_id}' \
                2>/dev/null || true
        )

        sleep 1
    done

    # 原部署脚本中 navigate.py 通常在 pane 1
    if tmux capture-pane \
        -p \
        -t "${CURRENT_SESSION}:0.1" \
        >/dev/null 2>&1; then

        NAV_PANE="${CURRENT_SESSION}:0.1"
        return 0
    fi

    return 1
}

# ============================================================
# 启动本轮导航
# ============================================================

start_deployment() {
    local launcher_log="$1"
    local before_sessions

    before_sessions="$(
        tmux list-sessions -F '#S' 2>/dev/null || true
    )"

    echo "启动导航："
    echo "  bash ${DEPLOY_SCRIPT} --model ${MODEL} --dir ${TOPOMAP_DIR}"

    setsid bash "$DEPLOY_SCRIPT" \
        --model "$MODEL" \
        --dir "$TOPOMAP_DIR" \
        >"$launcher_log" 2>&1 < /dev/null &

    DEPLOY_PID=$!

    if ! find_new_session "$before_sessions"; then
        echo "错误：部署脚本没有创建新的 tmux 会话。"
        return 1
    fi

    CURRENT_SESSION="$NEW_SESSION"

    echo "部署会话：$CURRENT_SESSION"

    if ! find_navigation_pane; then
        echo "错误：没有找到 navigate.py 所在窗格。"
        return 1
    fi

    echo "navigate.py 窗格：$NAV_PANE"

    sleep 5
    return 0
}

# ============================================================
# 监控导航
#
# 同时检查：
# 1. 碰撞传感器话题
# 2. navigate.py 日志中的碰撞关键词
# 3. 成功关键词
# 4. Python 异常
# ============================================================

monitor_navigation() {
    local nav_log="$1"
    local collision_flag="$2"

    local snapshot="${nav_log}.snapshot"
    local deadline=$((SECONDS + TEST_TIMEOUT))

    MONITOR_STATUS="TIMEOUT"

    while (( SECONDS < deadline )); do
        if ! tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then
            MONITOR_STATUS="SESSION_EXITED"
            return
        fi

        if [[ -f "$collision_flag" ]]; then
            cp "$snapshot" "$nav_log" 2>/dev/null || true
            MONITOR_STATUS="COLLISION"
            return
        fi

        if ! tmux capture-pane \
            -p \
            -S -5000 \
            -t "$NAV_PANE" \
            >"$snapshot" 2>/dev/null; then

            sleep "$POLL_INTERVAL"
            continue
        fi

        cp "$snapshot" "$nav_log"

        if grep -Eiq "$COLLISION_REGEX" "$snapshot"; then
            MONITOR_STATUS="COLLISION"
            return
        fi

        if grep -Eiq "$SUCCESS_REGEX" "$snapshot"; then
            MONITOR_STATUS="SUCCESS"
            return
        fi

        if grep -Eiq "$ERROR_REGEX" "$snapshot"; then
            MONITOR_STATUS="ERROR"
            return
        fi

        sleep "$POLL_INTERVAL"
    done

    MONITOR_STATUS="TIMEOUT"
}

# ============================================================
# 初始化
# ============================================================

create_collision_watcher

echo "test,status,duration_seconds,model,topomap,navigation_log" \
    > "$SUMMARY_FILE"

SUCCESS_COUNT=0
COLLISION_COUNT=0
FAIL_COUNT=0

echo "============================================================"
echo "NoMaD Gazebo 20 次测试"
echo "模型：$MODEL"
echo "引导图：$TOPOMAP_DIR"
echo "机器人模型：$GAZEBO_MODEL_NAME"
echo "测试次数：$TOTAL_TESTS"
echo "单轮超时：${TEST_TIMEOUT}s"
echo "碰撞话题：${COLLISION_TOPIC:-未配置，仅检查日志}"
echo
echo "Gazebo 必须已经手动启动。"
echo "脚本不会重启或重置 Gazebo 场景。"
echo "每轮只重置 scout 的位置。"
echo "结果目录：$RESULT_DIR"
echo "============================================================"

# ============================================================
# 20 次测试
# ============================================================

for ((TEST_ID=1; TEST_ID<=TOTAL_TESTS; TEST_ID++)); do
    echo
    echo "============================================================"
    echo "开始第 ${TEST_ID}/${TOTAL_TESTS} 次测试"
    echo "============================================================"

    RUN_PREFIX="run_$(printf '%02d' "$TEST_ID")"

    NAV_LOG="${RESULT_DIR}/${RUN_PREFIX}_navigation.log"
    LAUNCHER_LOG="${RESULT_DIR}/${RUN_PREFIX}_launcher.log"
    COLLISION_FLAG="${RESULT_DIR}/${RUN_PREFIX}_collision.flag"

    stop_deployment

    # 每轮开始前只重置机器人
    if ! reset_robot_position; then
        echo "机器人位置重置失败，终止全部测试。"
        exit 1
    fi

    START_TIME=$(date +%s)
    STATUS="UNKNOWN"

    if ! start_deployment "$LAUNCHER_LOG"; then
        STATUS="DEPLOY_FAILED"
    else
        start_collision_watcher "$COLLISION_FLAG"

        monitor_navigation "$NAV_LOG" "$COLLISION_FLAG"
        STATUS="$MONITOR_STATUS"
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    case "$STATUS" in
        SUCCESS)
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "第 ${TEST_ID} 次测试成功，用时 ${DURATION}s。"

            stop_deployment
            ;;

        COLLISION)
            COLLISION_COUNT=$((COLLISION_COUNT + 1))

            echo "检测到碰撞，本轮测试失败。"

            # 先停止导航，防止控制器继续覆盖零速度
            stop_deployment

            echo "碰撞后暂停小车 3 秒……"
            hold_robot_still 3

            echo "碰撞暂停结束，恢复小车初始位置……"
            if ! reset_robot_position; then
                echo "机器人位置重置失败，终止全部测试。"
                exit 1
            fi
            ;;

        *)
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "第 ${TEST_ID} 次测试结束：${STATUS}，用时 ${DURATION}s。"

            stop_deployment
            ;;
    esac

    echo "${TEST_ID},${STATUS},${DURATION},${MODEL},${TOPOMAP_DIR},${NAV_LOG}" \
        >> "$SUMMARY_FILE"

    if (( TEST_ID < TOTAL_TESTS )); then
        sleep 2
    fi
done

# ============================================================
# 正常结束
# ============================================================

trap - EXIT INT TERM HUP TSTP

cleanup_all

SUCCESS_RATE="$(
    awk \
        -v success="$SUCCESS_COUNT" \
        -v total="$TOTAL_TESTS" \
        'BEGIN {
            if (total == 0) {
                printf "0.00"
            } else {
                printf "%.2f", success * 100.0 / total
            }
        }'
)"

echo
echo "============================================================"
echo "20 次测试全部结束"
echo "成功：${SUCCESS_COUNT}"
echo "碰撞：${COLLISION_COUNT}"
echo "其他失败：${FAIL_COUNT}"
echo "成功率：${SUCCESS_RATE}%"
echo "汇总文件：${SUMMARY_FILE}"
echo "日志目录：${RESULT_DIR}"
echo "Gazebo 保持运行"
echo "============================================================"