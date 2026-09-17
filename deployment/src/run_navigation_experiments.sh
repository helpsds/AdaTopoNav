#!/usr/bin/env bash

# Run the original deployment and the AdaTopoNav V/C ablations against one
# already-running Gazebo world. Gazebo is never restarted; only model "scout"
# is reset between paired trials.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOYMENT_DIR}/.." && pwd)"
MAP_ROOT="${DEPLOYMENT_DIR}/topomaps"

# Several original deployment files resolve config paths relative to
# deployment/src, so make the working directory explicit.
cd "${SCRIPT_DIR}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
SCOUT_SETUP="${SCOUT_SETUP:-/home/ljx/scout_ws/devel/setup.bash}"
CONDA_SETUP="${CONDA_SETUP:-/home/ljx/miniconda3/etc/profile.d/conda.sh}"
PYTHON_BIN="${PYTHON_BIN:-/home/ljx/miniconda3/envs/nomad_blackwell/bin/python}"

# Equivalent to the interactive `ros_sim` alias. Shell aliases are not
# expanded inside this non-interactive script.
export ROS_MASTER_URI="${EXPERIMENT_ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${EXPERIMENT_ROS_HOSTNAME:-localhost}"
export ROS_IP="${EXPERIMENT_ROS_IP:-127.0.0.1}"

TRIALS="${TRIALS:-20}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
# All navigation comparisons use the same GNM backbone. Override MODEL only
# for a separate, explicitly labelled experiment.
MODEL="${MODEL:-gnm}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
WAYPOINT_INDEX="${WAYPOINT_INDEX:-2}"
CLOSE_THRESHOLD="${CLOSE_THRESHOLD:-3}"
LOCALIZATION_RADIUS="${LOCALIZATION_RADIUS:-4}"

ENVIRONMENT="${ENVIRONMENT:-L_world}"
GNM_MAP="${GNM_MAP:-L_GNM}"
# The adaptive map is stored on disk as "L_ours" (Linux paths are
# case-sensitive), with images plus vectors/poses/edges .pt files.
ADA_MAP="${ADA_MAP:-L_ours}"
GNM_GOAL_NODE="${GNM_GOAL_NODE:--1}"
ADA_GOAL_NODE="${ADA_GOAL_NODE:--1}"

# Physical goal and graph shortest-path distance come from the last node of
# the current L_ours adaptive map.
GOAL_X="${GOAL_X:-2.9081596863362345}"
GOAL_Y="${GOAL_Y:-14.005089781381477}"
GOAL_DISTANCE="${GOAL_DISTANCE:-3.0}"
SHORTEST_PATH_DISTANCE="${SHORTEST_PATH_DISTANCE:-15.86648042876158}"

COLLISION_TOPIC="${COLLISION_TOPIC:-/scout/has_obstacle_contact}"
COLLISION_DEBOUNCE="${COLLISION_DEBOUNCE:-0.5}"
RECOVERY_REVERSE_SECONDS="${RECOVERY_REVERSE_SECONDS:-0.8}"
RECOVERY_TURN_SECONDS="${RECOVERY_TURN_SECONDS:-1.2}"
RECOVERY_LINEAR_SPEED="${RECOVERY_LINEAR_SPEED:-0.12}"
RECOVERY_ANGULAR_SPEED="${RECOVERY_ANGULAR_SPEED:-0.35}"
RECOVERY_COOLDOWN="${RECOVERY_COOLDOWN:-1.0}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom}"
CMD_VEL_TOPIC="${CMD_VEL_TOPIC:-/cmd_vel}"
MODEL_NAME="${MODEL_NAME:-scout}"

# Initial pose supplied from Gazebo. Quaternion is calculated from
# roll=0.002238, pitch=0, yaw=0.023517.
START_X="${START_X:--0.30819023699107123}"
START_Y="${START_Y:-0.6546007247949986}"
START_Z="${START_Z:-0.174724}"
START_QX="${START_QX:-0.0011189224096101345}"
START_QY="${START_QY:-0.000013157455552443458}"
START_QZ="${START_QZ:-0.01175822168068999}"
START_QW="${START_QW:-0.9999302435982373}"

# Comma-separated subset may be supplied, for example:
# METHODS_CSV=original,ada_m,ada_mvc
METHODS_CSV="${METHODS_CSV:-ada_mvc,ada_m,ada_mv,ada_mc,original}"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/experiment_results/${RUN_STAMP}}"
RAW_CSV="${RESULT_DIR}/trials.csv"
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
MAP_CSV="${RESULT_DIR}/map_metrics.csv"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${RESULT_DIR}/runner.log") 2>&1

export PYTHONPATH="${REPO_ROOT}/diffusion_policy:${REPO_ROOT}/train:${REPO_ROOT}:${PYTHONPATH:-}"

# shellcheck disable=SC1090
source "${ROS_SETUP}"
if [[ -f "${SCOUT_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${SCOUT_SETUP}"
fi
if [[ -f "${CONDA_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SETUP}"
    conda activate nomad_blackwell
fi

TRIAL_PIDS=()
BRIDGE_PID=""
LAST_PID=""

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*"
}

topic_has_publisher() {
    local topic="$1"
    rostopic info "${topic}" 2>/dev/null |
        sed -n '/^Publishers:/,/^Subscribers:/p' |
        grep -q '^ \* '
}

wait_for_topic_publisher() {
    local topic="$1"
    local timeout_seconds="$2"
    local deadline=$((SECONDS + timeout_seconds))
    while (( SECONDS < deadline )); do
        if topic_has_publisher "${topic}"; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

wait_for_trial_topic_publisher() {
    local topic="$1"
    local timeout_seconds="$2"
    local deadline=$((SECONDS + timeout_seconds))
    local pid
    while (( SECONDS < deadline )); do
        if topic_has_publisher "${topic}"; then
            return 0
        fi
        for pid in "${TRIAL_PIDS[@]:-}"; do
            [[ -z "${pid}" ]] && continue
            if ! kill -0 "${pid}" 2>/dev/null; then
                log "ERROR: required trial process ${pid} exited before ${topic} was ready."
                return 2
            fi
        done
        sleep 0.2
    done
    return 1
}

start_process() {
    local log_file="$1"
    shift
    setsid "$@" >"${log_file}" 2>&1 &
    local pid=$!
    TRIAL_PIDS+=("${pid}")
    LAST_PID="${pid}"
}

stop_pid_group() {
    local pid="${1:-}"
    [[ -z "${pid}" ]] && return 0
    if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        return 0
    fi
    kill -INT -- "-${pid}" 2>/dev/null || true
    for _ in {1..15}; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}" 2>/dev/null || true
            return 0
        fi
        sleep 0.2
    done
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

stop_trial_processes() {
    local pid
    for pid in "${TRIAL_PIDS[@]:-}"; do
        stop_pid_group "${pid}"
    done
    TRIAL_PIDS=()
}

publish_zero_velocity() {
    # rostopic can ignore timeout's initial TERM while ROS transport is
    # wedged. Escalate to KILL after one second so trial cleanup cannot block
    # the entire experiment indefinitely.
    timeout --kill-after=1s 1.5s rostopic pub -r 20 \
        "${CMD_VEL_TOPIC}" geometry_msgs/Twist \
        '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
        >/dev/null 2>&1 || true
}

reset_robot() {
    publish_zero_velocity
    rosservice call /gazebo/pause_physics >/dev/null
    local response
    response="$(
        rosservice call /gazebo/set_model_state "
model_state:
  model_name: '${MODEL_NAME}'
  pose:
    position: {x: ${START_X}, y: ${START_Y}, z: ${START_Z}}
    orientation: {x: ${START_QX}, y: ${START_QY}, z: ${START_QZ}, w: ${START_QW}}
  twist:
    linear: {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}
  reference_frame: 'world'
" 2>&1
    )"
    rosservice call /gazebo/unpause_physics >/dev/null
    if ! grep -q 'success: True' <<<"${response}"; then
        log "ERROR: reset failed: ${response}"
        return 1
    fi
    publish_zero_velocity
    sleep 2
}

method_label() {
    case "$1" in
        original) printf 'Original_%s' "${MODEL}" ;;
        ada_m) printf 'Ada_M' ;;
        ada_mv) printf 'Ada_MV' ;;
        ada_mc) printf 'Ada_MC' ;;
        ada_mvc) printf 'Ada_MVC' ;;
        *) return 1 ;;
    esac
}

trial_is_complete() {
    local trial_id="$1"
    [[ -f "${RAW_CSV}" ]] || return 1
    awk -F, -v wanted="${trial_id}" '
        NR > 1 && $1 == wanted {
            # csv.writer emits CRLF records; strip the trailing CR before
            # deciding whether an interrupted trial is complete.
            reason = $15
            sub(/\r$/, "", reason)
            if (reason != "ROS_SHUTDOWN") found = 1
        }
        END { exit(found ? 0 : 1) }
    ' "${RAW_CSV}"
}

start_navigation() {
    local method="$1"
    local seed="$2"
    local prefix="$3"

    if [[ "${method}" == "original" ]]; then
        start_process "${prefix}_navigate.log" \
            "${PYTHON_BIN}" "${SCRIPT_DIR}/navigate.py" \
            --model "${MODEL}" \
            --dir "${GNM_MAP}" \
            --goal-node "${GNM_GOAL_NODE}" \
            --num-samples "${NUM_SAMPLES}" \
            --waypoint "${WAYPOINT_INDEX}" \
            --close-threshold "${CLOSE_THRESHOLD}" \
            --radius "${LOCALIZATION_RADIUS}" \
            --seed "${seed}" >/dev/null
        return
    fi

    local nav_flags=()
    local planner_flags=()
    case "${method}" in
        ada_m)
            nav_flags+=(--disable-waypoint-modulation)
            planner_flags+=(--disable-vlos)
            ;;
        ada_mv)
            nav_flags+=(--disable-waypoint-modulation)
            ;;
        ada_mc)
            planner_flags+=(--disable-vlos)
            ;;
        ada_mvc)
            ;;
        *)
            log "ERROR: unknown method ${method}"
            return 2
            ;;
    esac

    start_process "${prefix}_navigate_dynamic.log" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/navigate_dynamic.py" \
        --model "${MODEL}" \
        --dir "${ADA_MAP}" \
        --goal-node "${ADA_GOAL_NODE}" \
        --num-samples "${NUM_SAMPLES}" \
        --waypoint "${WAYPOINT_INDEX}" \
        --close-threshold "${CLOSE_THRESHOLD}" \
        --radius "${LOCALIZATION_RADIUS}" \
        --seed "${seed}" \
        "${nav_flags[@]}" >/dev/null

    start_process "${prefix}_global_planner.log" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/global_planner.py" \
        --dir "${ADA_MAP}" \
        --goal "${ADA_GOAL_NODE}" \
        --goal-distance "${GOAL_DISTANCE}" \
        --goal-confirmations 3 \
        --timeout "${TIMEOUT_SECONDS}" \
        "${planner_flags[@]}" >/dev/null
}

start_evaluator() {
    local method="$1"
    local label="$2"
    local seed="$3"
    local prefix="$4"
    local goal_topic="/topoplan/planner_goal_reached"
    if [[ "${method}" == "original" ]]; then
        goal_topic="/topoplan/reached_goal"
    fi

    local collision_args=()
    if [[ -n "${COLLISION_TOPIC}" ]]; then
        collision_args+=(
            --collision-topic "${COLLISION_TOPIC}"
            --collision-mode bool
            --collision-debounce "${COLLISION_DEBOUNCE}"
        )
    fi

    start_process "${prefix}_evaluator.log" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_navigation_trial.py" \
        --output "${RAW_CSV}" \
        --trial-id "${label}_${seed}" \
        --environment "${ENVIRONMENT}" \
        --method "${label}" \
        --seed "${seed}" \
        --goal-x "${GOAL_X}" \
        --goal-y "${GOAL_Y}" \
        --goal-distance "${GOAL_DISTANCE}" \
        --timeout "${TIMEOUT_SECONDS}" \
        --odom-topic "${ODOM_TOPIC}" \
        --goal-topic "${goal_topic}" \
        --shortest-path-distance "${SHORTEST_PATH_DISTANCE}" \
        "${collision_args[@]}"
}

wait_for_evaluator() {
    local evaluator_pid="$1"
    local deadline=$((SECONDS + TIMEOUT_SECONDS + 30))
    while kill -0 "${evaluator_pid}" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            log "Evaluator exceeded outer timeout; terminating it."
            stop_pid_group "${evaluator_pid}"
            return 1
        fi
        sleep 0.1
    done
    wait "${evaluator_pid}" 2>/dev/null || true
}

cleanup() {
    stop_trial_processes
    publish_zero_velocity
    if [[ -n "${BRIDGE_PID}" ]]; then
        stop_pid_group "${BRIDGE_PID}"
        BRIDGE_PID=""
    fi
}
trap cleanup EXIT INT TERM

validate_setup() {
    [[ -x "${PYTHON_BIN}" ]] || {
        log "ERROR: Python not found: ${PYTHON_BIN}"
        exit 1
    }
    [[ -d "${MAP_ROOT}/images/${GNM_MAP}" ]] || {
        log "ERROR: missing original map images: ${GNM_MAP}"
        exit 1
    }
    for suffix in vectors poses edges; do
        [[ -f "${MAP_ROOT}/${ADA_MAP}_${suffix}.pt" ]] || {
            log "ERROR: missing ${ADA_MAP}_${suffix}.pt"
            exit 1
        }
    done
    rostopic list >/dev/null 2>&1 || {
        log "ERROR: ROS master is unavailable at ${ROS_MASTER_URI}."
        log "Start Gazebo once in another terminal and keep it running."
        exit 1
    }
    rosservice list | grep -qx '/gazebo/set_model_state' || {
        log "ERROR: /gazebo/set_model_state is unavailable."
        exit 1
    }
    local model_response
    model_response="$(
        rosservice call /gazebo/get_model_state \
            "model_name: '${MODEL_NAME}'
relative_entity_name: 'world'" 2>/dev/null
    )"
    grep -q 'success: True' <<<"${model_response}" || {
        log "ERROR: Gazebo model '${MODEL_NAME}' was not found."
        exit 1
    }
    if [[ -n "${COLLISION_TOPIC}" ]] &&
        ! rostopic list | grep -Fxq "${COLLISION_TOPIC}"; then
        log "ERROR: collision topic missing: ${COLLISION_TOPIC}"
        exit 1
    fi
}

ensure_odometry() {
    if topic_has_publisher "${ODOM_TOPIC}"; then
        log "Using existing odometry publisher on ${ODOM_TOPIC}."
        return
    fi
    log "No ${ODOM_TOPIC} publisher; starting one Gazebo model-state bridge."
    setsid "${PYTHON_BIN}" "${SCRIPT_DIR}/gazebo_model_states_to_odom.py" \
        --model-name "${MODEL_NAME}" \
        --output-topic "${ODOM_TOPIC}" \
        >"${LOG_DIR}/odometry_bridge.log" 2>&1 &
    BRIDGE_PID=$!
    if ! wait_for_topic_publisher "${ODOM_TOPIC}" 15; then
        log "ERROR: odometry bridge did not publish ${ODOM_TOPIC}."
        exit 1
    fi
}

run_one_trial() {
    local method="$1"
    local seed="$2"
    local label
    label="$(method_label "${method}")"
    local prefix="${LOG_DIR}/$(printf '%02d' "${seed}")_${label}"

    if trial_is_complete "${label}_${seed}"; then
        log "SKIP completed seed=${seed}/${TRIALS} method=${label}"
        return
    fi

    log "START seed=${seed}/${TRIALS} method=${label}"
    stop_trial_processes
    reset_robot
    start_navigation "${method}" "${seed}" "${prefix}"

    if ! wait_for_trial_topic_publisher /waypoint 180; then
        log "ERROR: ${label} did not advertise /waypoint."
        log "Inspect logs with: tail -n 80 '${prefix}'_*.log"
        stop_trial_processes
        return 1
    fi
    log "${label}: navigation is ready; /waypoint publisher detected."

    local evaluator_pid
    start_evaluator "${method}" "${label}" "${seed}" "${prefix}"
    evaluator_pid="${LAST_PID}"
    log "${label}: evaluator started (pid=${evaluator_pid})."
    sleep 0.5

    start_process "${prefix}_controller.log" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/pd_controller.py" \
        --collision-topic "${COLLISION_TOPIC}" \
        --recovery-reverse-seconds "${RECOVERY_REVERSE_SECONDS}" \
        --recovery-turn-seconds "${RECOVERY_TURN_SECONDS}" \
        --recovery-linear-speed "${RECOVERY_LINEAR_SPEED}" \
        --recovery-angular-speed "${RECOVERY_ANGULAR_SPEED}" \
        --recovery-cooldown "${RECOVERY_COOLDOWN}" >/dev/null
    local controller_pid="${LAST_PID}"
    log "${label}: controller started; robot may now move."

    wait_for_evaluator "${evaluator_pid}" || true
    # Stop the command source first, then actively brake. This prevents the
    # controller from overwriting the emergency zero command while the other
    # navigation processes are shutting down.
    stop_pid_group "${controller_pid}"
    publish_zero_velocity
    stop_trial_processes
    publish_zero_velocity
    log "END seed=${seed}/${TRIALS} method=${label}"
    sleep 1
}

validate_setup
ensure_odometry

{
    echo "run_started=${RUN_STAMP}"
    echo "trials_per_method=${TRIALS}"
    echo "methods=${METHODS_CSV}"
    echo "model=${MODEL}"
    echo "original_map=${GNM_MAP}"
    echo "adaptive_map=${ADA_MAP}"
    echo "start_pose=${START_X},${START_Y},${START_Z},${START_QX},${START_QY},${START_QZ},${START_QW}"
    echo "goal=${GOAL_X},${GOAL_Y},radius=${GOAL_DISTANCE}"
    echo "obstacle=${EXPERIMENT_OBSTACLE:-none}"
    echo "timeout_seconds=${TIMEOUT_SECONDS}"
    echo "collision_topic=${COLLISION_TOPIC}"
    echo "collision_recovery=reverse:${RECOVERY_REVERSE_SECONDS}s@${RECOVERY_LINEAR_SPEED}mps,turn:${RECOVERY_TURN_SECONDS}s@${RECOVERY_ANGULAR_SPEED}radps,cooldown:${RECOVERY_COOLDOWN}s"
} >"${RESULT_DIR}/run_config.txt"

# Pair the same seed across methods and preserve the requested method order.
# By default, the complete Ada_MVC system runs first for every seed.
for ((seed = 1; seed <= TRIALS; seed++)); do
    for method in "${METHODS[@]}"; do
        run_one_trial "${method}" "${seed}"
    done
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_navigation_results.py" \
    --trials "${RAW_CSV}" \
    --summary "${SUMMARY_CSV}" \
    --map-root "${MAP_ROOT}" \
    --maps "${GNM_MAP}" "${ADA_MAP}" \
    --map-summary "${MAP_CSV}"

trap - EXIT INT TERM
cleanup
log "All experiments complete."
log "Raw trials: ${RAW_CSV}"
log "Summary: ${SUMMARY_CSV}"
log "Map metrics: ${MAP_CSV}"
