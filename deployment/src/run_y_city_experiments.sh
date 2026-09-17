#!/usr/bin/env bash

# Run the complete AdaTopoNav/GNM ablation suite in the Y-shaped route of
# city_osm_roundabout_combined. Gazebo must already be running.
#
# Default:
#   20 trials per method, 240 s per trial
#   order: Ada_MVC, Ada_M, Ada_MV, Ada_MC, Original_gnm
#
# Usage:
#   bash run_y_city_experiments.sh
#
# Optional override example:
#   TRIALS=1 TIMEOUT_SECONDS=120 METHODS_CSV=ada_mvc \
#     bash run_y_city_experiments.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTION_ROOT="/home/ljx/gazebo_models_worlds_collection"
BOOKSHELF_SDF="$COLLECTION_ROOT/models/bookshelf_large/model.sdf"
SPAWN_MODEL="/opt/ros/noetic/lib/gazebo_ros/spawn_model"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
SCOUT_SETUP="${SCOUT_SETUP:-/home/ljx/scout_ws/devel/setup.bash}"

export ROS_MASTER_URI="${EXPERIMENT_ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${EXPERIMENT_ROS_HOSTNAME:-localhost}"
export ROS_IP="${EXPERIMENT_ROS_IP:-127.0.0.1}"
export GAZEBO_MODEL_PATH="$COLLECTION_ROOT/models:/home/ljx/scout_ws/src/ugv_gazebo_sim/cmu_worlds/models${GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}"

# shellcheck disable=SC1090
source "$ROS_SETUP"
if [[ -f "$SCOUT_SETUP" ]]; then
    # shellcheck disable=SC1090
    source "$SCOUT_SETUP"
fi

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*"
}

wait_for_gazebo() {
    local deadline=$((SECONDS + 60))
    until rosservice list 2>/dev/null | grep -qx '/gazebo/get_model_state'; do
        if ((SECONDS >= deadline)); then
            log "ERROR: Gazebo services are unavailable at $ROS_MASTER_URI."
            log "Start the city simulation first, then run this script."
            exit 1
        fi
        sleep 1
    done
}

bookshelf_exists() {
    local response
    response="$(
        rosservice call /gazebo/get_model_state \
            "model_name: 'bookshelf'
relative_entity_name: 'world'" 2>/dev/null || true
    )"
    grep -q 'success: True' <<<"$response"
}

set_bookshelf_pose() {
    local response
    response="$(
        rosservice call /gazebo/set_model_state "
model_state:
  model_name: 'bookshelf'
  pose:
    position: {x: -199.245000, y: -76.599500, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  twist:
    linear: {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}
  reference_frame: 'world'
" 2>&1
    )"
    grep -q 'success: True' <<<"$response" || {
        log "ERROR: failed to set bookshelf pose: $response"
        exit 1
    }
}

ensure_bookshelf() {
    [[ -f "$BOOKSHELF_SDF" ]] || {
        log "ERROR: bookshelf model is missing: $BOOKSHELF_SDF"
        exit 1
    }
    [[ -x "$SPAWN_MODEL" ]] || {
        log "ERROR: gazebo_ros spawn_model is missing: $SPAWN_MODEL"
        exit 1
    }

    if bookshelf_exists; then
        log "Bookshelf already exists; resetting its pose."
        set_bookshelf_pose
        return
    fi

    log "Inserting bookshelf at x=-199.245, y=-76.5995, z=0, yaw=0."
    # spawn_model uses /usr/bin/env python3. Invoke it with the system Python
    # explicitly so ROS can import the system netifaces package even when the
    # caller currently has a Conda environment active.
    /usr/bin/python3 "$SPAWN_MODEL" \
        -sdf \
        -file "$BOOKSHELF_SDF" \
        -model bookshelf \
        -x -199.245000 \
        -y -76.599500 \
        -z 0.0 \
        -Y 0.0

    bookshelf_exists || {
        log "ERROR: bookshelf insertion did not create model 'bookshelf'."
        exit 1
    }
}

wait_for_gazebo
ensure_bookshelf

# Y-city maps and evaluation geometry. Environment variables supplied by the
# caller still take precedence over these defaults.
export ENVIRONMENT="${ENVIRONMENT:-Y_city}"
export GNM_MAP="${GNM_MAP:-Y_gnm}"
export ADA_MAP="${ADA_MAP:-Y_ours}"
export GNM_GOAL_NODE="${GNM_GOAL_NODE:--1}"
export ADA_GOAL_NODE="${ADA_GOAL_NODE:--1}"

# The Gazebo pose panel reported yaw=-0.040445, but that orientation points
# toward +X, opposite to the recorded route (the first graph edge points
# mainly toward -X). Rotate the robot by pi so its front faces the route:
# yaw = -0.040445 + pi = 3.101147653589793.
export START_X="${START_X:--181.007352}"
export START_Y="${START_Y:--61.859492}"
export START_Z="${START_Z:-0.180565}"
export START_QX="${START_QX:-0.000016148984682971143}"
export START_QY="${START_QY:-0.001119902199159081}"
export START_QZ="${START_QZ:-0.9997949049750782}"
export START_QW="${START_QW:-0.020221116293348083}"

# Last node in Y_ours_poses.pt.
export GOAL_X="${GOAL_X:--198.39313806520576}"
export GOAL_Y="${GOAL_Y:--74.32210115215062}"
export GOAL_DISTANCE="${GOAL_DISTANCE:-3.0}"

# Initial pose -> first graph node (1.641605 m), plus all graph edges
# (23.250385 m).
export SHORTEST_PATH_DISTANCE="${SHORTEST_PATH_DISTANCE:-24.891989827296374}"
export EXPERIMENT_OBSTACLE="${EXPERIMENT_OBSTACLE:-bookshelf@(-199.245000,-76.599500,0.0),yaw=0}"

export TRIALS="${TRIALS:-20}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
export METHODS_CSV="${METHODS_CSV:-ada_mvc,ada_m,ada_mv,ada_mc,original}"

log "Starting Y-city experiments with maps $ADA_MAP and $GNM_MAP."
log "Robot start: ($START_X, $START_Y); goal: ($GOAL_X, $GOAL_Y)."
log "Bookshelf: (-199.245000, -76.599500, 0), yaw=0."

exec bash "$SCRIPT_DIR/run_navigation_experiments.sh"
