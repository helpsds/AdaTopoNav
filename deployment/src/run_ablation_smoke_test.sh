#!/usr/bin/env bash

# Smoke-test the three partial AdaTopoNav ablations once each.
# Excluded intentionally:
#   ada_mvc  - full method
#   original - original GNM baseline

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_SCRIPT="${SCRIPT_DIR}/run_navigation_experiments.sh"

if [[ ! -f "${EXPERIMENT_SCRIPT}" ]]; then
    printf 'ERROR: experiment script not found: %s\n' "${EXPERIMENT_SCRIPT}" >&2
    exit 1
fi

# Callers may override the timeout, result directory, or other experiment
# settings in the environment. The method list and one-trial smoke-test scope
# remain fixed so this entry point cannot accidentally run the full suite.
export TRIALS=1
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
export METHODS_CSV="ada_m,ada_mv,ada_mc"

printf 'Ablation smoke test: Ada_M, Ada_MV, Ada_MC (one trial each)\n'
printf 'Timeout per method: %s seconds\n' "${TIMEOUT_SECONDS}"
printf 'Full Ada_MVC and Original_gnm are excluded.\n'

exec bash "${EXPERIMENT_SCRIPT}"
