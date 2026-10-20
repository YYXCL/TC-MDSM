#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${TC_MDSM_GPU:-0}"
for stage in generate train evaluate report; do
    python evaluate/tc_mdsm_v3_runner.py "$stage" "$@"
done
