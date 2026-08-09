#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "${PROJECT_ROOT}"

RUN_ROOT="${PROJECT_ROOT}/results/20260730_qksieve_keymse_deploy_256k_native_quality_multiwindow_7gpu" \
  bash scripts/launch_qksieve_256k_quality_multiwindow_7gpu_20260730.sh

RUN_ROOT="${PROJECT_ROOT}/results/20260730_qksieve_keymse_deploy_512k_extrap_retry_7gpu" \
  bash scripts/launch_qksieve_512k_extrap_quality_retry_7gpu_20260730.sh
