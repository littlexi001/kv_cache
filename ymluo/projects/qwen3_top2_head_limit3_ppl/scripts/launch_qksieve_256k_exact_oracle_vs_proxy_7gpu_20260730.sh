#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
export PROJECT_ROOT
export RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_256k_exact_oracle_vs_proxy_k1280_k2560_4window_7gpu}"
export VARIANTS="${VARIANTS:-exact_qk_oracle_k1280,exact_qk_oracle_k2560,qksieve_keymse_fulltopk_k1280,qksieve_keymse_fulltopk_k2560}"

exec bash \
  "${PROJECT_ROOT}/scripts/launch_qksieve_keymse_256k_multiwindow_7gpu_20260730.sh"
