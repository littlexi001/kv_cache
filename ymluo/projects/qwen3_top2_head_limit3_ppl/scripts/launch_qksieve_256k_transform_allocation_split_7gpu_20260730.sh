#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"

export PROJECT_ROOT
export PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
export TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
export RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_256k_transform_allocation_split_4window_7gpu}"
export HISTORY_TOKENS="${HISTORY_TOKENS:-262080}"
export EVAL_TOKENS="${EVAL_TOKENS:-64}"
export VARIANTS="${VARIANTS:-exact_qk_oracle_k1280,qksieve_keymse_fulltopk_k1280,qksieve_keymse_frozenbasis_realloc_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_fulltopk_k1280,qksieve_keymse_requestlocal_fulltopk_k1280}"

exec bash \
  "${PROJECT_ROOT}/scripts/launch_qksieve_keymse_256k_multiwindow_7gpu_20260730.sh"
