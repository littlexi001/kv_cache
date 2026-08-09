#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
HIGHBIT_TEMPLATE="${HIGHBIT_TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_all8_runtime.pt}"

if [[ ! -f "${HIGHBIT_TEMPLATE}" ]]; then
  "${PYTHON}" \
    "${PROJECT_ROOT}/src/rewrite_qksieve_template_allocation_20260729.py" \
    --input "${TEMPLATE}" \
    --output "${HIGHBIT_TEMPLATE}" \
    --allocation 8,8,8,8,8,8,8,8
fi

export PROJECT_ROOT
export PYTHON
export TEMPLATE
export HIGHBIT_TEMPLATE
export RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_256k_selector_cause_split_4window_7gpu}"
export HISTORY_TOKENS="${HISTORY_TOKENS:-262080}"
export EVAL_TOKENS="${EVAL_TOKENS:-64}"
export VARIANTS="${VARIANTS:-exact_qk_oracle_k1280,qksieve_keymse_fulltopk_k1280,qksieve_keymse_requestlocal_fulltopk_k1280,qksieve_keymse_highbit_fulltopk_k1280}"

exec bash \
  "${PROJECT_ROOT}/scripts/launch_qksieve_keymse_256k_multiwindow_7gpu_20260730.sh"
