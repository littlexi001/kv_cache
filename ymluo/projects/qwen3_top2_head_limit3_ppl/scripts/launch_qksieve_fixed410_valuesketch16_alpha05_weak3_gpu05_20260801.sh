#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_fixed410_valuesketch16_alpha05_weak3_gpu05}"
WORKER="${ROOT}/scripts/run_qksieve_valuesketch_alpha_pair_20260801.sh"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23/templates/llama31_global32_3domain_keymse_runtime.pt}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"

env \
  ROOT="${ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  TEMPLATE="${TEMPLATE}" \
  GPU_IDS="0,5" \
  ALPHA="0.5" \
  TOPICS="medicine:20260832 mixed_b:20260836 politics:20260834" \
  VARIANT="qksieve_fixed410_requestlocal_valuesketch16i4_sampled_k1280" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/worker.log" 2>&1

touch "${RUN_ROOT}/ALL_COMPLETE"
