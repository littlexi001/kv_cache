#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_meantail_alpha05_weak3_6gpu}"
WORKER="${ROOT}/scripts/run_qksieve_valuesketch_alpha_pair_20260801.sh"
VARIANT="${VARIANT:-qksieve_keymse_requestlocal_meantail_k1280_c32}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u \
  src/benchmark_qksieve_valuesketch_direct_stages_20260801.py \
  --lengths 8192,16384,32768,65536,131072 \
  --warmup 12 \
  --iterations 60 \
  --tail_alpha 0.5 \
  --output "${RUN_ROOT}/direct_stages.json" \
  >"${RUN_ROOT}/logs/direct_stages.log" 2>&1 &
direct_pid="$!"

env ROOT="${ROOT}" RUN_ROOT="${RUN_ROOT}" GPU_IDS="2,3" ALPHA="0.5" \
  TOPICS="medicine:20260832" VARIANT="${VARIANT}" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/medicine_worker.log" 2>&1 &
medicine_pid="$!"

env ROOT="${ROOT}" RUN_ROOT="${RUN_ROOT}" GPU_IDS="4,5" ALPHA="0.5" \
  TOPICS="mixed_b:20260836" VARIANT="${VARIANT}" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/mixed_worker.log" 2>&1 &
mixed_pid="$!"

wait "${direct_pid}"
env ROOT="${ROOT}" RUN_ROOT="${RUN_ROOT}" GPU_IDS="0,1" ALPHA="0.5" \
  TOPICS="politics:20260834" VARIANT="${VARIANT}" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/politics_worker.log" 2>&1 &
politics_pid="$!"

status=0
for pid in "${medicine_pid}" "${mixed_pid}" "${politics_pid}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
