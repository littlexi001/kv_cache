#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_alpha05_fused_numeric_followup_6gpu}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/results/20260717_real_qkv_traces_32k}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

pids=()

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u \
  src/benchmark_qksieve_valuesketch_direct_stages_20260801.py \
  --lengths 8192,16384,32768,65536,131072 \
  --warmup 12 \
  --iterations 60 \
  --tail_alpha 0.5 \
  --output "${RUN_ROOT}/direct_stages_alpha0p5.json" \
  >"${RUN_ROOT}/logs/direct_stages.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -u \
  src/analyze_qksieve_gaussian_tail_tilt_20260801.py \
  --traces "${TRACE_ROOT}/sports.pt" \
  --ranks 4,8,16,32,64,128 \
  --fractions 0.01,0.02 \
  --alphas 0.5,1.0 \
  --output_dir "${RUN_ROOT}/gaussian_tail_sports" \
  >"${RUN_ROOT}/logs/gaussian_tail_sports.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=2 "${PYTHON}" -u \
  src/analyze_qksieve_gaussian_tail_tilt_20260801.py \
  --traces "${TRACE_ROOT}/medicine.pt" \
  --ranks 4,8,16,32,64,128 \
  --fractions 0.01,0.02 \
  --alphas 0.5,1.0 \
  --output_dir "${RUN_ROOT}/gaussian_tail_medicine" \
  >"${RUN_ROOT}/logs/gaussian_tail_medicine.log" 2>&1 &
pids+=("$!")

env \
  ROOT="${ROOT}" \
  RUN_ROOT="${RUN_ROOT}/model_fusedv5" \
  GPU_IDS="4,5" \
  ALPHA="0.5" \
  TOPICS="computer:20260833" \
  VARIANT="qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280" \
  bash scripts/run_qksieve_valuesketch_alpha_pair_20260801.sh \
  >"${RUN_ROOT}/logs/model_fusedv5.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
