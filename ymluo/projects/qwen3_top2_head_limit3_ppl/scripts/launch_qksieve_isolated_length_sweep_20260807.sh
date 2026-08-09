#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/qksieve_isolated_length_sweep_r3_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
EVAL_TOKENS="${EVAL_TOKENS:-64}"
REPEATS="${REPEATS:-3}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10}"

mkdir -p "${RUN_ROOT}/logs"
{
  echo "purpose=single-process isolated frozen-QKSieve length sweep"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "repeats=${REPEATS}"
  echo "cooldown_seconds=${COOLDOWN_SECONDS}"
  echo "short_gpu=0"
  echo "long_gpus=0,1"
  echo "QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1"
  echo "QKSIEVE_BATCH_QMSE_ALLOCATION=1"
  echo "QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1"
  echo "QKSIEVE_TILED_VALUE_ATTENTION=0"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/qksieve_valuesketch_cuda_20260801.py" \
    "${ROOT}/scripts/launch_qksieve_optimized_profile_20260807.sh"
} >"${RUN_ROOT}/manifest.txt"

run_one() {
  local history="$1"
  local repeat="$2"
  local gpu="$3"
  local output="${RUN_ROOT}/n${history}/r${repeat}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}"
  ROOT="${ROOT}" \
  RUN_ROOT="${output}" \
  MODEL="${MODEL}" \
  GPU="${gpu}" \
  HISTORY_TOKENS="${history}" \
  EVAL_TOKENS="${EVAL_TOKENS}" \
  QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1 \
  QKSIEVE_TILED_VALUE_ATTENTION=0 \
    bash "${ROOT}/scripts/launch_qksieve_optimized_profile_20260807.sh" \
      >"${RUN_ROOT}/logs/n${history}_r${repeat}.log" 2>&1
  sleep "${COOLDOWN_SECONDS}"
}

for history in 8192 16384 32768 65536 131072; do
  gpu=0
  if [[ "${history}" == "131072" ]]; then
    gpu="0,1"
  fi
  for ((repeat=1; repeat<=REPEATS; repeat++)); do
    run_one "${history}" "${repeat}" "${gpu}"
  done
done

"${PYTHON}" "${ROOT}/src/summarize_qksieve_optimized_length_sweep_20260807.py" \
  "${RUN_ROOT}" >"${RUN_ROOT}/logs/summarize.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
