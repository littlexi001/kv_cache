#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260803_conditional_residual_crosschecks_6gpu_v1}"
ANALYZER="${ROOT}/src/analyze_qksieve_tail_partition_calibration_20260803.py"

LLAMA_128="${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt"
LLAMA_32_SPORTS="${ROOT}/results/20260717_real_qkv_traces_32k/sports.pt"
LLAMA_32_MEDICINE="${ROOT}/results/20260717_real_qkv_traces_32k/medicine.pt"
QWEN_32_SPORTS="${ROOT}/results/20260727_qkv_value_sensitive_32k/traces/qwen3_4b_sports_qkv.pt"
QWEN_32_MEDICINE="${ROOT}/results/20260727_qkv_value_sensitive_32k/traces/qwen3_4b_medicine_qkv.pt"

mkdir -p "${RUN_ROOT}/logs"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  mkdir -p "${RUN_ROOT}/${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ANALYZER}" \
    --output_dir "${RUN_ROOT}/${name}" \
    --device cuda \
    --top_k 1280 \
    --sample_counts 256 \
    --key_rate_budget 15 \
    --value_bits 4 \
    "$@" \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1 &
  LAST_PID="$!"
}

launch 0 llama128_r16 \
  --traces "${LLAMA_128}" \
  --block_sizes 256 \
  --conditional_dims 8,16,32 \
  --value_rank 16
pid0="${LAST_PID}"

launch 1 llama32_sports_medicine_r16 \
  --traces "${LLAMA_32_SPORTS},${LLAMA_32_MEDICINE}" \
  --block_sizes 256 \
  --conditional_dims 8,16,32 \
  --value_rank 16
pid1="${LAST_PID}"

launch 2 qwen32_sports_r16_m10 \
  --traces "${QWEN_32_SPORTS}" \
  --block_sizes 256 \
  --conditional_dims 8,16 \
  --value_rank 16 \
  --max_records_per_trace 10
pid2="${LAST_PID}"

launch 3 qwen32_medicine_r16_m10 \
  --traces "${QWEN_32_MEDICINE}" \
  --block_sizes 256 \
  --conditional_dims 8,16 \
  --value_rank 16 \
  --max_records_per_trace 10
pid3="${LAST_PID}"

launch 4 llama128_r32 \
  --traces "${LLAMA_128}" \
  --block_sizes 256 \
  --conditional_dims 8,16,32 \
  --value_rank 32
pid4="${LAST_PID}"

launch 5 llama128_r16_blockscale \
  --traces "${LLAMA_128}" \
  --block_sizes 256,1024,131072 \
  --conditional_dims 8 \
  --value_rank 16
pid5="${LAST_PID}"

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
