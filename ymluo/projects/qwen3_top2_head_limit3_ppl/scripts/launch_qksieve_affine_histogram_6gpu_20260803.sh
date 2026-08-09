#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_affine_histogram_6gpu_v1}"
KEY_RATE_BUDGET="${KEY_RATE_BUDGET:-15}"
VALUE_RANK="${VALUE_RANK:-16}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local trace="$3"
  local model="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_output_risk_budget_20260803.py \
    --trace "${trace}" \
    --output_dir "${OUTPUT}/${name}" \
    --model_name_or_path "${model}" \
    --device cuda \
    --fixed_top_k 1280 \
    --coverage_targets 0.90,0.95,0.975,0.99 \
    --coverage_histogram_bins 64,128,256 \
    --minimum_top_k 256 \
    --maximum_top_k 0 \
    --key_rate_budget "${KEY_RATE_BUDGET}" \
    --value_rank "${VALUE_RANK}" \
    --value_bits 4 \
    --risk_bits 4 \
    --focus_mass_ladder \
    >"${OUTPUT}/logs/${name}.log" 2>&1
  touch "${OUTPUT}/${name}_COMPLETE"
}

run_case 0 religion4k \
  "${ROOT}/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt" \
  "${LLAMA_MODEL}" & pid0=$!
run_case 1 sports32k \
  "${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/sports.pt" \
  "${QWEN_MODEL}" & pid1=$!
run_case 2 medicine32k \
  "${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/medicine.pt" \
  "${QWEN_MODEL}" & pid2=$!
run_case 3 sports96k \
  "${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/sports.pt" \
  "${QWEN_MODEL}" & pid3=$!
run_case 4 medicine96k \
  "${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/medicine.pt" \
  "${QWEN_MODEL}" & pid4=$!
run_case 5 computer128k \
  "${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt" \
  "${LLAMA_MODEL}" & pid5=$!

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit 1
fi
touch "${OUTPUT}/ALL_COMPLETE"
