#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TRACE="${TRACE:-${ROOT}/results/20260801_qksieve_qasper_reference_trace_m10_gpu4/traces/qasper__9fb085a1f47673d1907f2378c90843b4b6e8622a14fe1fa9__qksieve_fullprompt_auto_plain_fulltopk.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_qasper_worst_quantizer_objective_4gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

quantizers=(plain plain metric metric)
objectives=(key_mse qk_mse key_mse qk_mse)
pids=()
for gpu in 0 1 2 3; do
  quantizer="${quantizers[gpu]}"
  objective="${objectives[gpu]}"
  name="${quantizer}_${objective}"
  destination="${OUTPUT}/${name}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
      src/analyze_qksieve_output_risk_budget_20260803.py \
      --trace "${TRACE}" \
      --output_dir "${destination}" \
      --model_name_or_path "${MODEL}" \
      --device cuda \
      --fixed_top_k 1280 \
      --coverage_targets 0.95 \
      --coverage_histogram_bins 256 \
      --minimum_top_k 256 \
      --maximum_top_k 0 \
      --key_rate_budget 19 \
      --key_quantizer "${quantizer}" \
      --key_allocation_objective "${objective}" \
      --value_rank 16 \
      --value_bits 4 \
      --risk_bits 4 \
      --query_factor_source prefill \
      --query_factor_prefill_tokens 8 \
      --focus_mass_ladder \
      >"${OUTPUT}/logs/${name}.log" 2>&1
    touch "${destination}_COMPLETE"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit 1
fi
touch "${OUTPUT}/ALL_COMPLETE"
