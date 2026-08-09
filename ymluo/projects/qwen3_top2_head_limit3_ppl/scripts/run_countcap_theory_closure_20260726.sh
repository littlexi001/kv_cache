#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/20260726_countcap_theory_closure}"
MULTI_TRACE_ROOT="${PROJECT_ROOT}/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
QWEN25_TRACE_ROOT="${PROJECT_ROOT}/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_ROOT}"

run_prefix_drift() {
  if [[ -s "${OUTPUT_ROOT}/prefix_drift/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" \
    src/analyze_countcap_prefix_drift_20260726.py \
    --trace "llama31_8b=sports=${MULTI_TRACE_ROOT}/llama31_8b_sports.pt" \
    --trace "llama31_8b=medicine=${MULTI_TRACE_ROOT}/llama31_8b_medicine.pt" \
    --trace "qwen3_4b=sports=${MULTI_TRACE_ROOT}/qwen3_4b_sports.pt" \
    --trace "qwen3_4b=medicine=${MULTI_TRACE_ROOT}/qwen3_4b_medicine.pt" \
    --trace "qwen25_7b=sports=${QWEN25_TRACE_ROOT}/qwen25_7b_sports.pt" \
    --trace "qwen25_7b=medicine=${QWEN25_TRACE_ROOT}/qwen25_7b_medicine.pt" \
    --output_dir "${OUTPUT_ROOT}/prefix_drift" \
    --device cuda
}

run_margin_certificate() {
  if [[ -s "${OUTPUT_ROOT}/margin_certificate/summary.json" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES=1 "${PYTHON}" \
    src/analyze_countcap_margin_certificate_20260726.py \
    --trace "llama31_8b=sports=${MULTI_TRACE_ROOT}/llama31_8b_sports.pt" \
    --trace "llama31_8b=medicine=${MULTI_TRACE_ROOT}/llama31_8b_medicine.pt" \
    --trace "qwen3_4b=sports=${MULTI_TRACE_ROOT}/qwen3_4b_sports.pt" \
    --trace "qwen3_4b=medicine=${MULTI_TRACE_ROOT}/qwen3_4b_medicine.pt" \
    --trace "qwen25_7b=sports=${QWEN25_TRACE_ROOT}/qwen25_7b_sports.pt" \
    --trace "qwen25_7b=medicine=${QWEN25_TRACE_ROOT}/qwen25_7b_medicine.pt" \
    --output_dir "${OUTPUT_ROOT}/margin_certificate" \
    --device cuda
}

run_budget_probe() {
  local gpu="$1"
  local budget="$2"
  local tag="${3:-budget${budget}}"
  local fraction
  if [[ -s "${OUTPUT_ROOT}/budget_probe_32k/${tag}/case_summary.json" ]]; then
    return
  fi
  fraction="$("${PYTHON}" -c "print(${budget} / 32000)")"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "${MODEL}" \
    --output_dir "${OUTPUT_ROOT}/budget_probe_32k/${tag}" \
    --topics mixed_a,mixed_b \
    --window_indices 0,1 \
    --methods direct_countcap \
    --history_tokens 32000 \
    --eval_tokens 128 \
    --target_anchor_tokens 128000 \
    --direct_fraction "${fraction}" \
    --direct_min_tokens 1 \
    --direct_max_tokens "${budget}" \
    --projection_dim 48 \
    --sample_count 256 \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --device cuda \
    --device_map auto
}

run_prefix_drift >"${OUTPUT_ROOT}/logs/prefix_drift.log" 2>&1 &
prefix_pid=$!
run_margin_certificate >"${OUTPUT_ROOT}/logs/margin_certificate.log" 2>&1 &
margin_pid=$!
(
  run_budget_probe 2 320
  run_budget_probe 2 960
  run_budget_probe 2 640 cross_gpu2_budget640
  run_budget_probe 2 1280 cross_gpu2_budget1280
) >"${OUTPUT_ROOT}/logs/budget_gpu2.log" 2>&1 &
budget_gpu2_pid=$!
(
  run_budget_probe 3 640
  run_budget_probe 3 1280
  run_budget_probe 3 320 cross_gpu3_budget320
  run_budget_probe 3 960 cross_gpu3_budget960
) >"${OUTPUT_ROOT}/logs/budget_gpu3.log" 2>&1 &
budget_gpu3_pid=$!

status=0
for pid in "${prefix_pid}" "${margin_pid}" "${budget_gpu2_pid}" "${budget_gpu3_pid}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
