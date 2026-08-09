#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
SCRIPT="${PROJECT}/src/analyze_rabitq_vs_spectral_index_20260727.py"
OUTPUT="${PROJECT}/results/20260727_rabitq_vs_spectral_trace"
TRACE32="${PROJECT}/results/20260726_qk_matrix_spectrum_multimodel_32k/traces"
TRACE25="${PROJECT}/results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces"
TRACE96="${PROJECT}/results/20260727_hierarchical_spectral_quantization_128k_traces/traces"

mkdir -p "${OUTPUT}/logs"

labels=(
  qwen3_4b_sports32k
  qwen3_4b_medicine32k
  llama31_8b_sports32k
  llama31_8b_medicine32k
  qwen25_7b_sports32k
  qwen25_7b_medicine32k
  qwen3_4b_sports96k
  qwen3_4b_medicine96k
)
traces=(
  "${TRACE32}/qwen3_4b_sports.pt"
  "${TRACE32}/qwen3_4b_medicine.pt"
  "${TRACE32}/llama31_8b_sports.pt"
  "${TRACE32}/llama31_8b_medicine.pt"
  "${TRACE25}/qwen25_7b_sports.pt"
  "${TRACE25}/qwen25_7b_medicine.pt"
  "${TRACE96}/qwen3_4b_sports96k.pt"
  "${TRACE96}/qwen3_4b_medicine96k.pt"
)

pids=()
for gpu in "${!labels[@]}"; do
  label="${labels[$gpu]}"
  trace="${traces[$gpu]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${SCRIPT}" \
    --trace_path "${trace}" \
    --output_dir "${OUTPUT}/${label}" \
    --label "${label}" \
    --device cuda \
    --calibration_steps 8 \
    --selected_fractions 0.02,0.03,0.04,0.06 \
    --top_fraction 0.02 \
    >"${OUTPUT}/logs/${label}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
