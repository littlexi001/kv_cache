#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/rabitqcache/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_lowbit_formats_v2}"
SCRIPT="${PROJECT_ROOT}/src/analyze_qksieve_lowbit_formats_20260731.py"

mkdir -p "${OUTPUT_ROOT}/logs"

jobs=(
  "0|qwen3_4b_sports32k|results/20260728_qksieve_all_layer_bits_qwen3_32k/traces/qwen3_4b_sports32k_all_layers.pt"
  "1|qwen3_4b_medicine32k|results/20260728_qksieve_all_layer_bits_qwen3_32k/traces/qwen3_4b_medicine32k_all_layers.pt"
  "2|qwen3_4b_sports96k|results/20260727_qk_balanced_96k_independent/traces/qwen3_4b_sports96k_32steps.pt"
  "3|qwen3_4b_medicine96k|results/20260727_qk_balanced_96k_independent/traces/qwen3_4b_medicine96k_32steps.pt"
  "4|llama31_8b_sports32k|results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt"
  "5|llama31_8b_medicine32k|results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_medicine.pt"
  "6|qwen25_7b_sports32k|results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_sports.pt"
  "7|qwen25_7b_medicine32k|results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_medicine.pt"
)

for job in "${jobs[@]}"; do
  IFS="|" read -r gpu label trace <<<"${job}"
  trace_path="${PROJECT_ROOT}/${trace}"
  output_dir="${OUTPUT_ROOT}/${label}"
  log_path="${OUTPUT_ROOT}/logs/${label}.log"
  if [[ ! -f "${trace_path}" ]]; then
    echo "SKIP missing trace: ${trace_path}" >&2
    continue
  fi
  if [[ -f "${output_dir}/summary.json" ]]; then
    echo "SKIP completed: ${label}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON}" "${SCRIPT}" \
    --trace_path "${trace_path}" \
    --output_dir "${output_dir}" \
    --label "${label}" \
    --families int_maxabs_native,int_lsq_native,int_lsq_3bit,minifloat_lsq \
    --budgets 96,128,160,192,240 \
    --top_fraction 0.02 \
    --selected_fractions 0.02,0.04,0.06 \
    >"${log_path}" 2>&1 &
  echo "${label}: GPU ${gpu}, PID $!, log ${log_path}"
done
