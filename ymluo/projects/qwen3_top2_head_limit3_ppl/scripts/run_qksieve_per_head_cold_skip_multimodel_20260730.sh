#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

root="results/20260730_qksieve_per_head_coldskip_multimodel_32k"
python_bin="/home/fdong/miniconda3/envs/moe/bin/python"
mkdir -p "${root}"

names=(
  qwen3_sports
  qwen3_medicine
  llama_sports
  llama_medicine
  qwen25_sports
  qwen25_medicine
)
traces=(
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_sports.pt
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/qwen3_4b_medicine.pt
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt
  results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_medicine.pt
  results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_sports.pt
  results/20260726_qwen25_7b_qk_matrix_spectrum_32k/traces/qwen25_7b_medicine.pt
)

for gpu in 0 1 2 3 4 5; do
  name="${names[${gpu}]}"
  output="${root}/${name}"
  mkdir -p "${output}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=src \
    "${python_bin}" \
    src/analyze_qksieve_per_head_cold_skip_20260730.py \
    --trace "${name}=${traces[${gpu}]}" \
    --output_dir "${output}" \
    --hot_fractions 0.10,0.15,0.25,0.40,0.50,0.60,0.75 \
    --recent_tokens 256 \
    --cold_shards 0,4,8,16,32 \
    --carry_previous 1 \
    >"${output}/run.log" 2>&1 &
  echo "${gpu} ${name} $!"
done
