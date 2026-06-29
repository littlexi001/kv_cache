#!/usr/bin/env bash
set -euo pipefail

export PATH=/home/u21307130306/miniconda3/envs/cudatest/bin:${PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl
PY=/home/u21307130306/miniconda3/envs/cudatest/bin/python
MODEL=/home/u21307130306/kvcache_codex/kv_cache/fdong/Qwen3-0.6B

cd "${PROJECT_DIR}"

SINGLE_LAYER_DIR=outputs/icml_war80_eval200_single_layer_lm4096
BASELINE_CSV=outputs/icml_war80_eval200_baseline/ppl_by_mode.csv
MAP_DIR=outputs/icml_safe_layer_maps_lm4096s64

"${PY}" src/select_safe_layer_budget.py \
  --single_layer_dir "${SINGLE_LAYER_DIR}" \
  --baseline_csv "${BASELINE_CSV}" \
  --output_dir "${MAP_DIR}" \
  --layer_count 28 \
  --max_delta_ppl 0.20 \
  --max_candidates 8

run_candidate() {
  local gpu="$1"
  local compressed_count="$2"
  local full_count=$((28 - compressed_count))
  local map_path="${MAP_DIR}/safe_top${compressed_count}_layers_last.json"
  local out_dir="outputs/icml_war80_eval200_safe_top${compressed_count}_lm4096s64"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" src/evaluate_qwen3_top2_head_limit3_ppl.py \
    --model_name_or_path "${MODEL}" \
    --text_path data/war_and_peace_pg2600.txt \
    --output_dir "${out_dir}" \
    --prefill_tokens 80000 \
    --eval_tokens 200 \
    --chunk_size 512 \
    --eval_chunk_size 1 \
    --modes "fulll${full_count}landmarkr4096s64attn" \
    --full_layer_map_path "${map_path}" \
    --reuse_prefill_cache true \
    --protect_sink_tokens 0 \
    --disable_sparse_stats true \
    --log_every 1000 \
    --make_plots false < /dev/null > "${out_dir}/run.log" 2>&1
}

candidate_count=$(find "${MAP_DIR}" -maxdepth 1 -name 'safe_top*_layers_last.json' | wc -l)
if [[ "${candidate_count}" -eq 0 ]]; then
  echo "no candidate maps generated"
  exit 0
fi

batch_start=1
while [[ "${batch_start}" -le "${candidate_count}" ]]; do
  batch_end=$((batch_start + 7))
  if [[ "${batch_end}" -gt "${candidate_count}" ]]; then
    batch_end="${candidate_count}"
  fi
  for compressed_count in $(seq "${batch_start}" "${batch_end}"); do
    gpu=$((compressed_count - batch_start))
    run_candidate "${gpu}" "${compressed_count}" &
  done
  wait
  batch_start=$((batch_end + 1))
done

for csv_path in outputs/icml_war80_eval200_safe_top*_lm4096s64/ppl_by_mode.csv; do
  echo "===${csv_path}"
  cat "${csv_path}"
done
