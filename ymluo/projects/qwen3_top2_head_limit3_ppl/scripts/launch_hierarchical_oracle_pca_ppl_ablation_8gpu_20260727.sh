#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_hierarchical_ppl_ablation}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

topics=(sports medicine sports medicine sports medicine sports medicine)
fractions=(0.04 0.04 0.04 0.04 0.06 0.06 0.03 0.03)
score_modes=(
  pca_hierarchical_842
  pca_hierarchical_842
  qabs
  qabs
  qabs
  qabs
  pca_int4
  pca_int4
)
labels=(
  hier842_f04_sports
  hier842_f04_medicine
  oracle_f04_sports
  oracle_f04_medicine
  oracle_f06_sports
  oracle_f06_medicine
  pca128i4_f03_sports
  pca128i4_f03_medicine
)

for gpu in "${!labels[@]}"; do
  topic="${topics[$gpu]}"
  fraction="${fractions[$gpu]}"
  score_mode="${score_modes[$gpu]}"
  label="${labels[$gpu]}"
  output="$OUTPUT_ROOT/$label"
  log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" || -s "$output/ppl_summary.json" ]]; then
    echo "skip complete: $label"
    continue
  fi

  if [[ "$score_mode" == "qabs" ]]; then
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      "$ROOT/src/run_head_top2_targeted_ppl_20260714.py" \
      --model_name_or_path "$MODEL" \
      --output_dir "$output" \
      --topics "$topic" \
      --window_indices 0 \
      --history_tokens 32000 \
      --query_tokens 64 \
      --eval_tokens 64 \
      --top_fraction "$fraction" \
      --prefill_chunk_tokens 1024 \
      --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      >"$log" 2>&1 &
  else
    projection_dim=128
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
      --model_name_or_path "$MODEL" \
      --output_dir "$output" \
      --topics "$topic" \
      --window_indices 0 \
      --history_tokens 32000 \
      --query_tokens 64 \
      --eval_tokens 64 \
      --window_stride_tokens 32512 \
      --mass_thresholds 0.95 \
      --budget_fractions "$fraction" \
      --mass_estimator qabs_sampled_tail \
      --sample_fraction 0.0025 \
      --qabs_dim_count 8 \
      --candidate_fraction "$fraction" \
      --qabs_skip_candidate_rerank \
      --qabs_score_mode "$score_mode" \
      --qabs_projection_dim "$projection_dim" \
      --qabs_gqa_candidate_mode independent \
      --prefill_chunk_tokens 1024 \
      --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      >"$log" 2>&1 &
  fi
  echo "launched gpu=$gpu pid=$! label=$label"
done
