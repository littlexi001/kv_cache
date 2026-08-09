#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/20260727_hierarchical_dynamic_tail_ppl}"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

topics=(sports medicine sports medicine sports medicine sports medicine)
score_modes=(
  pca_hierarchical_842_stratified_mass
  pca_hierarchical_842_stratified_mass
  pca_hierarchical_842_stratified_mass
  pca_hierarchical_842_stratified_mass
  pca_hierarchical_841
  pca_hierarchical_841
  pca_hierarchical_841r50
  pca_hierarchical_841r50
)
targets=(0.94 0.94 0.95 0.95 0.95 0.95 0.95 0.95)
labels=(
  dynamic_t94_sports
  dynamic_t94_medicine
  dynamic_t95_sports
  dynamic_t95_medicine
  hier841_f06_sports
  hier841_f06_medicine
  hier841r50_f06_sports
  hier841r50_f06_medicine
)

for gpu in "${!labels[@]}"; do
  topic="${topics[$gpu]}"
  score_mode="${score_modes[$gpu]}"
  target="${targets[$gpu]}"
  label="${labels[$gpu]}"
  output="$OUTPUT_ROOT/$label"
  log="$LOG_DIR/$label.log"
  if [[ -s "$output/summary.json" ]]; then
    echo "skip complete: $label"
    continue
  fi

  common_args=(
    --model_name_or_path "$MODEL"
    --output_dir "$output"
    --topics "$topic"
    --window_indices 0
    --history_tokens 32000
    --query_tokens 64
    --eval_tokens 64
    --window_stride_tokens 32512
    --mass_thresholds "$target"
    --mass_estimator qabs_sampled_tail
    --qabs_dim_count 8
    --qabs_score_mode "$score_mode"
    --qabs_projection_dim 128
    --qabs_gqa_candidate_mode independent
    --prefill_chunk_tokens 1024
    --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn
    --dtype float16
    --device cuda
    --device_map auto
  )

  if [[ "$score_mode" == "pca_hierarchical_842_stratified_mass" ]]; then
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
      "${common_args[@]}" \
      --budget_fractions 0.005,0.01,0.02,0.03,0.04,0.06,0.08 \
      --sample_fraction 0.008 \
      --candidate_fraction 0.08 \
      --qabs_use_cuda_kernels \
      --qabs_partition_ucb_z 1.0 \
      --qabs_partition_overfetch_factor 1 \
      >"$log" 2>&1 &
  else
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
      "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
      "${common_args[@]}" \
      --budget_fractions 0.06 \
      --sample_fraction 0.0025 \
      --candidate_fraction 0.06 \
      --qabs_skip_candidate_rerank \
      >"$log" 2>&1 &
  fi
  echo "launched gpu=$gpu pid=$! label=$label"
done
