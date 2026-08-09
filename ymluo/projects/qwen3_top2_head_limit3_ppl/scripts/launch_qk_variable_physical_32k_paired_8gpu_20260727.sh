#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
RUN_ROOT=$ROOT/results/20260727_qk_variable_physical_32k_paired_8gpu
LOG_ROOT=$RUN_ROOT/logs
FACTORIAL=$ROOT/results/20260727_qkbalanced_allocation_scale_factorial_m20_5gpu
FACTORIAL_PATTERN='launch_qkbalanced_factorial_m20_5gpu_20260727.sh'

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$FACTORIAL/ALL_COMPLETE" ]]; do
  if ! pgrep -f "$FACTORIAL_PATTERN" >/dev/null; then
    echo "allocation/scale factorial exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

run_full() {
  local gpu="$1"
  local topic="$2"
  full_output=$RUN_ROOT/${topic}_full.json
  if [[ ! -s "$full_output" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu taskset -c 0-23,48-71 "$PYTHON" \
      src/run_full_cache_ppl_baseline_20260715.py \
      --model_name_or_path "$MODEL" \
      --output "$full_output" \
      --topic "$topic" \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 64 \
      --window_index 0 \
      --prefill_chunk_tokens 4096 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/${topic}_full.log" 2>&1
  fi
}

run_qkphysical() {
  local gpu="$1"
  local topic="$2"
  sparse_output=$RUN_ROOT/${topic}_qkphysical.json
  if [[ ! -s "$sparse_output" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu taskset -c 24-47,72-95 "$PYTHON" \
      src/run_hierarchical_physical_cache_ppl_20260715.py \
      --model_name_or_path "$MODEL" \
      --output "$sparse_output" \
      --topic "$topic" \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 64 \
      --window_index 0 \
      --index_mode qk_variable \
      --qk_metric_query_shrinkage 0.75 \
      --variable_rate_budget 15 \
      --candidate_fraction 0.06 \
      --candidate_min_tokens 256 \
      --candidate_max_tokens 1280 \
      --attention_fraction 0.06 \
      --candidate_selection_mode per_head_stream \
      --stream_group_size 1 \
      --exact_cache_fraction 0.032 \
      --directory_backend fused \
      --prefill_cache_mode offloaded_exact \
      --dense_query_before_conversion \
      --prefill_chunk_tokens 4096 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/${topic}_qkphysical.log" 2>&1
  fi
}

pids=()
(run_full 0 sports; run_qkphysical 0 religion) & pids+=("$!")
(run_full 1 medicine; run_qkphysical 1 computer) & pids+=("$!")
(run_full 2 computer; run_qkphysical 2 medicine) & pids+=("$!")
run_full 3 religion & pids+=("$!")
run_qkphysical 7 sports & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more paired 32K workers failed" >&2
  exit 1
fi

run_qksampled() {
  local gpu="$1"
  local topic="$2"
  sampled_output=$RUN_ROOT/${topic}_qksampled.json
  if [[ ! -s "$sampled_output" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu taskset -c 0-23,48-71 "$PYTHON" \
      src/run_hierarchical_physical_cache_ppl_20260715.py \
      --model_name_or_path "$MODEL" \
      --output "$sampled_output" \
      --topic "$topic" \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 64 \
      --window_index 0 \
      --index_mode qk_variable \
      --qk_metric_query_shrinkage 0.75 \
      --variable_rate_budget 15 \
      --candidate_fraction 0.06 \
      --candidate_min_tokens 256 \
      --candidate_max_tokens 1280 \
      --attention_fraction 0.06 \
      --candidate_selection_mode per_head_stream \
      --stream_group_size 1 \
      --retrieval_backend sampled_compact \
      --sampled_candidate_multiplier 1.5 \
      --exact_cache_fraction 0.032 \
      --directory_backend fused \
      --prefill_cache_mode offloaded_exact \
      --dense_query_before_conversion \
      --prefill_chunk_tokens 4096 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/${topic}_qksampled.log" 2>&1
  fi
}

run_pca64physical() {
  local gpu="$1"
  local topic="$2"
  pca_output=$RUN_ROOT/${topic}_pca64physical.json
  if [[ ! -s "$pca_output" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu taskset -c 24-47,72-95 "$PYTHON" \
      src/run_hierarchical_physical_cache_ppl_20260715.py \
      --model_name_or_path "$MODEL" \
      --output "$pca_output" \
      --topic "$topic" \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 64 \
      --window_index 0 \
      --projection_dim 64 \
      --index_bits 4 \
      --index_mode pca_fixed \
      --dense_query_before_conversion \
      --candidate_fraction 0.06 \
      --candidate_min_tokens 256 \
      --candidate_max_tokens 1280 \
      --attention_fraction 0.06 \
      --candidate_selection_mode per_head_stream \
      --stream_group_size 1 \
      --exact_cache_fraction 0.032 \
      --directory_backend fused \
      --prefill_cache_mode offloaded_exact \
      --prefill_chunk_tokens 4096 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      > "$LOG_ROOT/${topic}_pca64physical.log" 2>&1
  fi
}

second_wave_pids=()
(run_qksampled 0 sports; run_pca64physical 0 religion) &
second_wave_pids+=("$!")
(run_qksampled 1 medicine; run_pca64physical 1 computer) &
second_wave_pids+=("$!")
(run_qksampled 2 computer; run_pca64physical 2 medicine) &
second_wave_pids+=("$!")
run_qksampled 3 religion & second_wave_pids+=("$!")
run_pca64physical 7 sports & second_wave_pids+=("$!")
failed=0
for pid in "${second_wave_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more 32K ablation workers failed" >&2
  exit 1
fi

"$PYTHON" src/summarize_qk_variable_physical_paired_20260727.py \
  --input_dir "$RUN_ROOT" \
  --output_dir "$RUN_ROOT/summary" \
  > "$LOG_ROOT/summary.log" 2>&1
touch "$RUN_ROOT/ALL_COMPLETE"
