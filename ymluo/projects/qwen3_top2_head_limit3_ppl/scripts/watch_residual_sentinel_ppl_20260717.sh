#!/usr/bin/env bash
set -u

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
RESULT_ROOT="$ROOT/results/20260717_residual_sentinel_ppl_32k"
LOG_ROOT="$ROOT/logs"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

find_stable_free_gpu() {
  local index memory second_memory
  while true; do
    while IFS=, read -r index memory; do
      index="${index// /}"
      memory="${memory// /}"
      if [ "$memory" -le 100 ]; then
        sleep 20
        second_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$index" | tr -d ' ')
        if [ "$second_memory" -le 100 ]; then
          echo "$index"
          return 0
        fi
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    sleep 30
  done
}

run_topic() {
  local topic=$1
  local attempt=1
  local gpu output_dir log_file status
  output_dir="$RESULT_ROOT/${topic}_u8_c3"
  log_file="$LOG_ROOT/residual_sentinel_ppl_32k_${topic}.log"

  while [ "$attempt" -le 4 ]; do
    gpu=$(find_stable_free_gpu)
    printf '[%s] topic=%s attempt=%s gpu=%s\n' "$(date --iso-8601=seconds)" "$topic" "$attempt" "$gpu" >> "$log_file"
    PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PY" "$ROOT/src/run_adaptive_mass_budget_ppl_20260715.py" \
        --model_name_or_path "$MODEL" \
        --output_dir "$output_dir" \
        --topics "$topic" \
        --window_indices 0 \
        --history_tokens 32000 \
        --query_tokens 256 \
        --eval_tokens 256 \
        --window_stride_tokens 32512 \
        --mass_thresholds 0.000001 \
        --budget_fractions 0.02 \
        --mass_estimator qabs_sampled_tail \
        --sample_fraction 0.0025 \
        --qabs_dim_count 8 \
        --candidate_fraction 0.03 \
        --qabs_use_cuda_kernels \
        --qabs_score_mode pca_int4_residual_sentinel \
        --qabs_projection_dim 64 \
        --qabs_gqa_candidate_mode independent \
        --prefill_chunk_tokens 2048 \
        --dtype float16 \
        --device cuda \
        --device_map auto >> "$log_file" 2>&1
    status=$?
    if [ "$status" -eq 0 ] && [ -s "$output_dir/summary.json" ]; then
      printf '[%s] topic=%s complete\n' "$(date --iso-8601=seconds)" "$topic" >> "$log_file"
      return 0
    fi
    printf '[%s] topic=%s failed status=%s\n' "$(date --iso-8601=seconds)" "$topic" "$status" >> "$log_file"
    attempt=$((attempt + 1))
    sleep 60
  done
  return 1
}

run_topic sports || exit 1
run_topic medicine || exit 1
