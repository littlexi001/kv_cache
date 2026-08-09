#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
DATASET_CACHE=/home/fdong/ymluo/datasets/sklearn
PARENT=$ROOT/results/20260727_countcap_128k_quality_repair_qwen4gpu
RUN_ROOT=$ROOT/results/20260727_countcap_128k_quality_repair_stage2_qwen4gpu
RUNNER=$ROOT/src/run_direct_countcap_denseprompt_ppl_20260725.py
SAMPLED_MODE=pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN_ROOT/logs/launcher.log"
}

log "queued behind $PARENT"
while [[ ! -f "$PARENT/ALL_COMPLETE" ]]; do
  if ! pgrep -f \
    "launch_countcap_128k_quality_repair_qwen4gpu_20260727.sh" \
    >/dev/null; then
    log "parent stopped without ALL_COMPLETE"
    exit 1
  fi
  sleep 60
done

run_direct() {
  local gpus="$1"
  local label="$2"
  local rank="$3"
  local max_tokens="$4"
  local recent_tokens="$5"
  local output="$RUN_ROOT/$label"
  local log_path="$RUN_ROOT/logs/$label.log"

  if [[ -s "$output/case_summary.json" ]] &&
    [[ -s "$output/token_results.csv" ]]; then
    log "skip completed $label"
    return
  fi

  mkdir -p "$output"
  log "run $label on GPU $gpus"
  CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON" -u "$RUNNER" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics mixed_a,mixed_b \
    --window_indices 0,1 \
    --methods direct_countcap \
    --history_tokens 128000 \
    --eval_tokens 256 \
    --window_stride_tokens 128512 \
    --target_anchor_tokens 128000 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens "$max_tokens" \
    --projection_dim "$rank" \
    --sample_count 1024 \
    --protect_recent_tokens "$recent_tokens" \
    --direct_score_mode "$SAMPLED_MODE" \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --preallocated_cache_min_tokens 14000 \
    --dataset_cache_dir "$DATASET_CACHE" \
    --device_map balanced \
    >"$log_path" 2>&1
}

run_exact_top2() {
  local output="$RUN_ROOT/exact_top2"
  if [[ -s "$output/case_summary.json" ]] &&
    [[ -s "$output/token_results.csv" ]]; then
    log "skip completed exact_top2"
    return
  fi
  mkdir -p "$output"
  log "run exact_top2 on GPU 0,1"
  CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -u "$RUNNER" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics mixed_a,mixed_b \
    --window_indices 0,1 \
    --methods exact_top2 \
    --history_tokens 128000 \
    --eval_tokens 256 \
    --window_stride_tokens 128512 \
    --target_anchor_tokens 128000 \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --preallocated_cache_min_tokens 14000 \
    --dataset_cache_dir "$DATASET_CACHE" \
    --device_map balanced \
    >"$RUN_ROOT/logs/exact_top2.log" 2>&1
}

run_recent_smoke() {
  local output="$RUN_ROOT/_smoke_recent"
  if [[ -s "$output/case_summary.json" ]]; then
    log "skip completed recent smoke"
    return
  fi
  mkdir -p "$output"
  log "run recent-quota smoke on GPU 0"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$RUNNER" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --topics mixed_a \
    --window_indices 0 \
    --methods direct_countcap \
    --history_tokens 2048 \
    --eval_tokens 8 \
    --window_stride_tokens 2560 \
    --target_anchor_tokens 2048 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 256 \
    --projection_dim 48 \
    --sample_count 1024 \
    --protect_recent_tokens 64 \
    --direct_score_mode "$SAMPLED_MODE" \
    --prefill_chunk_tokens 2048 \
    --cache_mode dynamic \
    --dataset_cache_dir "$DATASET_CACHE" \
    --device_map balanced \
    >"$RUN_ROOT/logs/_smoke_recent.log" 2>&1
}

run_recent_smoke

run_exact_top2 &
pid_a=$!
run_direct 2,3 rank48_sample1024_k1280 48 1280 0 &
pid_b=$!
wait "$pid_a" "$pid_b"

run_direct 0,1 rank48_sample1024_k1280_recent256 48 1280 256 &
pid_a=$!
run_direct 2,3 rank48_sample1024_k1280_recent512 48 1280 512 &
pid_b=$!
wait "$pid_a" "$pid_b"

run_direct 0,1 rank64_sample1024_k1280 64 1280 0 &
pid_a=$!
run_direct 2,3 rank64_sample1024_k1280_recent256 64 1280 256 &
pid_b=$!
wait "$pid_a" "$pid_b"

run_direct 0,1 rank64_sample1024_k1280_recent512 64 1280 512 &
pid_a=$!
run_direct 2,3 rank64_sample1024_k1920_recent256 64 1920 256 &
pid_b=$!
wait "$pid_a" "$pid_b"

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
