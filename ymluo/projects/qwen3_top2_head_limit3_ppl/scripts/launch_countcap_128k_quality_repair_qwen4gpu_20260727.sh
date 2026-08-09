#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
DATASET_CACHE=/home/fdong/ymluo/datasets/sklearn
RUN_ROOT=$ROOT/results/20260727_countcap_128k_quality_repair_qwen4gpu
RUNNER=$ROOT/src/run_direct_countcap_denseprompt_ppl_20260725.py
SAMPLED_MODE=pca_int4_chunked_logscale16_sampleq_direct_qkvfused_qprojscan_qkvsplitauto
STRICT_MODE=pca_int4_chunked_logscale16_autosplit

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN_ROOT/logs/launcher.log"
}

run_variant() {
  local gpus="$1"
  local label="$2"
  local score_mode="$3"
  local max_tokens="$4"
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
    --projection_dim 48 \
    --sample_count 256 \
    --direct_score_mode "$score_mode" \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --preallocated_cache_min_tokens 14000 \
    --dataset_cache_dir "$DATASET_CACHE" \
    --device_map balanced \
    >"$log_path" 2>&1
}

run_pair() {
  run_variant 0,1 "$1" "$2" "$3" &
  local pid_a=$!
  run_variant 2,3 "$4" "$5" "$6" &
  local pid_b=$!
  wait "$pid_a" "$pid_b"
}

run_pair \
  strict_k1280 "$STRICT_MODE" 1280 \
  strict_k1920 "$STRICT_MODE" 1920

run_pair \
  strict_k2560 "$STRICT_MODE" 2560 \
  sampled_k1920 "$SAMPLED_MODE" 1920

run_variant 0,1 sampled_k2560 "$SAMPLED_MODE" 2560

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
