#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
OUT=$ROOT/results/20260716_128k_low_peak_ablation
LOG=$ROOT/outputs/logs/20260716_128k_low_peak_ablation.log

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
mkdir -p "$OUT" "$(dirname "$LOG")"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3 taskset -c 0-23,48-71 "$PYTHON" \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --model_name_or_path "$MODEL" \
  --output "$OUT/religion_w0_offloaded_exact.json" \
  --topic religion \
  --history_tokens 128000 \
  --query_tokens 256 \
  --eval_tokens 256 \
  --window_index 0 \
  --window_stride_tokens 128512 \
  --projection_dim 64 \
  --index_bits 4 \
  --candidate_fraction 0.015 \
  --attention_fraction 0.015 \
  --candidate_selection_mode per_head_stream \
  --stream_group_size 2 \
  --candidate_refresh_interval 1 \
  --exact_cache_fraction 0.032 \
  --directory_backend fused \
  --prefill_cache_mode offloaded_exact \
  --prefill_chunk_tokens 4096 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  > "$LOG" 2>&1

"$PYTHON" src/summarize_128k_low_peak_ablation_20260716.py \
  --full results/20260716_128k_multitopic_windows_w3/religion_w0_full.json \
  --dynamic results/20260716_128k_multitopic_windows_w3/religion_w0_sparse.json \
  --offloaded "$OUT/religion_w0_offloaded_exact.json" \
  --output_dir "$OUT/summary" \
  >> "$LOG" 2>&1

touch "$OUT/COMPLETE"
