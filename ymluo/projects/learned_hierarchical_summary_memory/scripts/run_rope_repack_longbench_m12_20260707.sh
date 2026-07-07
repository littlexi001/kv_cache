#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?gpu id}"
TOP_K="${2:?top k}"
OUTPUT_DIR="${3:?output dir}"
LOG_PATH="${4:?log path}"

mkdir -p "$(dirname "$LOG_PATH")"
CUDA_VISIBLE_DEVICES="$GPU_ID" /home/fdong/miniconda3/envs/moe/bin/python \
  /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_benchmark.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path /home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --longbench_tasks hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count \
  --ruler_tasks "" \
  --ruler_context_lengths "" \
  --max_examples_per_task 12 \
  --max_context_tokens 4096 \
  --page_tokens 512 \
  --top_k "$TOP_K" \
  --max_new_tokens_exact 48 \
  --max_new_tokens_summary 120 \
  --dtype float16 \
  --attn_implementation sdpa \
  --seed 2026070705 \
  2>&1 | tee "$LOG_PATH"
