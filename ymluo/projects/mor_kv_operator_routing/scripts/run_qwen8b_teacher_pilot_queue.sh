#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/fdong/ymluo/projects/mor_kv_operator_routing}"
MODEL="${MODEL:-/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
CORPUS="${CORPUS:-/home/fdong/ymluo/projects/parallel_block_retrieval/data/real_longbench_docqa_10m_holdout64_v2}"
OUTPUT="${OUTPUT:-$BASE/outputs/qwen8b_head_teacher_8q_2k_v1}"
FREE_MIB="${FREE_MIB:-1000}"

while true; do
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v limit="$FREE_MIB" '{ gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 < limit) print $1 }'
  )
  if (( ${#free_gpus[@]} >= 2 )); then
    gpu_pair="${free_gpus[0]},${free_gpus[1]}"
    echo "$(date --iso-8601=seconds) launching on GPUs $gpu_pair"
    exec env CUDA_VISIBLE_DEVICES="$gpu_pair" \
      /home/fdong/miniconda3/envs/moe/bin/python \
      "$BASE/src/generate_head_distortion_teacher.py" \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS" \
      --output_dir "$OUTPUT" \
      --query_start 0 \
      --max_queries 8 \
      --max_context_tokens 2048 \
      --query_vector_tokens 1 \
      --block_tokens 256 \
      --budget_blocks 4 \
      --sink_blocks 1 \
      --recent_blocks 1 \
      --layers all \
      --dtype float16 \
      --attn_implementation sdpa \
      --device_map balanced
  fi
  echo "$(date --iso-8601=seconds) waiting for two GPUs below ${FREE_MIB} MiB"
  sleep 30
done
