#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
MEMORY_TOKENIZER="${MEMORY_TOKENIZER:-/home/fdong/models/Qwen3-4B-Instruct}"
COARSE_ROOT="${COARSE_ROOT:-$ROOT/outputs/longmemeval_10m_owner_bm25_top128_all500_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/longmemeval_10m_per_head_pca_reader_all500_v1}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"

mkdir -p "$COARSE_ROOT" "$OUTPUT_ROOT"

coarse_pids=()
for part in $(seq 0 7); do
  out="$COARSE_ROOT/part$part"
  mkdir -p "$out"
  if [[ ! -s "$out/rows.jsonl" ]]; then
    OMP_NUM_THREADS=4 "$PYTHON" "$ROOT/src/evaluate_longmemeval_10m_hierarchical_bm25.py" \
      --data_dir "$ROOT/data/longmemeval_10m_partition${part}_v1" \
      --output_dir "$out" \
      --model_name_or_path "$MEMORY_TOKENIZER" \
      --owner_depths 1 \
      --session_depths 3 \
      --semantic_recency_pools 8 \
      --topks 8,31,128 \
      >"$out/run.log" 2>&1 &
    coarse_pids+=("$!")
  fi
done
for pid in "${coarse_pids[@]}"; do
  wait "$pid"
done

if [[ ${#GPUS[@]} -gt 8 ]]; then
  echo "GPU_LIST can contain at most eight devices" >&2
  exit 2
fi

pids=()
for part in $(seq 0 7); do
  gpu="${GPUS[$((part % ${#GPUS[@]}))]}"
  out="$OUTPUT_ROOT/part$part"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 "$PYTHON" \
    "$ROOT/src/evaluate_longmemeval_10m_per_head_pca_reader.py" \
    --data_dir "$ROOT/data/longmemeval_10m_partition${part}_v1" \
    --coarse_rows "$COARSE_ROOT/part${part}/rows.jsonl" \
    --output_dir "$out" \
    --model_name_or_path "$MODEL" \
    --memory_tokenizer_name_or_path "$MEMORY_TOKENIZER" \
    --device cuda:0 \
    --dtype float16 \
    --coarse_blocks 128 \
    --final_blocks 31 \
    --hybrid_bm25_blocks 16 \
    --retrieval_token_budget 2000 \
    --max_new_tokens 48 \
    --projection_dim 64 \
    --layers 3,7,11,15,19,23,27,31 \
    --segments 4 \
    --calibration_blocks 256 \
    --profile_batch_size 8 \
    --query_tail_tokens 8 \
    --active_head_fraction 0.25 \
    --selected_head_channels 16 \
    --per_head_depth 8 \
    --include_page_dates \
    --page_order chronological \
    >"$out/run.log" 2>&1 &
  pids+=("$!")
  if (( ${#GPUS[@]} < 8 && (part + 1) % ${#GPUS[@]} == 0 )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON" "$ROOT/src/analyze_longmemeval_10m_per_head_pca_reader.py" \
  --input_dir "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/all500_summary.json" \
  >"$OUTPUT_ROOT/analyze.log" 2>&1

cat "$OUTPUT_ROOT/all500_summary.json"
