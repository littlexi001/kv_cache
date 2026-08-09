#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
CORPUS="${ROOT}/data/real10m_controlled_v6_mix_seed20260715"
STEPS="${ROOT}/data/real10m_controlled_v6_mix_seed20260715_step_labels_v1/step_queries.jsonl"
ROWS="${ROOT}/outputs/real10m_controlled_v6_sparse_blocklocal_anchor_gated_sentence_devtest_v1/rows.jsonl"
OUTPUT="${ROOT}/outputs/real10m_controlled_v6_same_query_branch3_scaling_v1"

cd "$ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

while true; do
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) <= 1024 && ($3 + 0) < 10) print $1}'
  )
  if ((${#free_gpus[@]} >= 4)); then
    free_gpus=("${free_gpus[@]:0:4}")
    break
  fi
  echo "$(date -Is) waiting for 4 idle GPUs; found ${#free_gpus[@]}" >&2
  sleep 60
done

mkdir -p "$OUTPUT"
all_four="$(IFS=,; echo "${free_gpus[*]}")"
first_two="${free_gpus[0]},${free_gpus[1]}"
first_one="${free_gpus[0]}"

for world_size in 4 2 1; do
  if [[ -f "$OUTPUT/world${world_size}/summary.json" ]]; then
    echo "$(date -Is) world_size=$world_size already complete"
    continue
  fi
  case "$world_size" in
    4) gpu_list="$all_four" ;;
    2) gpu_list="$first_two" ;;
    1) gpu_list="$first_one" ;;
  esac
  echo "$(date -Is) world_size=$world_size GPUs=$gpu_list"
  CUDA_VISIBLE_DEVICES="$gpu_list" "$PYTHON_DIR/torchrun" --standalone \
    --nproc_per_node="$world_size" \
    src/benchmark_same_query_branch_parallel.py \
    --corpus_dir "$CORPUS" \
    --step_queries_path "$STEPS" \
    --retrieval_rows_path "$ROWS" \
    --output_dir "$OUTPUT/world${world_size}" \
    --split test \
    --step_types resolve_answer_from_bridge \
    --exclude_query_ids 375 \
    --max_queries 8 \
    --branches 3 \
    --max_new_tokens 12 \
    --device cuda
done

"$PYTHON_DIR/python" src/analyze_same_query_branch_scaling.py \
  --summary_paths "$OUTPUT/world1/summary.json,$OUTPUT/world2/summary.json,$OUTPUT/world4/summary.json" \
  --output_path "$OUTPUT/scaling.json"

echo "$(date -Is) strict scaling complete: $OUTPUT/scaling.json"
