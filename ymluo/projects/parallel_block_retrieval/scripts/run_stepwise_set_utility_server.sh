#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/stepwise_set_utility_v4_test_v1}"

mapfile -t FREE_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) <= 1024 && ($3 + 0) < 10) print $1}'
)
if ((${#FREE_GPUS[@]} < 1)); then
  echo "At least one idle GPU is required; found 0" >&2
  exit 1
fi
if ((${#FREE_GPUS[@]} > 4)); then
  FREE_GPUS=("${FREE_GPUS[@]:0:4}")
fi
GPU_LIST="$(IFS=,; echo "${FREE_GPUS[*]}")"
WORLD_SIZE="${#FREE_GPUS[@]}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/rows_rank*.jsonl "$OUTPUT_DIR/rows.jsonl" "$OUTPUT_DIR/summary.json"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

echo "Using idle GPUs: $GPU_LIST"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_DIR/torchrun" \
  --standalone \
  --nproc_per_node="$WORLD_SIZE" \
  src/evaluate_stepwise_set_utility.py \
  --corpus_dir data/synthetic_controlled_100k_500_v4_blind \
  --step_queries_path data/synthetic_controlled_100k_500_v4_blind_step_labels_v5/step_queries.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --splits test \
  --task_types multihop \
  --max_steps "${MAX_STEPS:-0}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-32}" \
  --modes "${MODES:-}" \
  --dtype float16 \
  --device cuda
