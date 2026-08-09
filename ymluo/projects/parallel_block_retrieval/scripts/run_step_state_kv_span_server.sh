#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
CORPUS_DIR="${CORPUS_DIR:-${ROOT}/data/synthetic_controlled_100k_500_v4_blind}"
STEP_LABELS="${STEP_LABELS:-${ROOT}/data/synthetic_controlled_100k_500_v4_blind_step_labels_v5/step_queries.jsonl}"
K_PROFILE_DIR="${K_PROFILE_DIR:-${ROOT}/outputs/synthetic_controlled_v4_stepkv_layers4_svd32_profile_v1}"
STEP_Q_DIR="${STEP_Q_DIR:-${ROOT}/outputs/synthetic_controlled_v4_stepkv_layers4_stepq_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/synthetic_controlled_v4_stepkv_span_retrieval_v1}"

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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

echo "Using idle GPUs: $GPU_LIST"
if [[ ! -f "$K_PROFILE_DIR/summary.json" ]]; then
  rm -rf "$K_PROFILE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_DIR/torchrun" \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    src/profile_all_head_qk.py \
    --corpus_dir "$CORPUS_DIR" \
    --profile_dir "$K_PROFILE_DIR" \
    --layers 3,6,16,21 \
    --svd_rank 32 \
    --calibration_blocks 32 \
    --skip_query_profiles \
    --dtype float16
fi

if [[ ! -f "$STEP_Q_DIR/step_query_profiles.pt" ]]; then
  rm -rf "$STEP_Q_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_DIR/torchrun" \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    src/profile_step_state_q.py \
    --base_profile_dir "$K_PROFILE_DIR" \
    --step_queries_path "$STEP_LABELS" \
    --output_dir "$STEP_Q_DIR" \
    --splits train,dev,test \
    --task_types multihop \
    --query_vector_tokens 16 \
    --dtype float16
fi

rm -rf "$OUTPUT_DIR"
"$PYTHON_DIR/python" src/run_step_state_kv_span_retrieval.py \
  --corpus_dir "$CORPUS_DIR" \
  --profile_dir "$K_PROFILE_DIR" \
  --step_query_profiles "$STEP_Q_DIR/step_query_profiles.pt" \
  --output_dir "$OUTPUT_DIR" \
  --span_mode sentence \
  --window_tokens 32 \
  --window_stride 8 \
  --specialist_heads 8 \
  --exclude_query_ids 375

echo "Results: $OUTPUT_DIR/summary.json"
