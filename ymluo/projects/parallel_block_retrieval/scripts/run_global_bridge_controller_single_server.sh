#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/global_bridge_controller_q0_v1}"

mapfile -t FREE_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) <= 1024 && ($3 + 0) < 10) print $1}'
)
if ((${#FREE_GPUS[@]} < 2)); then
  echo "At least two idle GPUs are required; found ${#FREE_GPUS[@]}" >&2
  exit 1
fi
if ((${#FREE_GPUS[@]} > 8)); then
  FREE_GPUS=("${FREE_GPUS[@]:0:8}")
fi
GPU_LIST="$(IFS=,; echo "${FREE_GPUS[*]}")"
WORLD_SIZE="${#FREE_GPUS[@]}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

echo "Using idle GPUs: $GPU_LIST"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_DIR/torchrun" \
  --standalone \
  --nproc_per_node="$WORLD_SIZE" \
  src/run_global_bridge_controller_single.py \
  --corpus_dir data/real_longbench_docqa_10m_clean_record64 \
  --profile_dir outputs/real_longbench_docqa_10m_prerope_qk64_question16_profile \
  --output_dir "$OUTPUT_DIR" \
  --query_id "${QUERY_ID:-0}" \
  --svd_rank 32 \
  --candidate_blocks 512 \
  --target_blocks 3 \
  --search_hops "${SEARCH_HOPS:-1}"
