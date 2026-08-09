#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/global_bridge_controller_holdout_h1_v1}"
INITIAL_RETRIEVER="${INITIAL_RETRIEVER:-qk}"

mapfile -t FREE_GPUS < <(
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $0); if (($2 + 0) <= 1024 && ($3 + 0) < 10) print $1}'
)
MIN_GPUS=2
if [[ "$INITIAL_RETRIEVER" == "question_bm25" ]]; then
  MIN_GPUS=1
fi
if ((${#FREE_GPUS[@]} < MIN_GPUS)); then
  echo "At least $MIN_GPUS idle GPU(s) are required; found ${#FREE_GPUS[@]}" >&2
  exit 1
fi
if ((${#FREE_GPUS[@]} > 8)); then
  FREE_GPUS=("${FREE_GPUS[@]:0:8}")
fi
GPU_LIST="$(IFS=,; echo "${FREE_GPUS[*]}")"
WORLD_SIZE="${#FREE_GPUS[@]}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/results.jsonl" "$OUTPUT_DIR/summary.json"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

echo "Using idle GPUs: $GPU_LIST"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_DIR/torchrun" \
  --standalone \
  --nproc_per_node="$WORLD_SIZE" \
  src/run_global_bridge_controller_batch.py \
  --corpus_dir data/real_longbench_docqa_10m_clean_record64 \
  --profile_dir outputs/real_longbench_docqa_10m_prerope_qk64_question16_profile \
  --output_dir "$OUTPUT_DIR" \
  --datasets "${DATASETS:-2wikimqa,hotpotqa,musique}" \
  --exclude_query_ids "${EXCLUDE_QUERY_IDS:-0,6}" \
  --max_queries "${MAX_QUERIES:-0}" \
  --initial_retriever "$INITIAL_RETRIEVER" \
  --controller_mode "${CONTROLLER_MODE:-forced_search}" \
  --bridge_channels "${BRIDGE_CHANNELS:-model_only}" \
  --final_prompt_mode "${FINAL_PROMPT_MODE:-bound_focus}" \
  --svd_rank 32 \
  --candidate_blocks 512 \
  --target_blocks 3 \
  --search_hops "${SEARCH_HOPS:-1}"
