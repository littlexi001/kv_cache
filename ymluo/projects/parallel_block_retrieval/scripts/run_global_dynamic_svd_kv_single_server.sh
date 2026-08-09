#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON_DIR="${PYTHON_DIR:-/home/fdong/miniconda3/envs/moe/bin}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
CORPUS_DIR="${CORPUS_DIR:-${ROOT}/data/real_longbench_docqa_10m_clean_record64}"
PROFILE_DIR="${PROFILE_DIR:-${ROOT}/outputs/real_longbench_docqa_10m_prerope_qk64_question16_profile}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/global_dynamic_svd_kv_q0_v1}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"
IDLE_UTIL="${IDLE_UTIL:-10}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-true}"
POLL_SECONDS="${POLL_SECONDS:-30}"

while true; do
  mapfile -t FREE_GPUS < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F',' -v mem="$IDLE_MEM_MB" -v util="$IDLE_UTIL" \
        '{gsub(/ /, "", $0); if (($2 + 0) <= mem && ($3 + 0) < util) print $1}'
  )
  if ((${#FREE_GPUS[@]} >= 2)) || [[ "$WAIT_FOR_GPUS" != "true" ]]; then
    break
  fi
  echo "Waiting for at least two idle GPUs; currently found ${#FREE_GPUS[@]}"
  sleep "$POLL_SECONDS"
done
if ((${#FREE_GPUS[@]} < 2)); then
  echo "At least two idle GPUs are required; found ${#FREE_GPUS[@]}" >&2
  exit 1
fi
if ((${#FREE_GPUS[@]} > 8)); then
  FREE_GPUS=("${FREE_GPUS[@]:0:8}")
fi

GPU_LIST="$(IFS=,; echo "${FREE_GPUS[*]}")"
WORLD_SIZE="${#FREE_GPUS[@]}"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

echo "Using idle GPUs: $GPU_LIST"
echo "World size: $WORLD_SIZE"
echo "Output: $OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_DIR/torchrun" \
  --standalone \
  --nproc_per_node="$WORLD_SIZE" \
  src/run_global_dynamic_svd_kv_single.py \
  --model_name_or_path "$MODEL" \
  --corpus_dir "$CORPUS_DIR" \
  --profile_dir "$PROFILE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --query_id 0 \
  --svd_rank "${SVD_RANK:-32}" \
  --candidate_blocks "${CANDIDATE_BLOCKS:-512}" \
  --target_blocks "${TARGET_BLOCKS:-3}" \
  ${ANCHOR_INITIAL_QUERY:+--anchor_initial_query} \
  ${INCLUDE_HOP2_PROBE:+--include_hop2_probe} \
  --coarse_reserve_blocks "${COARSE_RESERVE_BLOCKS:-0}" \
  --retrieval_interval "${RETRIEVAL_INTERVAL:-3}" \
  --dynamic_query_tokens "${DYNAMIC_QUERY_TOKENS:-3}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}"
