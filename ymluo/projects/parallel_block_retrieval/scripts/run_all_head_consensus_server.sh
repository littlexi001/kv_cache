#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DIR}/data/real_longbench_docqa_10m_clean_record64}"
PROFILE_DIR="${PROFILE_DIR:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_allhead_consensus_${STAMP}}"

GPU_IDS="${GPU_IDS:-auto}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"
FORCE_PROFILE="${FORCE_PROFILE:-false}"
RUN_NLL="${RUN_NLL:-true}"
TOP_PER_HEAD="${TOP_PER_HEAD:-16}"
TARGET_BLOCKS="${TARGET_BLOCKS:-39}"

cd "$PROJECT_DIR"
mkdir -p "$PROFILE_DIR" "$OUT_DIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

if [[ "$GPU_IDS" == "auto" ]]; then
  mapfile -t FREE_GPU_ARRAY < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, -v lim="$IDLE_MEM_MB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2 + 0) <= (lim + 0)) print $1}'
  )
else
  IFS=',' read -r -a FREE_GPU_ARRAY <<< "$GPU_IDS"
fi

FREE_COUNT="${#FREE_GPU_ARRAY[@]}"
if [[ "$FREE_COUNT" -lt 1 ]]; then
  echo "[all-head] no idle GPU found" >&2
  exit 1
fi
if [[ "$FREE_COUNT" -gt 8 ]]; then
  FREE_GPU_ARRAY=("${FREE_GPU_ARRAY[@]:0:8}")
  FREE_COUNT=8
fi
DEVICES="$(IFS=,; echo "${FREE_GPU_ARRAY[*]}")"
echo "[all-head] devices=$DEVICES"
echo "[all-head] profile=$PROFILE_DIR"
echo "[all-head] output=$OUT_DIR"

if [[ "$FORCE_PROFILE" == "true" || ! -f "$PROFILE_DIR/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/profile_all_head_qk.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --profile_dir "$PROFILE_DIR" \
      --layers all \
      --svd_rank 32 \
      --calibration_blocks 32 \
      --query_vector_tokens 16 \
      --dtype float16 \
      --attn_implementation sdpa \
      --log_every 10 \
    2>&1 | tee "$OUT_DIR/profile.log"
fi

CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$FREE_COUNT" \
  src/run_all_head_consensus_retrieval.py \
    --corpus_dir "$CORPUS_DIR" \
    --profile_dir "$PROFILE_DIR" \
    --output_dir "$OUT_DIR" \
    --target_blocks "$TARGET_BLOCKS" \
    --top_per_head "$TOP_PER_HEAD" \
    --query_batch 8 \
    --block_chunk 64 \
    --exclude_block_prefix_tokens 16 \
  2>&1 | tee "$OUT_DIR/retrieval.log"

"$PY" src/analyze_all_head_consensus.py \
  --corpus_dir "$CORPUS_DIR" \
  --retrieval_dir "$OUT_DIR" \
  --target_blocks "$TARGET_BLOCKS" \
  2>&1 | tee "$OUT_DIR/analysis.log"

if [[ "$RUN_NLL" == "true" ]]; then
  mkdir -p "$OUT_DIR/answer_nll"
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/evaluate_retrieved_answer_nll.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --retrieval_results "$OUT_DIR/query_results.csv" \
      --output_dir "$OUT_DIR/answer_nll" \
      --methods allhead_layer_consensus,allhead_head_vote,allhead_rrf,selected4_independent_rrf \
      --target_blocks "$TARGET_BLOCKS" \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT_DIR/answer_nll.log"
fi

echo "[all-head] done: $OUT_DIR"
