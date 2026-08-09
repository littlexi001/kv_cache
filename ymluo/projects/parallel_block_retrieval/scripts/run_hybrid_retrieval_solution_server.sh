#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LONGBENCH_DIR="${LONGBENCH_DIR:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DIR}/data/real_longbench_docqa_10m_clean_record64}"
LEXICAL_DIR="${LEXICAL_DIR:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_clean_bm25}"
PROFILE_DIR="${PROFILE_DIR:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_prerope_qk64_question16_profile}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/real_longbench_docqa_10m_hybrid_solution_${STAMP}}"
ROUTING_DIR="${ROUTING_DIR:-${OUT_ROOT}/record_question_nll}"
RETRIEVAL_DIR="${RETRIEVAL_DIR:-${OUT_ROOT}/retrieval}"
NLL_DIR="${NLL_DIR:-${OUT_ROOT}/answer_nll}"

GPU_IDS="${GPU_IDS:-auto}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_LEXICAL="${FORCE_LEXICAL:-false}"
FORCE_PROFILE="${FORCE_PROFILE:-false}"
FORCE_ROUTING="${FORCE_ROUTING:-false}"
RUN_NLL="${RUN_NLL:-true}"

DATASETS="${DATASETS:-hotpotqa,2wikimqa,musique,qasper,narrativeqa,multifieldqa_en,qmsum,gov_report}"
QUERY_DATASETS="${QUERY_DATASETS:-hotpotqa,2wikimqa,musique,qasper,narrativeqa,multifieldqa_en}"
SEQ_TOKENS="${SEQ_TOKENS:-10000000}"
BLOCK_TOKENS="${BLOCK_TOKENS:-256}"
TARGET_BLOCKS="${TARGET_BLOCKS:-39}"
NUM_QUERIES="${NUM_QUERIES:-64}"
MAX_BLOCKS_PER_RECORD="${MAX_BLOCKS_PER_RECORD:-64}"

cd "$PROJECT_DIR"
mkdir -p "$CORPUS_DIR" "$LEXICAL_DIR" "$PROFILE_DIR" "$OUT_ROOT" "$ROUTING_DIR" "$RETRIEVAL_DIR"
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
  echo "[hybrid] no idle GPU found" >&2
  exit 1
fi
if [[ "$FREE_COUNT" -gt 8 ]]; then
  FREE_GPU_ARRAY=("${FREE_GPU_ARRAY[@]:0:8}")
  FREE_COUNT=8
fi
DEVICES="$(IFS=,; echo "${FREE_GPU_ARRAY[*]}")"
echo "[hybrid] devices=$DEVICES"
echo "[hybrid] output=$OUT_ROOT"

if [[ "$FORCE_PREPARE" == "true" || ! -f "$CORPUS_DIR/summary.json" ]]; then
  "$PY" src/prepare_real_longbench_corpus.py \
    --model_name_or_path "$MODEL" \
    --longbench_dir "$LONGBENCH_DIR" \
    --datasets "$DATASETS" \
    --query_datasets "$QUERY_DATASETS" \
    --output_dir "$CORPUS_DIR" \
    --seq_tokens "$SEQ_TOKENS" \
    --block_tokens "$BLOCK_TOKENS" \
    --num_queries "$NUM_QUERIES" \
    --max_blocks_per_record "$MAX_BLOCKS_PER_RECORD"
fi

if [[ "$FORCE_LEXICAL" == "true" || ! -f "$LEXICAL_DIR/summary.json" ]]; then
  "$PY" src/run_lexical_block_retrieval.py \
    --model_name_or_path "$MODEL" \
    --corpus_dir "$CORPUS_DIR" \
    --output_dir "$LEXICAL_DIR" \
    --target_blocks "$TARGET_BLOCKS" \
    --record_allocations 20,30,39 \
    2>&1 | tee "$OUT_ROOT/lexical.log"
fi

if [[ "$FORCE_PROFILE" == "true" || ! -f "$PROFILE_DIR/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/profile_real_qk.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --profile_dir "$PROFILE_DIR" \
      --pairs 3:10,21:8,6:7,16:14 \
      --svd_rank 64 \
      --calibration_blocks 32 \
      --batch_blocks 8 \
      --query_vector_tokens 16 \
      --query_vector_mode question_content \
      --profile_space pre_rope_record_qk \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT_ROOT/profile.log"
fi

if [[ "$FORCE_ROUTING" == "true" || ! -f "$ROUTING_DIR/routing.csv" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/rerank_records_by_question_nll.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --lexical_dir "$LEXICAL_DIR" \
      --output_dir "$ROUTING_DIR" \
      --top_records 5 \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT_ROOT/record_question_nll.log"
fi

CUDA_VISIBLE_DEVICES="${FREE_GPU_ARRAY[0]}" "$PY" src/run_hybrid_block_retrieval.py \
  --corpus_dir "$CORPUS_DIR" \
  --profile_dir "$PROFILE_DIR" \
  --lexical_dir "$LEXICAL_DIR" \
  --output_dir "$RETRIEVAL_DIR" \
  --target_blocks "$TARGET_BLOCKS" \
  --record_allocations 20,30,39 \
  --semantic_rank 32 \
  --global_candidates 782 \
  --top_record_candidates 5 \
  --record_margin_threshold 0.04 \
  --record_routing_csv "$ROUTING_DIR/routing.csv" \
  2>&1 | tee "$OUT_ROOT/hybrid_retrieval.log"

if [[ "$RUN_NLL" == "true" ]]; then
  mkdir -p "$NLL_DIR"
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/evaluate_retrieved_answer_nll.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --retrieval_results "$RETRIEVAL_DIR/query_results.csv" \
      --output_dir "$NLL_DIR" \
      --methods deep_ql_record39_svd32 \
      --target_blocks "$TARGET_BLOCKS" \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT_ROOT/answer_nll.log"
fi

echo "[hybrid] done: $OUT_ROOT"
