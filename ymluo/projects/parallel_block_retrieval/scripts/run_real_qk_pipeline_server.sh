#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LONGBENCH_DIR="${LONGBENCH_DIR:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
CORPUS_DIR="${CORPUS_DIR:-${PROJECT_DIR}/data/real_longbench_10m_record64}"
PROFILE_DIR="${PROFILE_DIR:-${PROJECT_DIR}/outputs/real_longbench_10m_postrope_qk64_profile}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/real_longbench_10m_postrope_scaling_${STAMP}}"
GPU_IDS="${GPU_IDS:-auto}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_PROFILE="${FORCE_PROFILE:-false}"
RUN_NLL="${RUN_NLL:-false}"

SEQ_TOKENS="${SEQ_TOKENS:-10000000}"
BLOCK_TOKENS="${BLOCK_TOKENS:-256}"
TARGET_TOKENS="${TARGET_TOKENS:-10000}"
NUM_QUERIES="${NUM_QUERIES:-64}"
DATASETS="${DATASETS:-hotpotqa,2wikimqa,musique,qasper,narrativeqa,triviaqa}"
QUERY_DATASETS="${QUERY_DATASETS:-$DATASETS}"
PAIRS="${PAIRS:-3:10,21:8,6:7,16:14}"
SVD_RANK="${SVD_RANK:-64}"
PROFILE_SPACE="${PROFILE_SPACE:-post_rope_record_qk}"
MAX_BLOCKS_PER_RECORD="${MAX_BLOCKS_PER_RECORD:-64}"
CALIBRATION_BLOCKS="${CALIBRATION_BLOCKS:-32}"
PROFILE_BATCH_BLOCKS="${PROFILE_BATCH_BLOCKS:-8}"
QUERY_VECTOR_TOKENS="${QUERY_VECTOR_TOKENS:-1}"
QUERY_VECTOR_MODE="${QUERY_VECTOR_MODE:-prompt_tail}"
CANDIDATE_FRACTION="${CANDIDATE_FRACTION:-0.02}"
QABS_DIMS="${QABS_DIMS:-8}"
METHODS="${METHODS:-full128,svd32,svd64,svd32_rerank,qabs8}"
NLL_METHODS="${NLL_METHODS:-full128,svd32,svd32_rerank}"
QUERY_BATCH="${QUERY_BATCH:-16}"
BLOCK_CHUNK="${BLOCK_CHUNK:-256}"
EXCLUDE_BLOCK_PREFIX_TOKENS="${EXCLUDE_BLOCK_PREFIX_TOKENS:-16}"
WARMUP="${WARMUP:-1}"
REPEATS="${REPEATS:-3}"

cd "$PROJECT_DIR"
mkdir -p "$CORPUS_DIR" "$PROFILE_DIR" "$OUT_ROOT"
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
  echo "[real-qk] no idle GPU found" >&2
  exit 1
fi
if [[ "$FREE_COUNT" -gt 8 ]]; then
  FREE_GPU_ARRAY=("${FREE_GPU_ARRAY[@]:0:8}")
  FREE_COUNT=8
fi
DEVICES="$(IFS=,; echo "${FREE_GPU_ARRAY[*]}")"
echo "[real-qk] devices=$DEVICES"
echo "[real-qk] corpus=$CORPUS_DIR"
echo "[real-qk] profile=$PROFILE_DIR"
echo "[real-qk] output=$OUT_ROOT"

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

if [[ "$FORCE_PROFILE" == "true" || ! -f "$PROFILE_DIR/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/profile_real_qk.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --profile_dir "$PROFILE_DIR" \
      --pairs "$PAIRS" \
      --svd_rank "$SVD_RANK" \
      --calibration_blocks "$CALIBRATION_BLOCKS" \
      --batch_blocks "$PROFILE_BATCH_BLOCKS" \
      --query_vector_tokens "$QUERY_VECTOR_TOKENS" \
      --query_vector_mode "$QUERY_VECTOR_MODE" \
      --profile_space "$PROFILE_SPACE" \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT_ROOT/profile.log"
fi

GPU_COUNTS=()
for count in 1 2 4 8; do
  if [[ "$count" -le "$FREE_COUNT" && $((FREE_COUNT % count)) -eq 0 ]]; then
    GPU_COUNTS+=("$count")
  fi
done
if [[ "${GPU_COUNTS[-1]}" != "$FREE_COUNT" ]]; then
  GPU_COUNTS+=("$FREE_COUNT")
fi
echo "[real-qk] scaling gpu counts: ${GPU_COUNTS[*]}"

for NPROC in "${GPU_COUNTS[@]}"; do
  RUN_DEVICES="$(IFS=,; echo "${FREE_GPU_ARRAY[*]:0:NPROC}")"
  OUT_DIR="$OUT_ROOT/gpus${NPROC}"
  mkdir -p "$OUT_DIR"
  echo "[real-qk] retrieval nproc=$NPROC devices=$RUN_DEVICES"
  CUDA_VISIBLE_DEVICES="$RUN_DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NPROC" \
    src/run_real_qk_retrieval.py \
      --corpus_dir "$CORPUS_DIR" \
      --profile_dir "$PROFILE_DIR" \
      --out_dir "$OUT_DIR" \
      --target_tokens "$TARGET_TOKENS" \
      --candidate_fraction "$CANDIDATE_FRACTION" \
      --qabs_dims "$QABS_DIMS" \
      --methods "$METHODS" \
      --query_batch "$QUERY_BATCH" \
      --block_chunk "$BLOCK_CHUNK" \
      --exclude_block_prefix_tokens "$EXCLUDE_BLOCK_PREFIX_TOKENS" \
      --warmup "$WARMUP" \
      --repeats "$REPEATS" \
    2>&1 | tee "$OUT_ROOT/gpus${NPROC}.log"
done

"$PY" src/summarize_real_scaling.py \
  --root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/scaling_summary.csv" \
  --out_md "$OUT_ROOT/scaling_summary.md"

if [[ "$RUN_NLL" == "true" ]]; then
  NLL_OUT="${NLL_OUT:-${OUT_ROOT}/answer_nll}"
  mkdir -p "$NLL_OUT"
  echo "[real-qk] answer NLL nproc=$FREE_COUNT devices=$DEVICES"
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$FREE_COUNT" \
    src/evaluate_retrieved_answer_nll.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS_DIR" \
      --retrieval_results "$OUT_ROOT/gpus${FREE_COUNT}/query_results.csv" \
      --output_dir "$NLL_OUT" \
      --methods "$NLL_METHODS" \
      --target_blocks "$((TARGET_TOKENS / BLOCK_TOKENS))" \
      --dtype float16 \
      --attn_implementation sdpa \
    2>&1 | tee "$OUT_ROOT/answer_nll.log"
fi

echo "[real-qk] done: $OUT_ROOT/scaling_summary.md"
