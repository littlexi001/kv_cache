#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/outputs/parallel_10m_${STAMP}}"
GPU_LIST="${GPU_LIST:-auto}"
GPU_IDS="${GPU_IDS:-auto}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1024}"

SEQ_TOKENS="${SEQ_TOKENS:-10000000}"
BLOCK_TOKENS="${BLOCK_TOKENS:-256}"
TARGET_TOKENS="${TARGET_TOKENS:-10000}"
NUM_QUERIES="${NUM_QUERIES:-512}"
DENSE_DIM="${DENSE_DIM:-128}"
SVD_RANK="${SVD_RANK:-64}"
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"
SEED="${SEED:-20260710}"

cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT" outputs/logs

echo "[parallel-retrieval] out_root=$OUT_ROOT"

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
  echo "[parallel-retrieval] no idle GPU found. Set GPU_IDS=0,1,... to override." >&2
  exit 1
fi

if [[ "$GPU_LIST" == "auto" ]]; then
  GPU_COUNTS=()
  for c in 1 2 4 8; do
    if [[ "$c" -le "$FREE_COUNT" ]]; then
      GPU_COUNTS+=("$c")
    fi
  done
  if [[ "${GPU_COUNTS[-1]}" != "$FREE_COUNT" ]]; then
    GPU_COUNTS+=("$FREE_COUNT")
  fi
else
  read -r -a GPU_COUNTS <<< "$GPU_LIST"
fi

echo "[parallel-retrieval] idle_gpus=$(IFS=,; echo "${FREE_GPU_ARRAY[*]}")"
echo "[parallel-retrieval] gpu_counts=${GPU_COUNTS[*]}"

for NPROC in "${GPU_COUNTS[@]}"; do
  if [[ "$NPROC" -gt "$FREE_COUNT" ]]; then
    echo "[parallel-retrieval] skip nproc=${NPROC}; only ${FREE_COUNT} idle GPUs"
    continue
  fi
  DEVICES="$(IFS=,; echo "${FREE_GPU_ARRAY[*]:0:NPROC}")"
  OUT_DIR="${OUT_ROOT}/gpus${NPROC}"
  LOG="${OUT_ROOT}/gpus${NPROC}.log"
  mkdir -p "$OUT_DIR"
  echo "[parallel-retrieval] running nproc=${NPROC} devices=${DEVICES}"
  CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NPROC" \
    src/run_parallel_block_retrieval.py \
      --seq_tokens "$SEQ_TOKENS" \
      --block_tokens "$BLOCK_TOKENS" \
      --target_tokens "$TARGET_TOKENS" \
      --num_queries "$NUM_QUERIES" \
      --dense_dim "$DENSE_DIM" \
      --svd_rank "$SVD_RANK" \
      --repeats "$REPEATS" \
      --warmup "$WARMUP" \
      --seed "$SEED" \
      --out_dir "$OUT_DIR" \
    2>&1 | tee "$LOG"
done

"$PY" src/summarize_scaling.py \
  --root "$OUT_ROOT" \
  --out_csv "$OUT_ROOT/scaling_summary.csv" \
  --out_md "$OUT_ROOT/scaling_summary.md"

echo "[parallel-retrieval] done"
echo "[parallel-retrieval] summary: $OUT_ROOT/scaling_summary.md"
