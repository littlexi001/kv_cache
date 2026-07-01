#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/calibrated_lrsvd_candidate_recall_0630}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

nohup python -u src/analyze_calibrated_lrsvd_candidate_recall.py \
  --model_name_or_path "${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}" \
  --output_dir "$OUT" \
  --variants "${VARIANTS:-compact_kv,json_kv,needle_sentence,topic_table}" \
  --tasks_per_variant "${TASKS_PER_VARIANT:-8}" \
  --records_per_task "${RECORDS_PER_TASK:-16}" \
  --chunk_size "${CHUNK_SIZE:-256}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --candidate_fractions "${CANDIDATE_FRACTIONS:-0.02,0.05,0.08}" \
  --calib_samples "${CALIB_SAMPLES:-128,256,512}" \
  --ranks "${RANKS:-16,32,64}" \
  --layers "${LAYERS:-0,4,8,13,20,27}" \
  --heads "${HEADS:-0,4,8,12}" \
  --max_query_tokens_per_task "${MAX_QUERY_TOKENS_PER_TASK:-2}" \
  --svd_device "${SVD_DEVICE:-cuda}" \
  --svd_dtype "${SVD_DTYPE:-float32}" \
  --center_k "${CENTER_K:-true}" \
  --write_per_query "${WRITE_PER_QUERY:-false}" \
  --log_every "${LOG_EVERY:-2}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
