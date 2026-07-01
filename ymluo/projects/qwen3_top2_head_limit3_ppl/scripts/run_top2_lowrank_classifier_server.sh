#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_lowrank_classifier_0630}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

nohup python -u src/train_top2_lowrank_classifier.py \
  --model_name_or_path "${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}" \
  --output_dir "$OUT" \
  --variants "${VARIANTS:-compact_kv,json_kv,needle_sentence,topic_table}" \
  --train_tasks_per_variant "${TRAIN_TASKS_PER_VARIANT:-2}" \
  --eval_tasks_per_variant "${EVAL_TASKS_PER_VARIANT:-1}" \
  --records_per_task "${RECORDS_PER_TASK:-16}" \
  --chunk_size "${CHUNK_SIZE:-256}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --layers "${LAYERS:-0,4,8,13,20,27}" \
  --heads "${HEADS:-0,4,8,12}" \
  --ranks "${RANKS:-4,8,16,32,64}" \
  --max_query_tokens_per_task "${MAX_QUERY_TOKENS_PER_TASK:-2}" \
  --svd_device "${SVD_DEVICE:-cuda}" \
  --svd_dtype "${SVD_DTYPE:-float32}" \
  --negative_per_positive "${NEGATIVE_PER_POSITIVE:-8}" \
  --train_epochs "${TRAIN_EPOCHS:-80}" \
  --learning_rate "${LEARNING_RATE:-0.05}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --standardize_features "${STANDARDIZE_FEATURES:-true}" \
  --save_model_weights "${SAVE_MODEL_WEIGHTS:-false}" \
  --log_every "${LOG_EVERY:-1}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
