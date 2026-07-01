#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/token_type_spectral_correlations_smoke_0630}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

nohup python -u src/analyze_token_type_spectral_correlations.py \
  --model_name_or_path "${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}" \
  --output_dir "$OUT" \
  --variants "${VARIANTS:-compact_kv,json_kv,needle_sentence,topic_table}" \
  --tasks_per_variant "${TASKS_PER_VARIANT:-4}" \
  --records_per_task "${RECORDS_PER_TASK:-16}" \
  --chunk_size "${CHUNK_SIZE:-256}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --layers "${LAYERS:-0,4,8,13,20,27}" \
  --heads "${HEADS:-all}" \
  --rank_cutoffs "${RANK_CUTOFFS:-1,2,4,8,16,32,64,128}" \
  --direction_count "${DIRECTION_COUNT:-16}" \
  --sink_tokens "${SINK_TOKENS:-16}" \
  --recent_tokens "${RECENT_TOKENS:-64}" \
  --max_query_tokens_per_task "${MAX_QUERY_TOKENS_PER_TASK:-2}" \
  --max_tokens_per_group_per_row "${MAX_TOKENS_PER_GROUP_PER_ROW:-48}" \
  --include_other_sample "${INCLUDE_OTHER_SAMPLE:-true}" \
  --center_k "${CENTER_K:-true}" \
  --svd_device "${SVD_DEVICE:-cuda}" \
  --svd_dtype "${SVD_DTYPE:-float32}" \
  --log_every "${LOG_EVERY:-1}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
