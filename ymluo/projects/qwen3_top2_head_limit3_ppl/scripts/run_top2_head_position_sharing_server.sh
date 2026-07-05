#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_war_4k_v1}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

nohup python -u src/analyze_top2_head_position_sharing.py \
  --model_name_or_path "${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}" \
  --text_path "${TEXT_PATH:-data/war_and_peace_pg2600.txt}" \
  --output_dir "$OUT" \
  --total_tokens "${TOTAL_TOKENS:-4160}" \
  --prefill_tokens "${PREFILL_TOKENS:-4096}" \
  --eval_tokens "${EVAL_TOKENS:-64}" \
  --chunk_size "${CHUNK_SIZE:-64}" \
  --dtype "${DTYPE:-float16}" \
  --device cuda \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation eager \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --remote_only "${REMOTE_ONLY:-false}" \
  --exclude_sink_tokens "${EXCLUDE_SINK_TOKENS:-64}" \
  --exclude_recent_tokens "${EXCLUDE_RECENT_TOKENS:-512}" \
  --max_query_samples "${MAX_QUERY_SAMPLES:-64}" \
  --query_stride "${QUERY_STRIDE:-0}" \
  --group_recall_thresholds "${GROUP_RECALL_THRESHOLDS:-0.50,0.60,0.70,0.80,0.90}" \
  --stability_thresholds "${STABILITY_THRESHOLDS:-0.50,0.70,0.80,0.90}" \
  --include_token_text "${INCLUDE_TOKEN_TEXT:-true}" \
  --write_query_layer_group_metrics "${WRITE_QUERY_LAYER_GROUP_METRICS:-false}" \
  --write_query_layer_pair_metrics "${WRITE_QUERY_LAYER_PAIR_METRICS:-false}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
