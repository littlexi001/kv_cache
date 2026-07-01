#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT="${OUT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/sink_ablation_diagnostics_smoke_0630}"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

nohup python -u src/run_sink_ablation_diagnostics.py \
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
  --sink_tokens "${SINK_TOKENS:-16}" \
  --keep_prefix_tokens "${KEEP_PREFIX_TOKENS:-2}" \
  --recent_tokens "${RECENT_TOKENS:-16}" \
  --replacement_text "${REPLACEMENT_TEXT:- X}" \
  --conditions "${CONDITIONS:-baseline,replace_sink_content,move_sink_middle,move_sink_end,keep_prefix2_text,zero_sink_kv,drop_sink_kv,keep_prefix2_drop_sink_kv}" \
  --collect_attention "${COLLECT_ATTENTION:-true}" \
  --max_eval_query_tokens "${MAX_EVAL_QUERY_TOKENS:-0}" \
  --log_every "${LOG_EVERY:-1}" \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
