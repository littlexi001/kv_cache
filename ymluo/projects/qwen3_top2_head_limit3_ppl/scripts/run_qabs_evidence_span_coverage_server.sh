#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/evidence_span_coverage_qabs5_shortctx_v2
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python -u src/run_qabs_evidence_span_coverage.py \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --output_dir "$OUT" \
  --variants compact_kv,json_kv,topic_table,needle_sentence \
  --tasks_per_variant 16 \
  --records_per_task 16 \
  --chunk_size 256 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  --top_fraction 0.05 \
  --protect_sink_tokens 10 \
  --protect_recent_tokens 10 \
  --log_every 8 \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
