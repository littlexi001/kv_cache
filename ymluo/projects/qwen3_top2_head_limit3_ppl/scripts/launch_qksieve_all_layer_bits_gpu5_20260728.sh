#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_qksieve_all_layer_bits_qwen3_32k}"
TRACE_ROOT="$RUN_ROOT/traces"
ANALYSIS_ROOT="$RUN_ROOT/analysis"
LOG_ROOT="$RUN_ROOT/logs"
LAYERS="$(seq -s, 0 35)"

export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TRACE_ROOT" "$ANALYSIS_ROOT" "$LOG_ROOT"
cd "$ROOT"

for topic in sports medicine; do
  trace="$TRACE_ROOT/qwen3_4b_${topic}32k_all_layers.pt"
  analysis="$ANALYSIS_ROOT/${topic}"
  if [[ ! -s "$trace" ]]; then
    "$PYTHON" -u src/collect_real_qk_trace_20260715.py \
      --model_name_or_path "$MODEL" \
      --output_path "$trace" \
      --topic "$topic" \
      --history_tokens 32000 \
      --steps 16 \
      --layers "$LAYERS" \
      --prefill_chunk_tokens 2048 \
      --omit_values \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      >"$LOG_ROOT/collect_${topic}.log" 2>&1
  fi
  if [[ ! -s "$analysis/summary.json" ]]; then
    "$PYTHON" -u src/analyze_qk_balanced_spectral_rate_20260727.py \
      --trace_path "$trace" \
      --output_dir "$analysis" \
      --label "qwen3_${topic}32k_all_layers" \
      --device cuda \
      --sample_stride 32 \
      --calibration_steps 8 \
      --total_rate_budget 15 \
      --query_shrinkage 0.75 \
      --selected_fractions 0.01,0.02,0.06 \
      --top_fraction 0.01 \
      >"$LOG_ROOT/analyze_${topic}.log" 2>&1
  fi
done

"$PYTHON" src/summarize_qksieve_head_bits_20260728.py \
  --inputs \
  "$ANALYSIS_ROOT/sports/allocations.csv" \
  "$ANALYSIS_ROOT/medicine/allocations.csv" \
  --output-dir "$RUN_ROOT/bit_summary" \
  --method qk_balanced \
  >"$LOG_ROOT/summarize.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
echo "ALL_COMPLETE"
