#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260727_qk_balanced_96k_independent}"
TRACE_ROOT="$RUN_ROOT/traces"
LOG_ROOT="$RUN_ROOT/logs"
ANALYSIS_ROOT="$RUN_ROOT/analysis"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$TRACE_ROOT" "$LOG_ROOT" "$ANALYSIS_ROOT"
cd "$ROOT"

collect_and_analyze() {
  local gpus="$1"
  local topic="$2"
  local trace="$TRACE_ROOT/qwen3_4b_${topic}96k_32steps.pt"
  local output="$ANALYSIS_ROOT/qwen3_${topic}96k"

  if [[ ! -s "$trace" ]]; then
    CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON" -u \
      src/collect_real_qk_trace_20260715.py \
      --model_name_or_path "$MODEL" \
      --output_path "$trace" \
      --topic "$topic" \
      --history_tokens 96000 \
      --steps 32 \
      --layers "0,8,17,26,35" \
      --prefill_chunk_tokens 2048 \
      --omit_values \
      --dtype float16 \
      --device cuda \
      --device_map balanced \
      >"$LOG_ROOT/collect_${topic}.log" 2>&1
  fi

  if [[ ! -s "$output/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON" -u \
      src/analyze_qk_balanced_spectral_rate_20260727.py \
      --trace_path "$trace" \
      --output_dir "$output" \
      --label "qwen3_${topic}96k" \
      --device cuda \
      --sample_stride 32 \
      --calibration_steps 8 \
      --total_rate_budget 15 \
      --query_shrinkage 0.5 \
      --selected_fractions 0.01,0.02,0.06 \
      --top_fraction 0.01 \
      >"$LOG_ROOT/analyze_${topic}.log" 2>&1
  fi
}

collect_and_analyze "0,1" sports &
pid_sports=$!
collect_and_analyze "2,3" medicine &
pid_medicine=$!

wait "$pid_sports"
wait "$pid_medicine"
touch "$RUN_ROOT/ALL_COMPLETE"
echo "ALL_COMPLETE"
