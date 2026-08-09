#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
LLAMA_MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
QWEN_MODEL=/home/fdong/models/Qwen3-4B-Instruct
PARENT_RUN=$ROOT/results/20260726_pca48_int4_delta_invariance_32k_v4
RUN_ROOT=$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k
TRACE_ROOT=$RUN_ROOT/traces
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$TRACE_ROOT" "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

log "queued behind $PARENT_RUN"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  if ! pgrep -f "launch_countcap_crossing_mass_after_logit_20260726.sh" >/dev/null; then
    log "parent is no longer running and ALL_COMPLETE is absent"
    exit 1
  fi
  sleep 300
done

PIDS=()
LABELS=()

launch_trace() {
  local gpu="$1"
  local label="$2"
  local model="$3"
  local topic="$4"
  local layers="$5"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "$model" \
    --output_path "$TRACE_ROOT/${label}_${topic}.pt" \
    --topic "$topic" \
    --history_tokens 32000 \
    --steps 64 \
    --layers "$layers" \
    --prefill_chunk_tokens 2048 \
    --omit_values \
    --dtype float16 --device cuda --device_map auto \
    >"$LOG_ROOT/${label}_${topic}.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("${label}_${topic}")
  log "launched ${label}_${topic} gpu=${gpu} pid=$!"
}

launch_trace 0 llama31_8b "$LLAMA_MODEL" sports "0,8,16,24,31"
launch_trace 1 llama31_8b "$LLAMA_MODEL" medicine "0,8,16,24,31"
launch_trace 2 qwen3_4b "$QWEN_MODEL" sports "0,8,17,26,35"
launch_trace 3 qwen3_4b "$QWEN_MODEL" medicine "0,8,17,26,35"

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    log "completed ${LABELS[$index]}"
  else
    status=$?
    log "failed ${LABELS[$index]} status=${status}"
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  touch "$RUN_ROOT/FAILED"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/analyze_qk_matrix_spectrum_20260726.py \
  --trace "llama31_8b=sports=$TRACE_ROOT/llama31_8b_sports.pt" \
  --trace "llama31_8b=medicine=$TRACE_ROOT/llama31_8b_medicine.pt" \
  --trace "qwen3_4b=sports=$TRACE_ROOT/qwen3_4b_sports.pt" \
  --trace "qwen3_4b=medicine=$TRACE_ROOT/qwen3_4b_medicine.pt" \
  --output_dir "$RUN_ROOT/analysis" \
  --rank 48 \
  --basis_prefix_tokens 2048 \
  --basis_sample_stride 32 \
  --device cuda \
  >"$LOG_ROOT/analysis.log" 2>&1

"$PYTHON" - "$RUN_ROOT" <<'PY'
import csv
import json
import sys
from collections import Counter

root = sys.argv[1]
with open(
    root + "/analysis/qk_spectrum_rows.csv",
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 160, len(rows)
assert Counter(row["model"] for row in rows) == {
    "llama31_8b": 80,
    "qwen3_4b": 80,
}
assert all(int(row["query_count"]) == 256 for row in rows)
assert all(float(row["qk_tail_bound_satisfied"]) == 1.0 for row in rows)
with open(root + "/analysis/summary.json", encoding="utf-8") as handle:
    summary = json.load(handle)
assert len(summary["by_model"]) == 2
print("validated 160 model-topic-layer-head QK spectrum cases")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
