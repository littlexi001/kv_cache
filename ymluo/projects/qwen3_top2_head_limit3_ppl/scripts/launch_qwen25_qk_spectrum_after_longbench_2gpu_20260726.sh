#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen2.5-7B-Instruct
PARENT_RUN=$ROOT/results/20260726_countcap_qwen25_7b_longbench_m100_prompt7500
RUN_ROOT=$ROOT/results/20260726_qwen25_7b_qk_matrix_spectrum_32k
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

log "waiting for Qwen2.5 LongBench"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  if [[ -f "$PARENT_RUN/FAILED" ]]; then
    log "parent failed"
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  if ! pgrep -f \
    "launch_countcap_qwen25_7b_longbench_after_final_4gpu_20260726.sh" \
    >/dev/null; then
    log "parent stopped without ALL_COMPLETE"
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  sleep 300
done

PIDS=()
LABELS=()
for item in "0 sports" "1 medicine"; do
  read -r gpu topic <<<"$item"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "$MODEL" \
    --output_path "$TRACE_ROOT/qwen25_7b_${topic}.pt" \
    --topic "$topic" \
    --history_tokens 32000 \
    --steps 64 \
    --layers "0,7,14,21,27" \
    --prefill_chunk_tokens 2048 \
    --omit_values \
    --dtype float16 --device cuda --device_map auto \
    >"$LOG_ROOT/qwen25_7b_${topic}.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$topic")
  log "launched topic=$topic gpu=$gpu pid=$!"
done

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    log "completed ${LABELS[$index]}"
  else
    status=$?
    log "failed ${LABELS[$index]} status=$status"
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  touch "$RUN_ROOT/FAILED"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/analyze_qk_matrix_spectrum_20260726.py \
  --trace "qwen25_7b=sports=$TRACE_ROOT/qwen25_7b_sports.pt" \
  --trace "qwen25_7b=medicine=$TRACE_ROOT/qwen25_7b_medicine.pt" \
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

root = sys.argv[1]
with open(
    root + "/analysis/qk_spectrum_rows.csv",
    encoding="utf-8",
    newline="",
) as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 40, len(rows)
assert {row["model"] for row in rows} == {"qwen25_7b"}
assert all(int(row["query_count"]) == 448 for row in rows)
assert all(float(row["qk_tail_bound_satisfied"]) == 1.0 for row in rows)
with open(root + "/analysis/summary.json", encoding="utf-8") as handle:
    summary = json.load(handle)
assert len(summary["by_model"]) == 1
print("validated 40 Qwen2.5 topic-layer-KV-head spectrum cases")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
