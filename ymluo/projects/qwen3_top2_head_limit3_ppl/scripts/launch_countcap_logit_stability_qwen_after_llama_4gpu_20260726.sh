#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
PARENT_RUN=$ROOT/results/20260726_countcap_logit_stability_32k_4gpu
RUN_ROOT=$ROOT/results/20260726_countcap_logit_stability_qwen3_4b_32k_4gpu
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

log "queued behind $PARENT_RUN"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  if [[ -f "$PARENT_RUN/FAILED" ]]; then
    log "parent failed"
    exit 1
  fi
  if ! pgrep -f "launch_countcap_logit_stability_after_multimodel_4gpu_20260726.sh" >/dev/null; then
    log "parent is no longer running and ALL_COMPLETE is absent"
    exit 1
  fi
  sleep 300
done

PIDS=()
LABELS=()
TOPICS=(sports medicine mixed_a mixed_b)
for gpu in 0 1 2 3; do
  topic="${TOPICS[$gpu]}"
  output_dir="$RUN_ROOT/$topic"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$output_dir" \
    --topics "$topic" \
    --window_indices 0,1,2 \
    --methods full_attention,direct_countcap \
    --history_tokens 32000 \
    --eval_tokens 256 \
    --window_stride_tokens 32512 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 48 \
    --sample_count 256 \
    --prefill_chunk_tokens 2048 \
    --collect_logit_stability \
    --dtype float16 --device cuda --device_map auto \
    >"$LOG_ROOT/${topic}.log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$topic")
  log "launched topic=${topic} gpu=${gpu} pid=$!"
done

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

"$PYTHON" src/summarize_countcap_logit_stability_20260726.py \
  --input_glob "$RUN_ROOT/*/token_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$RUN_ROOT" <<'PY'
import csv
import glob
import sys
from collections import Counter

root = sys.argv[1]
rows = []
for path in glob.glob(root + "/*/token_results.csv"):
    with open(path, encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))

counts = Counter(row["method"] for row in rows)
assert counts == Counter(
    {"full_attention": 3072, "direct_countcap": 3072}
), counts
sparse = [row for row in rows if row["method"] == "direct_countcap"]
assert all(row.get("kl_full_to_sparse", "") != "" for row in sparse)
assert all(
    not (
        int(float(row["margin_certificate_satisfied"])) == 1
        and int(float(row["top1_agreement"])) == 0
    )
    for row in sparse
)
print("validated 3072 paired Qwen token-logit comparisons")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
