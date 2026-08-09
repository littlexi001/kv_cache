#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PARENT_RUN=$ROOT/results/20260726_countcap_logit_stability_qwen3_4b_32k_4gpu
TRACE_ROOT=$ROOT/results/20260717_real_qkv_traces_32k
RUN_ROOT=$ROOT/results/20260726_pca48_int4_delta_invariance_32k_v4
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  if ! pgrep -f "launch_countcap_logit_stability_qwen_after_llama_4gpu_20260726.sh" >/dev/null; then
    log "parent is no longer running and ALL_COMPLETE is absent"
    exit 1
  fi
  sleep 300
done

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/analyze_pca_int4_delta_invariance_20260726.py \
  --trace "sports=$TRACE_ROOT/sports.pt" \
  --trace "medicine=$TRACE_ROOT/medicine.pt" \
  --output_dir "$RUN_ROOT" \
  --rank 48 \
  --sample_stride 32 \
  --fractions 0.02,0.04,0.06,0.08 \
  --device cuda \
  >"$LOG_ROOT/analysis.log" 2>&1

"$PYTHON" src/plot_pca_int4_delta_invariance_20260726.py \
  --summary_path "$RUN_ROOT/summary.json" \
  --output_path "$RUN_ROOT/error_chain.png" \
  >"$LOG_ROOT/plot.log" 2>&1

"$PYTHON" - "$RUN_ROOT" <<'PY'
import csv
import sys
from collections import Counter

root = sys.argv[1]
with open(root + "/candidate_rows.csv", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 8960, len(rows)
counts = Counter(row["method"] for row in rows)
threshold_method = (
    "production_pca48_int4k_int8q_sampled_quantile_uncapped"
)
assert counts[threshold_method] == 1280, counts
topk_rows = [row for row in rows if row["method"] != threshold_method]
assert len(topk_rows) == 7680
assert all(
    float(row["deterministic_mass_bound_satisfied"]) == 1.0
    for row in topk_rows
)
assert all(
    float(row["output_bound_satisfied"]) == 1.0
    for row in topk_rows
)
threshold_rows = [row for row in rows if row["method"] == threshold_method]
assert all(float(row["sample_count"]) == 256.0 for row in threshold_rows)
assert all(float(row["sampled_selected_count"]) > 0.0 for row in threshold_rows)
print("validated 7680 top-k rows and 1280 sampled-threshold rows")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
