#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
RUN=$ROOT/results/20260727_qkbalanced_longbench_official75k_full_8gpu
LOG=$RUN/logs/finalize_promptfix.log
REPAIR_PATTERN='[r]un_sample_calibrated_longbench_20260717.py.*20260727_qkbalanced_longbench_official75k_full_8gpu/shard4'

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

log "waiting for corrected shard4 repair"
while pgrep -f "$REPAIR_PATTERN" >/dev/null; do
  sleep 30
done

log "repair process exited; validating raw shard CSV files"
"$PYTHON" - "$RUN" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("shard[0-9]*/sample_results.csv")):
    with path.open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))

expected = {
    "full_kv",
    "countcap_fullprompt_qkbalanced_packed_direct",
}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 7500, (len(rows), counts)
assert counts == Counter({method: 3750 for method in expected}), counts
assert len(pairs) == 3750, len(pairs)
assert all(methods == expected for methods in pairs.values())
assert len({row["task"] for row in rows}) == 16
assert max(int(float(row["prompt_tokens"])) for row in rows) <= 7500
print("validated raw 3750 strict official-protocol pairs")
PY

log "raw validation passed; generating merged and paired summaries"
"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN/shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN/merged" \
  >"$RUN/logs/summary.log" 2>&1

"$PYTHON" src/analyze_qkbalanced_longbench_paired_20260727.py \
  --input_csv "$RUN/merged/sample_results.csv" \
  --output_dir "$RUN/paired_analysis" \
  >"$RUN/logs/paired_analysis.log" 2>&1

"$PYTHON" - "$RUN/merged/sample_results.csv" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

expected = {
    "full_kv",
    "countcap_fullprompt_qkbalanced_packed_direct",
}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 7500, (len(rows), counts)
assert counts == Counter({method: 3750 for method in expected}), counts
assert len(pairs) == 3750
assert all(methods == expected for methods in pairs.values())
assert len({row["task"] for row in rows}) == 16
print("validated merged 3750 strict official-protocol pairs")
PY

touch "$RUN/ALL_COMPLETE"
log "official LongBench is complete; restoring dependent experiment queue"

mkdir -p results/20260727_qkbalanced_queue_logs
scripts=(
  launch_qkbalanced_factorial_m20_5gpu_20260727.sh
  launch_qk_variable_physical_32k_paired_8gpu_20260727.sh
  launch_qk_variable_physical_128k_4gpu_20260727.sh
  launch_qk_progressive_refinement_after_physical_2gpu_20260727.sh
  launch_qk_matched_rate_after_physical_2gpu_20260727.sh
  launch_qk_norm_certified_after_matched_2gpu_20260727.sh
  launch_qkbalanced_additivity_after_norm_2gpu_20260727.sh
  launch_qkbalanced_factorial_holdout_offset100_5gpu_20260727.sh
  launch_qkbalanced_ruler_4k128k_8gpu_20260727.sh
  launch_qkbalanced_qwen25_longbench_m100_8gpu_20260727.sh
)
for script in "${scripts[@]}"; do
  if pgrep -f "[b]ash scripts/$script" >/dev/null; then
    log "already running: $script"
    continue
  fi
  setsid -f bash "scripts/$script" \
    >"results/20260727_qkbalanced_queue_logs/${script%.sh}.log" \
    2>&1 </dev/null
  log "started: $script"
done

log "finalization and requeue complete"
