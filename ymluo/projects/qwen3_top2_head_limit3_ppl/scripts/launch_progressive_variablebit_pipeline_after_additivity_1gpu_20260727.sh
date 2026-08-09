#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
ADDITIVITY=$ROOT/results/20260727_qkbalanced_additivity_closure_32k
RUN_ROOT=$ROOT/results/20260727_progressive_variablebit_pipeline_3090
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$ADDITIVITY/ALL_COMPLETE" ]]; do
  if ! pgrep -f \
    '^bash scripts/launch_qkbalanced_additivity_after_norm_2gpu_20260727.sh$' \
    >/dev/null; then
    echo "additivity prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

CUDA_VISIBLE_DEVICES=7 "$PYTHON" -u \
  src/benchmark_progressive_variablebit_pipeline_20260727.py \
  --output "$RUN_ROOT/progressive_pipeline.json" \
  --lengths 32768,65536,131072 \
  --candidate_ratios 0.38,0.51 \
  --warmup 5 \
  --repeats 20 \
  >"$LOG_ROOT/benchmark.log" 2>&1

"$PYTHON" - "$RUN_ROOT/progressive_pipeline.json" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
rows = payload["results"]
assert [row["history_count"] for row in rows] == [32768, 65536, 131072]
for row in rows:
    assert row["score_max_abs_error"] <= 5.0e-4, row
    assert row["full_pipeline_ms"] > 0.0, row
    for replay in row["candidate_replays"]:
        assert replay["progressive_pipeline_ms"] > 0.0, replay
        assert math.isfinite(replay["pipeline_speedup_vs_full"]), replay
print("validated progressive packed-index CUDA benchmark")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
