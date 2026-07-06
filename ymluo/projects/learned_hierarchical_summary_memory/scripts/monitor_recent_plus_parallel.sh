#!/usr/bin/env bash
set -euo pipefail

BASE="/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706"
for name in longbench_exact longbench_summary ruler_4k8k ruler_16k; do
  pid="$(cat "$BASE/pids/$name.pid")"
  status="done"
  if kill -0 "$pid" 2>/dev/null; then
    status="running"
  fi
  cases="$(grep -c '^finished case' "$BASE/logs/$name.log" 2>/dev/null || true)"
  last="$(grep '^finished case' "$BASE/logs/$name.log" 2>/dev/null | tail -1 || true)"
  echo "$name pid=$pid status=$status cases=$cases $last"
done
echo "gpu"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | egrep '^(4|5|6|7),'
