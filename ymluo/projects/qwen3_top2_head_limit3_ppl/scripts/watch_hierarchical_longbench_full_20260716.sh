#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PREFIX=20260716_hierarchical_longbench_full_v1_shard
OUTPUT_DIR="$ROOT/outputs/20260716_hierarchical_longbench_full_v1_merged"
LOG="$ROOT/outputs/logs/20260716_hierarchical_longbench_full_v1_watch.log"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG")"

while true; do
  completed=0
  for shard in $(seq 0 7); do
    if [[ -s "outputs/${PREFIX}${shard}/summary.json" ]]; then
      completed=$((completed + 1))
    fi
  done
  printf '%s completed_shards=%d/8\n' "$(date --iso-8601=seconds)" "$completed" >> "$LOG"
  if [[ "$completed" -eq 8 ]]; then
    break
  fi
  if ! pgrep -f 'run_hierarchical_longbench_probe_20260715.py.*20260716_hierarchical_longbench_full_v1_shard' >/dev/null; then
    printf '%s workers exited before all shards completed\n' "$(date --iso-8601=seconds)" >> "$LOG"
    exit 1
  fi
  sleep 60
done

/home/fdong/miniconda3/envs/moe/bin/python \
  src/summarize_hierarchical_longbench_shards_20260716.py \
  --input_glob 'outputs/20260716_hierarchical_longbench_full_v1_shard*/sample_results.csv' \
  --output_dir "$OUTPUT_DIR" \
  --expected_tasks 16 \
  --expected_samples_per_method 3750 \
  >> "$LOG" 2>&1

printf '%s aggregation complete\n' "$(date --iso-8601=seconds)" >> "$LOG"
