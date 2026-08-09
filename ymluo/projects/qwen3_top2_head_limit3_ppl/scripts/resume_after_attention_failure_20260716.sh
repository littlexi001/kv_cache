#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
LOG="$ROOT/outputs/logs/20260716_after_attention_resume.log"
LOCK="$ROOT/outputs/20260716_after_attention_resume.lock"

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p outputs/logs
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "resume lock already exists: $LOCK" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

log_status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" >> "$LOG"
}

wait_for_shards() {
  local prefix=$1
  local count=$2
  local process_pattern=$3
  while true; do
    local completed=0
    local shard
    for shard in $(seq 0 $((count - 1))); do
      if [[ -s "outputs/${prefix}${shard}/summary.json" ]]; then
        completed=$((completed + 1))
      fi
    done
    log_status "prefix=$prefix completed=$completed/$count"
    if [[ "$completed" -eq "$count" ]]; then
      return 0
    fi
    if ! pgrep -f "$process_pattern" >/dev/null; then
      log_status "ERROR prefix=$prefix workers_exited_before_completion"
      return 1
    fi
    sleep 300
  done
}

log_status "restarting corrected 128K attention-path benchmark"
bash scripts/launch_128k_attention_bottleneck_20260716.sh >> "$LOG" 2>&1

log_status "starting RULER 4K-32K stage"
bash scripts/launch_hierarchical_ruler_stage1_20260716.sh >> "$LOG" 2>&1
wait_for_shards \
  20260716_hierarchical_ruler_4k32k_m10_shard 8 \
  'run_hierarchical_ruler_probe_20260716.py.*20260716_hierarchical_ruler_4k32k_m10_shard'
"$PYTHON" src/summarize_hierarchical_ruler_shards_20260716.py \
  --input_glob 'outputs/20260716_hierarchical_ruler_4k32k_m10_shard*/sample_results.csv' \
  --output_dir outputs/20260716_hierarchical_ruler_4k32k_m10_merged \
  --expected_task_lengths 36 \
  --expected_samples_per_method 360 \
  --auto_gate_requested_length 16384 \
  >> "$LOG" 2>&1

log_status "starting RULER 64K-128K stage"
bash scripts/launch_hierarchical_ruler_stage2_20260716.sh >> "$LOG" 2>&1
wait_for_shards \
  20260716_hierarchical_ruler_64k128k_m5_shard 2 \
  'run_hierarchical_ruler_probe_20260716.py.*20260716_hierarchical_ruler_64k128k_m5_shard'
"$PYTHON" src/summarize_hierarchical_ruler_shards_20260716.py \
  --input_glob 'outputs/20260716_hierarchical_ruler_64k128k_m5_shard*/sample_results.csv' \
  --output_dir outputs/20260716_hierarchical_ruler_64k128k_m5_merged \
  --expected_task_lengths 18 \
  --expected_samples_per_method 90 \
  --auto_gate_requested_length 16384 \
  >> "$LOG" 2>&1

log_status "starting frozen LongBench-v2 Long ICL calibration split"
bash scripts/launch_longicl_physical_20260716.sh >> "$LOG" 2>&1
wait_for_shards \
  20260716_longicl_physical_calibration_m14_shard 8 \
  'run_hierarchical_longicl_probe_20260716.py.*20260716_longicl_physical_calibration_m14_shard'
"$PYTHON" src/summarize_hierarchical_longbench_shards_20260716.py \
  --input_glob 'outputs/20260716_longicl_physical_calibration_m14_shard*/sample_results.csv' \
  --output_dir outputs/20260716_longicl_physical_calibration_m14_merged \
  --expected_tasks 1 \
  --expected_samples_per_method 14 \
  >> "$LOG" 2>&1

log_status "starting 32K matched physical ablation matrix"
bash scripts/launch_32k_matched_ablation_20260716.sh >> "$LOG" 2>&1

log_status "starting 128K six-topic three-window validation"
bash scripts/launch_128k_multitopic_windows_20260716.sh >> "$LOG" 2>&1
"$PYTHON" src/summarize_128k_multitopic_windows_20260716.py \
  --input_dir results/20260716_128k_multitopic_windows_w3 \
  --output_dir results/20260716_128k_multitopic_windows_w3_summary \
  --expected_cases 18 \
  --bootstrap_samples 10000 \
  >> "$LOG" 2>&1

log_status "starting 128K offloaded-exact low-peak ablation"
bash scripts/launch_128k_low_peak_ablation_20260716.sh >> "$LOG" 2>&1

log_status "starting leakage-free 128K dynamic budget router"
bash scripts/launch_router_128k_split_20260716.sh >> "$LOG" 2>&1
"$PYTHON" src/summarize_router_128k_split_20260716.py \
  --router_dir results/20260716_router_128k_split \
  --paired_dir results/20260716_128k_multitopic_windows_w3 \
  --output_dir results/20260716_router_128k_split_summary \
  --bootstrap_samples 10000 \
  >> "$LOG" 2>&1

log_status "starting independent 128K 2048-token generation pair"
bash scripts/launch_128k_long_generation_20260716.sh >> "$LOG" 2>&1

log_status "starting local-vs-remote NUMA ablation"
bash scripts/launch_128k_numa_ablation_20260716.sh >> "$LOG" 2>&1

log_status "starting true batch=1/2/4 throughput benchmark"
bash scripts/launch_batch_throughput_20260716.sh >> "$LOG" 2>&1

log_status "starting Qwen3-4B multi-model validation"
bash scripts/launch_qwen3_4b_multimodel_20260716.sh >> "$LOG" 2>&1

log_status "all resumed stages complete"
