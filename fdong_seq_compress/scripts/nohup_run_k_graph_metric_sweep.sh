#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d_%H%M%S)"
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-fdong_seq_compress/outputs/k_graph_metric_sweep_${timestamp}}"
mkdir -p "$SWEEP_OUTPUT_DIR"

log_path="$SWEEP_OUTPUT_DIR/nohup.log"
pid_path="$SWEEP_OUTPUT_DIR/pid.txt"
latest_path_file="fdong_seq_compress/outputs/k_graph_metric_sweep_latest_path.txt"

SWEEP_OUTPUT_DIR="$SWEEP_OUTPUT_DIR" \
nohup bash fdong_seq_compress/scripts/run_k_graph_metric_sweep.sh > "$log_path" 2>&1 &

pid="$!"
printf '%s\n' "$pid" > "$pid_path"
printf '%s\n' "$SWEEP_OUTPUT_DIR" > "$latest_path_file"

echo "Started K graph metric sweep in background."
echo "PID:    $pid"
echo "Output: $SWEEP_OUTPUT_DIR"
echo "Log:    $log_path"
echo
echo "Monitor:"
echo "  tail -f $log_path"
echo "  ps -p $pid"
