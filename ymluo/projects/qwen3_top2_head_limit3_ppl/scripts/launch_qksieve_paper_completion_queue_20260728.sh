#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
UPSTREAM_PID="${UPSTREAM_PID:-0}"
UPSTREAM_ROOT="$ROOT/results/20260728_qksieve_submission_queue_6gpu"
QUEUE_ROOT="$ROOT/results/20260728_qksieve_paper_completion_queue"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export QKSIEVE_GPUS=0,1,2,3,4
mkdir -p "$QUEUE_ROOT/logs"
cd "$ROOT"

if [[ "$UPSTREAM_PID" =~ ^[1-9][0-9]*$ ]]; then
  while kill -0 "$UPSTREAM_PID" 2>/dev/null; do
    sleep 60
  done
fi

if [[ ! -e "$UPSTREAM_ROOT/ALL_COMPLETE" ]]; then
  echo "upstream LongBench/RULER queue exited without ALL_COMPLETE" >&2
  exit 1
fi

run_stage() {
  local name="$1"
  local marker="$2"
  shift 2
  if [[ -e "$marker" ]]; then
    echo "[skip] $name"
    return
  fi
  echo "[start] $name $(date --iso-8601=seconds)"
  "$@" >"$QUEUE_ROOT/logs/$name.log" 2>&1
  if [[ ! -e "$marker" ]]; then
    echo "$name exited without marker $marker" >&2
    exit 1
  fi
  echo "[done] $name $(date --iso-8601=seconds)"
}

run_stage \
  uniform1_ablation_m20 \
  "$ROOT/results/20260728_qksieve_uniform1_fier_longbench_m20_paired_5gpu/ALL_COMPLETE" \
  env QKSIEVE_GPUS=0,1,2,3,4 \
  bash scripts/launch_qksieve_uniform1_ablation_longbench_m20_5gpu_20260728.sh

run_stage \
  fier_packed_longbench \
  "$ROOT/results/20260728_qksieve_fier_packed_longbench_llama31_8b_paired_5gpu/ALL_COMPLETE" \
  env QKSIEVE_GPUS=0,1,2,3,4 \
  bash scripts/launch_qksieve_fier_packed_longbench_5gpu_20260728.sh

run_stage \
  public_selectors_longbench \
  "$ROOT/results/20260728_qksieve_public_selectors_longbench_official_middle_5gpu/ALL_COMPLETE" \
  env QKSIEVE_GPUS=0,1,2,3,4 \
  bash scripts/launch_qksieve_public_selectors_longbench_5gpu_20260728.sh

run_stage \
  frozen_samepath_length \
  "$ROOT/results/20260728_qksieve_frozen_samepath_length_6gpu/ALL_COMPLETE" \
  bash scripts/launch_qksieve_frozen_samepath_length_6gpu_20260728.sh

run_stage \
  free_generation_drift \
  "$ROOT/results/20260728_qksieve_free_generation_drift_6gpu/ALL_COMPLETE" \
  bash scripts/launch_qksieve_free_generation_drift_6gpu_20260728.sh

run_stage \
  teacher_forced_drift \
  "$ROOT/results/20260728_qksieve_teacher_forced_drift_32k_4k_6gpu/ALL_COMPLETE" \
  bash scripts/launch_qksieve_teacher_forced_drift_6gpu_20260728.sh

touch "$QUEUE_ROOT/ALL_COMPLETE"
