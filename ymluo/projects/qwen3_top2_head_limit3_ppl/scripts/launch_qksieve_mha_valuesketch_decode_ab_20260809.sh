#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/fdong/qksieve_iclr2027
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260809_qksieve_mha_valuesketch_decode_ab_v1}"
LAUNCHER="$ROOT/scripts/launch_qksieve_mha_real_decode_20260809.sh"
MODEL="$ROOT/models/Yarn-Llama-2-7b-128k"

mkdir -p "$RUN_ROOT/launcher_logs"

run_one() {
  local devices=$1 length=$2 method=$3
  local tag=${devices//,/-}
  CUDA_VISIBLE_DEVICES="$devices" \
  QKSIEVE_TRUST_REMOTE_CODE=0 \
  RUN_ROOT="$RUN_ROOT" \
  MODEL="$MODEL" \
  HISTORY_TOKENS="$length" \
  GENERATION_STEPS=64 \
  STEADY_START=16 \
  GPU_TAG="$tag" \
  bash "$LAUNCHER" "$method" \
    >"$RUN_ROOT/launcher_logs/n${length}_${method}_gpu${tag}.log" 2>&1
}

status=0

# Wave 1 uses seven cards: strict 32K A/B and the 64K no-Value control.
run_one 0,1 32768 qksieve_no_value_top1280 & p0=$!
run_one 2,3 32768 qksieve_valuesketch_top1280 & p1=$!
run_one 4,5,6 65536 qksieve_no_value_top1280 & p2=$!
for pid in $p0 $p1 $p2; do wait "$pid" || status=1; done

# Wave 2 completes the 64K pair with the same three-card topology.
run_one 0,1,2 65536 qksieve_valuesketch_top1280 || status=1

# 128K requires all eight 3090s; run the pair sequentially.
run_one 0,1,2,3,4,5,6,7 131072 qksieve_no_value_top1280 || status=1
run_one 0,1,2,3,4,5,6,7 131072 qksieve_valuesketch_top1280 || status=1

if [[ $status -eq 0 ]]; then
  touch "$RUN_ROOT/ALL_COMPLETE"
else
  touch "$RUN_ROOT/FAILED"
fi
exit "$status"
