#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
OLD_RUN=${OLD_RUN:-$PARENT/f47_distance_bf16}
RUN=${RUN:-$PARENT/f47_distance_bf16_exactprefix}
GRID=${GRID:-$PARENT/specs/f47_distance_conditioned.json}

mkdir -p "$RUN/logs"

# Do not compete with the already-running reference length sweep.
while ! test -f "$OLD_RUN/length_transfer/launcher.done" \
  && ! test -f "$OLD_RUN/length_transfer/aborted_bf16_rounding_bug"; do
  sleep 30
done

PARENT="$PARENT" RUN="$RUN" SPECS="$GRID" \
  bash "$ROOT/scripts/run_f47_distance_bf16_validation_6gpu.sh" \
  >"$RUN/logs/validation.log" 2>&1

PARENT="$PARENT" RUN="$RUN" DATA_RUN="$OLD_RUN" \
  bash "$ROOT/scripts/run_f47_distance_bf16_test_6gpu.sh" \
  >"$RUN/logs/test.log" 2>&1

bash "$ROOT/scripts/run_cross_benchmarks_8gpu.sh" "$RUN" \
  >"$RUN/logs/cross.log" 2>&1

PARENT="$PARENT" METHOD_RUN="$RUN" \
  bash "$ROOT/scripts/run_f47_distance_length_transfer_6gpu.sh" \
  >"$RUN/logs/length.log" 2>&1

touch "$RUN/pipeline.done"
