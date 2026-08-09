#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
RUN=${RUN:-$PARENT/f46_adaptive_gate_bf16}
DATA_ROOT=${DATA_ROOT:-$PARENT/adaptive_frozen_data}

while ! test -f "$RUN/validation.done"; do sleep 30; done

"$PY" "$ROOT/src/select_adaptive_test_specs.py" \
  --validation-summary "$RUN/validation/combined/summary.json" \
  --output "$RUN/specs/test.json" \
  >"$RUN/specs/test_selection.log"

RUN="$RUN" DATA_ROOT="$DATA_ROOT" \
  TEST_SEED0=60 TEST_SEED1=61 TEST_SEED2=62 \
  SKIP_SELECTION=1 PREFILL_CHUNK_SIZE=64 \
  bash "$ROOT/scripts/run_f47_relative_distance_test_8gpu.sh"

RUN="$RUN" PREFILL_CHUNK_SIZE=64 \
  bash "$ROOT/scripts/run_f47_relative_cross_8gpu.sh"

touch "$RUN/adaptive_pipeline.done"
