#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
WAIT_FILE=${WAIT_FILE:-$PARENT/f46_adaptive_gate_bf16/long64_bf16_multiseed/run.done}
RUN=${RUN:-$PARENT/f46_semantic_topk_bf16}
STAGED=${STAGED:-$ROOT/staging/f46_semantic_topk/head_frequency_intervention.py}

while ! test -f "$WAIT_FILE"; do sleep 30; done
while ! test -f "$RUN/data_validation/data.done"; do sleep 30; done
while ! test -f "$RUN/data_frozen/data.done"; do sleep 30; done

cp "$STAGED" "$ROOT/src/head_frequency_intervention.py"
"$PY" -m py_compile "$ROOT/src/head_frequency_intervention.py"

RUN="$RUN" DATA_ROOT="$RUN/data_validation" \
  VALIDATION_SEED0=66 VALIDATION_SEED1=67 \
  SPEC_GENERATOR="$ROOT/src/make_f46_semantic_topk_specs.py" \
  PREFILL_CHUNK_SIZE=64 WAIT_FILE="$WAIT_FILE" \
  bash "$ROOT/scripts/run_f46_relative_distance_validation_8gpu.sh"

"$PY" "$ROOT/src/select_semantic_test_specs.py" \
  --validation-summary "$RUN/validation/combined/summary.json" \
  --output "$RUN/specs/test.json" \
  >"$RUN/specs/test_selection.log"

RUN="$RUN" DATA_ROOT="$RUN/data_frozen" \
  TEST_SEED0=68 TEST_SEED1=69 TEST_SEED2=70 \
  SKIP_SELECTION=1 PREFILL_CHUNK_SIZE=64 \
  bash "$ROOT/scripts/run_f47_relative_distance_test_8gpu.sh"

RUN="$RUN" PREFILL_CHUNK_SIZE=64 \
  bash "$ROOT/scripts/run_f47_relative_cross_8gpu.sh"

touch "$RUN/semantic_pipeline.done"
