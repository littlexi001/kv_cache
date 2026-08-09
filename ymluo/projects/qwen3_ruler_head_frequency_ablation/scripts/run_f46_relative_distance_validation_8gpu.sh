#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
DATA_ROOT=${DATA_ROOT:-$PARENT/data}
PREVIOUS=${PREVIOUS:-$PARENT/f47_relative_distance_bf16}
RUN=${RUN:-$PARENT/f46_relative_distance_bf16}
SPEC_GENERATOR=${SPEC_GENERATOR:-$ROOT/src/make_f46_relative_distance_specs.py}
PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-128}
VALIDATION_SEED0=${VALIDATION_SEED0:-43}
VALIDATION_SEED1=${VALIDATION_SEED1:-44}

if test -n "${WAIT_FILE:-}"; then
  while ! test -f "$WAIT_FILE"; do sleep 30; done
else
  while ! test -f "$PREVIOUS/cross.done" \
    && ! test -f "$PREVIOUS/cross.no_candidate" \
    && ! test -f "$PREVIOUS/cross.rejected"; do
    sleep 30
  done
fi

mkdir -p "$RUN/specs"
"$PY" "$SPEC_GENERATOR" \
  --output "$RUN/specs/validation.json"

run_seed() {
  local seed=$1
  local first_gpu=$2
  local outroot="$RUN/validation/seed${seed}"
  local data="$DATA_ROOT/ruler32k_seed${seed}_m1.jsonl"
  local pids=()
  mkdir -p "$outroot"
  for shard in 0 1 2 3; do
    local gpu=$((first_gpu + shard))
    local out="$outroot/shard${shard}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
        --model-name-or-path "$MODEL" \
        --examples-jsonl "$data" \
        --specs-json "$RUN/specs/validation.json" \
        --output-dir "$out" \
        --target-length 32768 \
        --max-new-tokens-cap 128 \
        --prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
        --dtype bfloat16 \
        --attn-implementation sdpa \
        --original-max-position-embeddings 40960 \
        --global-max-position 40960 \
        --spec-shard-count 4 \
        --spec-shard-index "$shard" \
        >"$out/stdout.log" 2>"$out/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$outroot" \
    >"$outroot/summary_stdout.log" 2>"$outroot/summary_stderr.log"
}

run_seed "$VALIDATION_SEED0" 0 & p0=$!
run_seed "$VALIDATION_SEED1" 4 & p1=$!
wait "$p0" "$p1"

mkdir -p "$RUN/validation/combined"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "$VALIDATION_SEED0=$RUN/validation/seed${VALIDATION_SEED0}" \
  --seed-run "$VALIDATION_SEED1=$RUN/validation/seed${VALIDATION_SEED1}" \
  --output-dir "$RUN/validation/combined" \
  >"$RUN/validation/combined_stdout.log" \
  2>"$RUN/validation/combined_stderr.log"
touch "$RUN/validation.done"
