#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
RUN=${RUN:-$PARENT/f47_distance_bf16}
SPECS=${SPECS:-$PARENT/specs/f47_distance_conditioned.json}
GPU_IDS=(2 3 4 5 6 7)

mkdir -p "$RUN"

run_seed() {
  local seed=$1
  local outroot="$RUN/validation/seed${seed}"
  local data="$PARENT/data/ruler32k_seed${seed}_m1.jsonl"
  local shards=${#GPU_IDS[@]}
  local pids=()
  mkdir -p "$outroot"
  for shard in $(seq 0 $((shards - 1))); do
    local gpu=${GPU_IDS[$shard]}
    local out="$outroot/shard${shard}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$data" \
      --specs-json "$SPECS" \
      --output-dir "$out" \
      --target-length 32768 \
      --max-new-tokens-cap 128 \
      --prefill-chunk-size 128 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --spec-shard-count "$shards" \
      --spec-shard-index "$shard" \
      >"$out/stdout.log" 2>"$out/stderr.log" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$outroot" \
    >"$outroot/summary_stdout.log" 2>"$outroot/summary_stderr.log"
  touch "$outroot/stage.done"
}

run_seed 43
run_seed 44

mkdir -p "$RUN/validation/combined" "$RUN/specs"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "43=$RUN/validation/seed43" \
  --seed-run "44=$RUN/validation/seed44" \
  --output-dir "$RUN/validation/combined" \
  >"$RUN/validation/combined_stdout.log" 2>"$RUN/validation/combined_stderr.log"
"$PY" "$ROOT/src/select_test_specs.py" \
  --validation-summary "$RUN/validation/combined/summary.json" \
  --output "$RUN/specs/test.json" \
  --limit 2 --control "" \
  >"$RUN/specs/test_selection.log"
touch "$RUN/validation.done"
