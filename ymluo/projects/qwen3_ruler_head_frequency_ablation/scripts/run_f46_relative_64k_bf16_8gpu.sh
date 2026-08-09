#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
SOURCE_RUN=${SOURCE_RUN:-$PARENT/f46_relative_distance_bf16}
WAIT_RUN=${WAIT_RUN:-$PARENT/f46_global_continuous_bf16}
DATA_ROOT=${DATA_ROOT:-$PARENT/f47_distance_bf16_exactprefix/long64_data}
OUT=${OUT:-$SOURCE_RUN/long64_bf16_multiseed}
SPECS=${SPECS:-$SOURCE_RUN/specs/test.json}
PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-128}
TEST_SEED0=${TEST_SEED0:-57}
TEST_SEED1=${TEST_SEED1:-58}
TEST_SEED2=${TEST_SEED2:-59}

while ! test -f "$WAIT_RUN/cross.done" \
  && ! test -f "$WAIT_RUN/cross.rejected" \
  && ! test -f "$WAIT_RUN/cross.no_candidate"; do
  sleep 30
done

mkdir -p "$OUT"

first_sample_id=$(
  "$PY" - "$DATA_ROOT/ruler64k_seed${TEST_SEED0}_m1.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.loads(next(handle))["sample_id"])
PY
)

smoke="$OUT/smoke"
mkdir -p "$smoke"
if ! test -f "$OUT/smoke.done"; then
  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$DATA_ROOT/ruler64k_seed${TEST_SEED0}_m1.jsonl" \
      --specs-json "$SPECS" \
      --sample-ids "$first_sample_id" \
      --output-dir "$smoke" \
      --target-length 65536 \
      --max-new-tokens-cap 128 \
      --prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --device-map balanced \
      --original-max-position-embeddings 40960 \
      --global-max-position 131072 \
      >"$smoke/stdout.log" 2>"$smoke/stderr.log"
  touch "$OUT/smoke.done"
fi

run_shard() {
  local seed=$1
  local shard=$2
  local shard_count=$3
  local gpu_pair=$4
  local data="$DATA_ROOT/ruler64k_seed${seed}_m1.jsonl"
  local out_dir="$OUT/seed${seed}/shard${shard}"
  local sample_ids
  sample_ids=$(
    "$PY" - "$data" "$shard" "$shard_count" <<'PY'
import json
import sys

path, shard, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = []
with open(path, encoding="utf-8") as handle:
    for index, line in enumerate(handle):
        if index % count == shard:
            ids.append(json.loads(line)["sample_id"])
print(",".join(ids))
PY
  )
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu_pair" TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$data" \
      --specs-json "$SPECS" \
      --sample-ids "$sample_ids" \
      --output-dir "$out_dir" \
      --target-length 65536 \
      --max-new-tokens-cap 128 \
      --prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --device-map balanced \
      --original-max-position-embeddings 40960 \
      --global-max-position 131072 \
      >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
}

run_shard "$TEST_SEED0" 0 2 0,1 & p0=$!
run_shard "$TEST_SEED0" 1 2 2,3 & p1=$!
run_shard "$TEST_SEED1" 0 2 4,5 & p2=$!
run_shard "$TEST_SEED1" 1 2 6,7 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

run_shard "$TEST_SEED2" 0 4 0,1 & p0=$!
run_shard "$TEST_SEED2" 1 4 2,3 & p1=$!
run_shard "$TEST_SEED2" 2 4 4,5 & p2=$!
run_shard "$TEST_SEED2" 3 4 6,7 & p3=$!
wait "$p0" "$p1" "$p2" "$p3"

for seed in "$TEST_SEED0" "$TEST_SEED1" "$TEST_SEED2"; do
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$OUT/seed${seed}" \
    >"$OUT/seed${seed}/summary_stdout.log" \
    2>"$OUT/seed${seed}/summary_stderr.log"
done

mkdir -p "$OUT/combined"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "$TEST_SEED0=$OUT/seed${TEST_SEED0}" \
  --seed-run "$TEST_SEED1=$OUT/seed${TEST_SEED1}" \
  --seed-run "$TEST_SEED2=$OUT/seed${TEST_SEED2}" \
  --output-dir "$OUT/combined" \
  >"$OUT/combined_stdout.log" 2>"$OUT/combined_stderr.log"
touch "$OUT/run.done"
