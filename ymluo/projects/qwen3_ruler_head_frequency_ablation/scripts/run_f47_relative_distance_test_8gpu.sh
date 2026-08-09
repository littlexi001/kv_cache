#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
PARENT=${PARENT:-$ROOT/outputs/multiseed_frequency_scaling_20260806}
DATA_ROOT=${DATA_ROOT:-$PARENT/f47_distance_bf16/data}
RUN=${RUN:-$PARENT/f47_relative_distance_bf16}
TEST_SEED0=${TEST_SEED0:-54}
TEST_SEED1=${TEST_SEED1:-55}
TEST_SEED2=${TEST_SEED2:-56}
PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-128}

while ! test -f "$RUN/validation.done"; do sleep 30; done

if test "${SKIP_SELECTION:-0}" != "1"; then
  "$PY" "$ROOT/src/select_test_specs.py" \
    --validation-summary "$RUN/validation/combined/summary.json" \
    --output "$RUN/specs/test.json" \
    --limit 1 \
    --control "" \
    >"$RUN/specs/test_selection.log"
fi

spec_count=$("$PY" -c "import json; print(len(json.load(open('$RUN/specs/test.json'))['specs']))")
if test "$spec_count" -lt 2; then
  echo "No relative-distance candidate passed the frozen validation rule" >"$RUN/test.no_candidate"
  exit 0
fi

run_shard() {
  local seed=$1
  local shard=$2
  local shard_count=$3
  local gpu=$4
  local data="$DATA_ROOT/ruler32k_seed${seed}_m2.jsonl"
  local out_dir="$RUN/test/seed${seed}/shard${shard}"
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
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$data" \
      --specs-json "$RUN/specs/test.json" \
      --sample-ids "$sample_ids" \
      --output-dir "$out_dir" \
      --target-length 32768 \
      --max-new-tokens-cap 128 \
      --prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --original-max-position-embeddings 40960 \
      --global-max-position 40960 \
      >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
}

run_shard "$TEST_SEED0" 0 3 0 & p0=$!
run_shard "$TEST_SEED0" 1 3 1 & p1=$!
run_shard "$TEST_SEED0" 2 3 2 & p2=$!
run_shard "$TEST_SEED1" 0 3 3 & p3=$!
run_shard "$TEST_SEED1" 1 3 4 & p4=$!
run_shard "$TEST_SEED1" 2 3 5 & p5=$!
run_shard "$TEST_SEED2" 0 2 6 & p6=$!
run_shard "$TEST_SEED2" 1 2 7 & p7=$!
wait "$p0" "$p1" "$p2" "$p3" "$p4" "$p5" "$p6" "$p7"

for seed in "$TEST_SEED0" "$TEST_SEED1" "$TEST_SEED2"; do
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$RUN/test/seed${seed}" \
    >"$RUN/test/seed${seed}/summary_stdout.log" \
    2>"$RUN/test/seed${seed}/summary_stderr.log"
done

mkdir -p "$RUN/test/combined"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "$TEST_SEED0=$RUN/test/seed${TEST_SEED0}" \
  --seed-run "$TEST_SEED1=$RUN/test/seed${TEST_SEED1}" \
  --seed-run "$TEST_SEED2=$RUN/test/seed${TEST_SEED2}" \
  --output-dir "$RUN/test/combined" \
  >"$RUN/test/combined_stdout.log" 2>"$RUN/test/combined_stderr.log"
touch "$RUN/test.done"
