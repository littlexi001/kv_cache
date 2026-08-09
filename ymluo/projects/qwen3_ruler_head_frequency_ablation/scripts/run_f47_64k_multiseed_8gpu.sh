#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
RUN=${RUN:-$ROOT/outputs/multiseed_frequency_scaling_20260806/f47_distance_bf16_exactprefix}
SOURCE_SPECS=${SOURCE_SPECS:-$RUN/specs/test.json}
OUT=${OUT:-$RUN/long64_multiseed}
KEEP_NAMES=${KEEP_NAMES:-native_rope,l18_23_g4_f47_piecewise_s16384_a0.25}

while ! test -f "$RUN/pipeline.done" || ! test -f "$RUN/long64_data/data.done"; do
  sleep 30
done

mkdir -p "$OUT/specs"
"$PY" - "$SOURCE_SPECS" "$OUT/specs/selected.json" "$KEEP_NAMES" <<'PY'
import json
import sys
from pathlib import Path

source, output, keep_csv = sys.argv[1:]
payload = json.loads(Path(source).read_text(encoding="utf-8"))
keep = [name.strip() for name in keep_csv.split(",") if name.strip()]
by_name = {spec["name"]: spec for spec in payload["specs"]}
missing = [name for name in keep if name not in by_name]
if missing:
    raise SystemExit(f"Missing requested specs: {missing}")
payload["specs"] = [by_name[name] for name in keep]
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

SPECS="$OUT/specs/selected.json"

run_shard() {
  local seed=$1
  local shard=$2
  local shard_count=$3
  local gpu=$4
  local data="$RUN/long64_data/ruler64k_seed${seed}_m1.jsonl"
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
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$data" \
      --specs-json "$SPECS" \
      --sample-ids "$sample_ids" \
      --output-dir "$out_dir" \
      --target-length 65536 \
      --max-new-tokens-cap 128 \
      --prefill-chunk-size 256 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --load-in-4bit \
      --original-max-position-embeddings 40960 \
      --global-max-position 131072 \
      >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
}

run_shard 57 0 3 0 & p0=$!
run_shard 57 1 3 1 & p1=$!
run_shard 57 2 3 2 & p2=$!
run_shard 58 0 3 3 & p3=$!
run_shard 58 1 3 4 & p4=$!
run_shard 58 2 3 5 & p5=$!
run_shard 59 0 2 6 & p6=$!
run_shard 59 1 2 7 & p7=$!
wait "$p0" "$p1" "$p2" "$p3" "$p4" "$p5" "$p6" "$p7"

for seed in 57 58 59; do
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$OUT/seed${seed}" \
    >"$OUT/seed${seed}/summary_stdout.log" 2>"$OUT/seed${seed}/summary_stderr.log"
done

mkdir -p "$OUT/combined"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "57=$OUT/seed57" \
  --seed-run "58=$OUT/seed58" \
  --seed-run "59=$OUT/seed59" \
  --output-dir "$OUT/combined" \
  >"$OUT/combined_stdout.log" 2>"$OUT/combined_stderr.log"
touch "$OUT/run.done"
