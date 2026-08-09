#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
ROOT="${ROOT:-$PROJECT/outputs/attention_confidence_qwen3_8b_fixed_relative_328_128k_20260719}"
MIDDLE_ROOT="${MIDDLE_ROOT:-$PROJECT/outputs/attention_confidence_qwen3_8b_english_single_token_128k_20260718}"
FIXED_BODY_OVERHEAD="${FIXED_BODY_OVERHEAD:-290}"

mkdir -p "$ROOT/logs" "$ROOT/analysis"
rm -f "$ROOT/launcher.done" "$ROOT/launcher.failed"

# The variable requested by the experiment is prefix-filler length 0:128000:500.
# The clean rule block contains 34 tokens and is followed by exactly 256 filler
# tokens, so the runner's total body lengths are 290:128290:500.  This keeps the
# final evidence token exactly 328 positions before the final query token.
# Balance the four two-GPU jobs using measured per-length timings from the
# completed middle-placement sweep.
mapfile -t SHARD_LENGTHS < <(
  "$PY" - "$MIDDLE_ROOT/data" "$FIXED_BODY_OVERHEAD" <<'PY'
import glob
import json
import sys

baseline_dir = sys.argv[1]
overhead = int(sys.argv[2])
timings = {}
for path in glob.glob(baseline_dir + "/length_*.json"):
    payload = json.load(open(path, encoding="utf-8"))
    timings[int(payload["target_context_tokens"])] = float(payload["timing"]["total_seconds"])

specs = []
for filler in range(0, 128001, 500):
    total_body = filler + overhead
    nearest = min(timings, key=lambda value: abs(value - total_body))
    specs.append((total_body, timings[nearest]))

loads = [0.0] * 4
shards = [[] for _ in range(4)]
for total_body, cost in sorted(specs, key=lambda item: item[1], reverse=True):
    index = min(range(4), key=loads.__getitem__)
    shards[index].append(total_body)
    loads[index] += cost
for shard in shards:
    print(",".join(str(value) for value in sorted(shard)))
print("estimated_pair_hours=" + ",".join(f"{value / 3600:.4f}" for value in loads), file=sys.stderr)
PY
)

if [[ "${#SHARD_LENGTHS[@]}" -ne 4 ]]; then
  echo "expected four shard length lists, got ${#SHARD_LENGTHS[@]}" >&2
  date -Is > "$ROOT/launcher.failed"
  exit 1
fi

run_shard() {
  local shard="$1"
  local devices="$2"
  local lengths="$3"
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PY" -u "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$ROOT" \
    --lengths "$lengths" \
    --seed 0 \
    --code_mode english_single_token \
    --placement recent \
    --query_mode full2 \
    --prompt_style legacy \
    --max_top 100 \
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-128}" \
    --dtype float16 \
    --device_map balanced \
    --attn_implementation sdpa \
    --original_max_position_embeddings 40960 \
    --global_max_position 130000 \
    --shard_label "$shard" \
    >"$ROOT/logs/$shard.log" 2>&1
}

declare -a pids
run_shard shard0 0,1 "${SHARD_LENGTHS[0]}" & pids[0]=$!
run_shard shard1 2,3 "${SHARD_LENGTHS[1]}" & pids[1]=$!
run_shard shard2 4,5 "${SHARD_LENGTHS[2]}" & pids[2]=$!
run_shard shard3 6,7 "${SHARD_LENGTHS[3]}" & pids[3]=$!

status=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  date -Is > "$ROOT/launcher.failed"
  exit 1
fi

"$PY" "$PROJECT/src/analyze_attention_confidence_sweep.py" --output_dir "$ROOT" \
  >"$ROOT/logs/aggregate.log" 2>&1
"$PY" "$PROJECT/src/analyze_fixed_relative_distance_sweep.py" \
  --fixed_data_dir "$ROOT/data" \
  --middle_data_dir "$MIDDLE_ROOT/data" \
  --fixed_body_overhead "$FIXED_BODY_OVERHEAD" \
  --output_dir "$ROOT/analysis" \
  >"$ROOT/logs/fixed_relative_analysis.log" 2>&1

date -Is > "$ROOT/launcher.done"
