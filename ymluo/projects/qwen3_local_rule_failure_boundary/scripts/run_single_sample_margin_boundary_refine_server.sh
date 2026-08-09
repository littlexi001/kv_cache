#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/single_sample_margin_boundary_refine_20260723}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FREE_MEMORY_USED_MB="${FREE_MEMORY_USED_MB:-2000}"
FREE_UTILIZATION="${FREE_UTILIZATION:-10}"

mkdir -p "$OUT/data"
exec 8>"$OUT/launcher.lock"
if ! flock -n 8; then
  echo "another refinement launcher already owns $OUT/launcher.lock"
  exit 0
fi
if [[ -f "$OUT/launcher.done" ]]; then
  echo "already complete: $OUT"
  exit 0
fi
rm -f "$OUT/launcher.failed"
echo "$$" >"$OUT/launcher.pid"

choose_free_gpu() {
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, \
      -v max_mem="$FREE_MEMORY_USED_MB" \
      -v max_util="$FREE_UTILIZATION" \
      '{
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        gsub(/ /, "", $3);
        if (($1 + 0) >= 4 && ($1 + 0) <= 7 &&
            ($2 + 0) < max_mem && ($3 + 0) < max_util) {
          print $1;
          exit;
        }
      }'
}

gpu=""
while [[ -z "$gpu" ]]; do
  gpu="$(choose_free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    echo "$(date -Is) waiting: no idle GPU"
    sleep "$POLL_SECONDS"
  fi
done
echo "$(date -Is) selected idle GPU $gpu"

common_args=(
  --model_name_or_path "$MODEL"
  --output_dir "$OUT"
  --seed 0
  --code_mode english_single_token
  --placement middle
  --query_mode full2
  --prompt_style legacy
  --max_top 100
  --prefill_chunk_size 128
  --dtype float16
  --device cuda
  --device_map none
  --attn_implementation sdpa
  --original_max_position_embeddings 40960
  --global_max_position 130000
)

# The two verified rule lines occupy 34 tokens. Target lengths 1..33 are
# structurally invalid, while length=0 is the special no-filler baseline.
coarse_lengths="0,34,$(seq 50 25 500 | paste -sd, -)"
CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
  "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
  "${common_args[@]}" \
  --lengths "$coarse_lengths" \
  --shard_label coarse_25 \
  >"$OUT/coarse.log" 2>&1

fine_lengths="$(
  "$PY" - "$OUT/data" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
paths = sorted(
    root.glob("length_*.json"),
    key=lambda path: int(path.stem.split("_")[-1]),
)
for path in paths:
    length = int(path.stem.split("_")[-1])
    payload = json.loads(path.read_text())
    answer = payload["answer"]
    gold = answer["gold_token_scores"][0]
    gold_id = int(gold["token_id"])
    wrong = max(
        (
            item
            for item in answer["next_token_top5"]
            if int(item["token_id"]) != gold_id
        ),
        key=lambda item: float(item["probability"]),
    )
    margin = math.log(float(gold["probability"])) - math.log(
        float(wrong["probability"])
    )
    rows.append((length, margin))

for (left_length, left_margin), (right_length, right_margin) in zip(rows, rows[1:]):
    if left_margin > 0.0 and right_margin <= 0.0:
        first_valid = max(34, left_length)
        print(",".join(str(length) for length in range(first_valid, right_length + 1)))
        break
else:
    raise SystemExit("no positive-to-nonpositive margin crossing in 0..500")
PY
)"
echo "fine lengths: $fine_lengths"

CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u \
  "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
  "${common_args[@]}" \
  --lengths "$fine_lengths" \
  --shard_label fine_1 \
  >"$OUT/fine.log" 2>&1

"$PY" "$PROJECT/src/analyze_single_sample_failure_trace.py" \
  --input_dir "$OUT/data" \
  --output_dir "$OUT/analysis" \
  >"$OUT/analysis.log" 2>&1

date -Is >"$OUT/launcher.done"
echo "$(date -Is) complete: $OUT"
