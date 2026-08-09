#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TEMPLATE="${TEMPLATE:-$ROOT/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260730_qksieve_tail_c_speed_ablation_gpu01}"
GPU_C16="${GPU_C16:-0}"
GPU_C64="${GPU_C64:-1}"

for gpu in "$GPU_C16" "$GPU_C64"; do
  if [[ ! "$gpu" =~ ^[0-6]$ ]]; then
    echo "GPU ids are restricted to 0-6; got $gpu" >&2
    exit 2
  fi
done
if [[ "$GPU_C16" == "$GPU_C64" ]]; then
  echo "GPU_C16 and GPU_C64 must differ" >&2
  exit 2
fi

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$ROOT/src"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RUN_ROOT"
cd "$ROOT"
test -f "$TEMPLATE"

run_case() {
  local gpu="$1"
  local target_tail_count="$2"
  local output="$RUN_ROOT/c${target_tail_count}.json"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/benchmark_qksieve_per_head_cold_skip_20260730.py \
    --template "$TEMPLATE" \
    --lengths 8192,16384,32768,65536,131072 \
    --pool_fractions 1.0 \
    --max_fraction 0.06 \
    --min_tokens 256 \
    --max_tokens 1280 \
    --target_tail_count "$target_tail_count" \
    --sample_alignment 256 \
    --sample_cap 8192 \
    --warmup 3 \
    --iterations 10 \
    --seed 20260730 \
    --output "$output" \
    >"${output%.json}.log" 2>&1
}

run_case "$GPU_C16" 16 &
pid_c16=$!
run_case "$GPU_C64" 64 &
pid_c64=$!

status=0
wait "$pid_c16" || status=$?
wait "$pid_c64" || status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON" - "$RUN_ROOT/c16.json" "$RUN_ROOT/c64.json" \
  "$RUN_ROOT/summary.json" <<'PY'
import json
import sys
from pathlib import Path

c16 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
c64 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
by_length = {}
for label, payload in (("c16", c16), ("c64", c64)):
    for row in payload["rows"]:
        length = int(row["history_tokens"])
        by_length.setdefault(length, {})[label] = {
            "sample_count": int(row["quantile_sample_count"]),
            "candidate_tokens_mean": float(row["candidate_tokens_mean"]),
            "attention_model_ms": float(
                row["current_qksieve_attention_model_ms"]
            ),
            "speedup_vs_full": float(
                row["current_qksieve_speedup_vs_full"]
            ),
        }
for length, methods in by_length.items():
    assert set(methods) == {"c16", "c64"}, (length, methods)
    methods["c64_vs_c16"] = (
        methods["c16"]["attention_model_ms"]
        / methods["c64"]["attention_model_ms"]
    )
result = {
    "schema": "qksieve_tail_c_speed_ablation_v1",
    "rows": [
        {"history_tokens": length, **methods}
        for length, methods in sorted(by_length.items())
    ],
}
Path(sys.argv[3]).write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
