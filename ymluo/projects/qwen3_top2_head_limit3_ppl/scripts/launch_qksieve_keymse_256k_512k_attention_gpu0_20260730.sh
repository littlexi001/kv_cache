#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TEMPLATE="${TEMPLATE:-$ROOT/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-$ROOT/results/20260730_qksieve_keymse_attention_256k_512k_b24_gpu0.json}"
GPU="${GPU:-0}"
MAX_FRACTION="${MAX_FRACTION:-0.24}"
MAX_TOKENS="${MAX_TOKENS:-125830}"
TARGET_TAIL_COUNT="${TARGET_TAIL_COUNT:-128}"

if [[ ! "$GPU" =~ ^[0-6]$ ]]; then
  echo "GPU must be one physical GPU in 0-6; got $GPU" >&2
  exit 2
fi

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$ROOT/src"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$(dirname "$OUTPUT")"
cd "$ROOT"
test -f "$TEMPLATE"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
  src/benchmark_qksieve_per_head_cold_skip_20260730.py \
  --template "$TEMPLATE" \
  --lengths 262144,524288 \
  --pool_fractions 1.0 \
  --max_fraction "$MAX_FRACTION" \
  --min_tokens 256 \
  --max_tokens "$MAX_TOKENS" \
  --target_tail_count "$TARGET_TAIL_COUNT" \
  --sample_alignment 256 \
  --sample_cap 8192 \
  --warmup 3 \
  --iterations 10 \
  --seed 20260730 \
  --output "$OUTPUT" \
  >"${OUTPUT%.json}.log" 2>&1

"$PYTHON" - "$OUTPUT" "$MAX_FRACTION" "$MAX_TOKENS" \
  "$TARGET_TAIL_COUNT" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_fraction = float(sys.argv[2])
max_tokens = int(sys.argv[3])
target_tail_count = int(sys.argv[4])
payload = json.loads(path.read_text(encoding="utf-8"))
rows = payload["rows"]
assert {int(row["history_tokens"]) for row in rows} == {262144, 524288}
for row in rows:
    assert int(row["layers"]) == 36
    history_tokens = int(row["history_tokens"])
    expected = min(
        history_tokens,
        max_tokens,
        max(256, math.ceil(max_fraction * history_tokens)),
    )
    assert int(row["selected_tokens"]) == expected
    assert int(row["target_tail_count"]) == target_tail_count
    assert int(row["quantile_sample_count"]) % 256 == 0
    assert float(row["current_qksieve_attention_model_ms"]) > 0.0
    assert float(row["full_hf_gqa_model_attention_ms"]) > 0.0
print(json.dumps({
    int(row["history_tokens"]): {
        "preexpanded_sdpa_ms": row[
            "full_sdpa_model_attention_ms"
        ],
        "hf_gqa_full_ms": row["full_hf_gqa_model_attention_ms"],
        "qksieve_attention_model_ms": row[
            "current_qksieve_attention_model_ms"
        ],
        "speedup_vs_preexpanded": row[
            "current_qksieve_speedup_vs_full"
        ],
        "speedup_vs_hf_gqa": row[
            "current_qksieve_speedup_vs_hf_gqa_full"
        ],
    }
    for row in rows
}, indent=2))
PY
