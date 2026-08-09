#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
REFERENCE=$ROOT/results/20260727_quantized_offloaded_prefill_gpu7
RUN_ROOT=$ROOT/results/20260727_quantized_offloaded_prefill_group16_gpu7
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"
trap 'touch "$RUN_ROOT/TERMINAL"' EXIT

if [[ ! -e "$REFERENCE/ALL_COMPLETE" ]]; then
  echo "missing completed per-head quantization reference" >&2
  exit 1
fi

common=(
  --model_name_or_path "$MODEL"
  --topic mixed_a
  --query_tokens 256
  --window_index 2
  --window_stride_tokens 128512
  --index_mode qk_variable
  --qk_metric_query_shrinkage 0.75
  --variable_rate_budget 15
  --fixed_bit_allocation 4,4,2,1,0,0,0,0
  --candidate_fraction 0.06
  --candidate_min_tokens 256
  --candidate_max_tokens 1280
  --retrieval_backend sampled_compact
  --sampled_candidate_multiplier 1.5
  --attention_fraction 0.06
  --candidate_selection_mode per_head_stream
  --stream_group_size 1
  --exact_cache_fraction 0.032
  --directory_backend fused
  --prefill_chunk_tokens 4096
  --dtype float16
  --device cuda
  --device_map auto
  --prefill_cache_mode quantized_offloaded_exact
  --prefill_quantization_bits 4
  --prefill_quantization_group_size 16
)

CUDA_VISIBLE_DEVICES=7 "$PYTHON" -u \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --output "$RUN_ROOT/mixed_a_w2_32k_int4_group16.json" \
  --history_tokens 32000 \
  --eval_tokens 64 \
  "${common[@]}" \
  >"$LOG_ROOT/32k_int4_group16.log" 2>&1

quality=$(
  "$PYTHON" - "$REFERENCE" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

reference = Path(sys.argv[1])
run_root = Path(sys.argv[2])
exact = json.loads(
    (reference / "mixed_a_w2_32k_offloaded_exact.json").read_text(
        encoding="utf-8"
    )
)
groupwise = json.loads(
    (run_root / "mixed_a_w2_32k_int4_group16.json").read_text(
        encoding="utf-8"
    )
)
if exact["target_token_ids"] != groupwise["target_token_ids"]:
    raise RuntimeError("32K target token mismatch")
print(100.0 * float(exact["ppl"]) / float(groupwise["ppl"]))
PY
)
echo "32K groupwise-INT4 quality versus exact: $quality"

# Only spend the 128K run when the mechanistic 32K test recovers practical quality.
if "$PYTHON" - "$quality" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 99.0 else 1)
PY
then
  CUDA_VISIBLE_DEVICES=7 "$PYTHON" -u \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$RUN_ROOT/mixed_a_w2_128k_int4_group16.json" \
    --history_tokens 128000 \
    --eval_tokens 256 \
    "${common[@]}" \
    >"$LOG_ROOT/128k_int4_group16.log" 2>&1
fi

"$PYTHON" - "$REFERENCE" "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

reference = Path(sys.argv[1])
run_root = Path(sys.argv[2])
paths = {
    "offloaded_exact": reference / "mixed_a_w2_32k_offloaded_exact.json",
    "transient_int8_per_head": reference / "mixed_a_w2_32k_int8.json",
    "transient_int4_per_head": reference / "mixed_a_w2_32k_int4.json",
    "transient_int4_group16": (
        run_root / "mixed_a_w2_32k_int4_group16.json"
    ),
}
payloads = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in paths.items()
}
exact = payloads["offloaded_exact"]
rows = []
for name, payload in payloads.items():
    if exact["target_token_ids"] != payload["target_token_ids"]:
        raise RuntimeError(f"target token mismatch for {name}")
    rows.append(
        {
            "method": name,
            "ppl": float(payload["ppl"]),
            "quality_vs_exact_percent": (
                100.0 * float(exact["ppl"]) / float(payload["ppl"])
            ),
            "prefill_plus_conversion_seconds": float(
                payload["prefill_plus_conversion_seconds"]
            ),
            "online_seconds": float(
                payload["synchronized_model_forward_seconds"]
            ),
            "peak_gpu_allocated_gib": float(
                payload["process_peak_gpu_allocated_during_prefill_conversion"]
            )
            / (1024.0**3),
        }
    )
summary = {
    "schema": "groupwise_int4_prefill_probe_v1",
    "quantization_group_size": 16,
    "paired_32k": rows,
    "ran_128k": (
        run_root / "mixed_a_w2_128k_int4_group16.json"
    ).exists(),
}
(run_root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
