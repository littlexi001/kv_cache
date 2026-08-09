#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
OLD_REFERENCE=$ROOT/results/20260727_quantized_offloaded_prefill_group16_gpu7
FULL_REFERENCE=$ROOT/results/20260727_qk_variable_physical_128k_4gpu
RUN_ROOT=$ROOT/results/20260727_direct_quantized_key_conversion_3gpu
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"
trap 'touch "$RUN_ROOT/TERMINAL"' EXIT

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
)

run_32k() {
  local gpu="$1"
  local name="$2"
  shift 2
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$RUN_ROOT/${name}_32k.json" \
    --history_tokens 32000 \
    --eval_tokens 64 \
    "$@" \
    "${common[@]}" \
    >"$LOG_ROOT/${name}_32k.log" 2>&1
}

run_32k 0 exact_reuse \
  --prefill_cache_mode offloaded_exact \
  --prefill_conversion_source exact_host &
pid0=$!
run_32k 1 group16_exact_reuse \
  --prefill_cache_mode quantized_offloaded_exact \
  --prefill_quantization_bits 4 \
  --prefill_quantization_group_size 16 \
  --prefill_conversion_source exact_host &
pid1=$!
run_32k 2 group16_direct \
  --prefill_cache_mode quantized_offloaded_exact \
  --prefill_quantization_bits 4 \
  --prefill_quantization_group_size 16 \
  --prefill_conversion_source transient_quantized_key &
pid2=$!

failed=0
for pid in "$pid0" "$pid1" "$pid2"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

direct_quality=$(
  "$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
exact_index = json.loads(
    (root / "group16_exact_reuse_32k.json").read_text(encoding="utf-8")
)
direct = json.loads(
    (root / "group16_direct_32k.json").read_text(encoding="utf-8")
)
if exact_index["target_token_ids"] != direct["target_token_ids"]:
    raise RuntimeError("32K target token mismatch")
print(100.0 * float(exact_index["ppl"]) / float(direct["ppl"]))
PY
)
echo "32K direct-transcode quality versus exact-key conversion: $direct_quality"

if "$PYTHON" - "$direct_quality" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 99.5 else 1)
PY
then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$RUN_ROOT/group16_direct_128k.json" \
    --history_tokens 128000 \
    --eval_tokens 256 \
    --prefill_cache_mode quantized_offloaded_exact \
    --prefill_quantization_bits 4 \
    --prefill_quantization_group_size 16 \
    --prefill_conversion_source transient_quantized_key \
    "${common[@]}" \
    >"$LOG_ROOT/group16_direct_128k.log" 2>&1
fi

"$PYTHON" - "$RUN_ROOT" "$OLD_REFERENCE" "$FULL_REFERENCE" <<'PY'
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
old_reference = Path(sys.argv[2])
full_reference = Path(sys.argv[3])
names = ("exact_reuse", "group16_exact_reuse", "group16_direct")
payloads = {
    name: json.loads(
        (run_root / f"{name}_32k.json").read_text(encoding="utf-8")
    )
    for name in names
}
reference_ids = payloads["exact_reuse"]["target_token_ids"]
rows = []
for name, payload in payloads.items():
    if payload["target_token_ids"] != reference_ids:
        raise RuntimeError(f"32K target token mismatch for {name}")
    rows.append(
        {
            "method": name,
            "ppl": float(payload["ppl"]),
            "quality_vs_exact_reuse_percent": (
                100.0
                * float(payloads["exact_reuse"]["ppl"])
                / float(payload["ppl"])
            ),
            "prefill_seconds_including_query": float(
                payload["prefill_seconds"]
            ),
            "conversion_seconds": float(
                payload["cache_conversion_seconds"]
            ),
            "fixed_seconds": float(
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
    "schema": "direct_quantized_key_conversion_v1",
    "paired_32k": rows,
    "direct_quality_vs_group16_exact_key_percent": (
        100.0
        * float(payloads["group16_exact_reuse"]["ppl"])
        / float(payloads["group16_direct"]["ppl"])
    ),
    "ran_128k": (run_root / "group16_direct_128k.json").exists(),
}
if summary["ran_128k"]:
    direct = json.loads(
        (run_root / "group16_direct_128k.json").read_text(encoding="utf-8")
    )
    old = json.loads(
        (
            old_reference / "mixed_a_w2_128k_int4_group16.json"
        ).read_text(encoding="utf-8")
    )
    physical = json.loads(
        (
            full_reference / "mixed_a_w2_qkfixed4421sampled.json"
        ).read_text(encoding="utf-8")
    )
    if not (
        direct["target_token_ids"]
        == old["target_token_ids"]
        == physical["target_token_ids"]
    ):
        raise RuntimeError("128K target token mismatch")
    full_ppl = 10.543396400317109
    full_fixed_seconds = 217.42699244990945
    full_online_seconds = 75.79606706742197
    direct_fixed = float(direct["prefill_plus_conversion_seconds"])
    direct_online = float(direct["synchronized_model_forward_seconds"])
    token_count = len(direct["token_nll"])
    online_saving_per_token = (
        full_online_seconds - direct_online
    ) / token_count
    summary["direct_128k"] = {
        "ppl": float(direct["ppl"]),
        "quality_vs_full_percent": (
            100.0 * full_ppl / float(direct["ppl"])
        ),
        "quality_vs_group16_exact_key_percent": (
            100.0 * float(old["ppl"]) / float(direct["ppl"])
        ),
        "prefill_seconds_including_query": float(
            direct["prefill_seconds"]
        ),
        "conversion_seconds": float(
            direct["cache_conversion_seconds"]
        ),
        "fixed_seconds": direct_fixed,
        "old_group16_exact_key_fixed_seconds": float(
            old["prefill_plus_conversion_seconds"]
        ),
        "online_seconds": direct_online,
        "online_speedup": full_online_seconds / direct_online,
        "total_speedup_at_256_tokens": (
            (full_fixed_seconds + full_online_seconds)
            / (direct_fixed + direct_online)
        ),
        "break_even_tokens": (
            max(0.0, direct_fixed - full_fixed_seconds)
            / online_saving_per_token
            if online_saving_per_token > 0.0
            else None
        ),
        "peak_gpu_allocated_gib": float(
            direct["process_peak_gpu_allocated_during_prefill_conversion"]
        )
        / (1024.0**3),
    }
(run_root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
