#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
REFERENCE=$ROOT/results/20260727_qkmetric_qscale_128k_holdout
OFFLOADED128=$ROOT/results/20260727_qk_variable_physical_128k_4gpu
RUN_ROOT=$ROOT/results/20260727_quantized_offloaded_prefill_gpu7
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

CUDA_VISIBLE_DEVICES=7 "$PYTHON" -u \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --output "$RUN_ROOT/mixed_a_w2_32k_offloaded_exact.json" \
  --history_tokens 32000 \
  --eval_tokens 64 \
  --prefill_cache_mode offloaded_exact \
  "${common[@]}" \
  >"$LOG_ROOT/32k_offloaded_exact.log" 2>&1

for bits in 8 4; do
  CUDA_VISIBLE_DEVICES=7 "$PYTHON" -u \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$RUN_ROOT/mixed_a_w2_32k_int${bits}.json" \
    --history_tokens 32000 \
    --eval_tokens 64 \
    --prefill_cache_mode quantized_offloaded_exact \
    --prefill_quantization_bits "$bits" \
    "${common[@]}" \
    >"$LOG_ROOT/32k_int${bits}.log" 2>&1
done

CUDA_VISIBLE_DEVICES=7 "$PYTHON" -u \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --output "$RUN_ROOT/mixed_a_w2_128k_int4.json" \
  --history_tokens 128000 \
  --eval_tokens 256 \
  --prefill_cache_mode quantized_offloaded_exact \
  --prefill_quantization_bits 4 \
  "${common[@]}" \
  >"$LOG_ROOT/128k_int4.log" 2>&1

"$PYTHON" - "$RUN_ROOT" "$REFERENCE" "$OFFLOADED128" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
reference = Path(sys.argv[2])
offloaded_root = Path(sys.argv[3])

rows_32k = []
paths_32k = {
    "offloaded_exact": run_root / "mixed_a_w2_32k_offloaded_exact.json",
    "transient_int8": run_root / "mixed_a_w2_32k_int8.json",
    "transient_int4": run_root / "mixed_a_w2_32k_int4.json",
}
payloads_32k = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in paths_32k.items()
}
reference_ids = payloads_32k["offloaded_exact"]["target_token_ids"]
for name, payload in payloads_32k.items():
    if payload["target_token_ids"] != reference_ids:
        raise RuntimeError(f"32K target token mismatch for {name}")
    rows_32k.append(
        {
            "method": name,
            "ppl": float(payload["ppl"]),
            "quality_vs_offloaded_exact_percent": (
                100.0
                * float(payloads_32k["offloaded_exact"]["ppl"])
                / float(payload["ppl"])
            ),
            "prefill_seconds_including_query": float(
                payload["prefill_seconds"]
            ),
            "conversion_seconds": float(
                payload["cache_conversion_seconds"]
            ),
            "online_seconds": float(
                payload["synchronized_model_forward_seconds"]
            ),
            "gpu_kv_ratio": float(
                payload["hierarchical_over_final_length_full_kv"]
            ),
            "peak_gpu_allocated_bytes": int(
                payload["process_peak_gpu_allocated_during_prefill_conversion"]
            ),
        }
    )

resident = json.loads(
    (run_root / "mixed_a_w2_128k_int4.json").read_text(encoding="utf-8")
)
offloaded = json.loads(
    (offloaded_root / "mixed_a_w2_qkfixed4421sampled.json").read_text(
        encoding="utf-8"
    )
)
if resident["target_token_ids"] != offloaded["target_token_ids"]:
    raise RuntimeError("128K transient/offloaded target token mismatch")
case_rows = json.loads(
    (reference / "mixed_a_w2" / "case_summary.json").read_text(
        encoding="utf-8"
    )
)
full = next(row for row in case_rows if row["method"] == "full_attention")
with (
    offloaded_root / "summary" / "per_case.csv"
).open(encoding="utf-8", newline="") as handle:
    physical_rows = list(csv.DictReader(handle))
matched = next(
    row
    for row in physical_rows
    if row["case"] == "mixed_a_w2"
    and row["variant"] == "fixed4421_sampled_compact"
)
full_ppl = float(matched["full_ppl"])
full_fixed = float(full["dense_prompt_seconds"])
full_online = float(full["sparse_decode_seconds"])
token_count = len(resident["token_nll"])
resident_ppl = math.exp(
    sum(float(value) for value in resident["token_nll"]) / token_count
)
resident_fixed = float(resident["prefill_seconds"]) + float(
    resident["cache_conversion_seconds"]
)
resident_online = float(resident["synchronized_model_forward_seconds"])
online_saving = (full_online - resident_online) / token_count
break_even = (
    max(0.0, resident_fixed - full_fixed) / online_saving
    if online_saving > 0.0
    else None
)
summary = {
    "schema": "quantized_offloaded_prefill_probe_v1",
    "paired_32k": rows_32k,
    "transient_int4_128k": {
        "full_ppl": full_ppl,
        "transient_ppl": resident_ppl,
        "quality_retention_percent": 100.0 * full_ppl / resident_ppl,
        "quality_vs_offloaded_exact_percent": (
            100.0 * float(offloaded["ppl"]) / resident_ppl
        ),
        "full_fixed_seconds": full_fixed,
        "transient_prefill_seconds_including_query": float(
            resident["prefill_seconds"]
        ),
        "transient_conversion_seconds": float(
            resident["cache_conversion_seconds"]
        ),
        "transient_fixed_seconds": resident_fixed,
        "offloaded_fixed_seconds": float(offloaded["prefill_seconds"])
        + float(offloaded["cache_conversion_seconds"]),
        "full_online_seconds": full_online,
        "transient_online_seconds": resident_online,
        "online_speedup": full_online / resident_online,
        "total_speedup_at_256_tokens": (
            (full_fixed + full_online) / (resident_fixed + resident_online)
        ),
        "break_even_tokens": break_even,
        "gpu_kv_ratio": float(
            resident["hierarchical_over_final_length_full_kv"]
        ),
        "peak_gpu_allocated_bytes": int(
            resident["process_peak_gpu_allocated_during_prefill_conversion"]
        ),
    },
}
(run_root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
