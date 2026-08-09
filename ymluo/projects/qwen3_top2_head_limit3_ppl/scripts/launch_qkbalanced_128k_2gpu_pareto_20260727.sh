#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
REFERENCE=$ROOT/results/20260727_qkmetric_qscale_128k_holdout
PREREQUISITE=$ROOT/results/20260727_qkbalanced_longbench_official_middle_5way_5gpu
RUN_ROOT=$ROOT/results/20260727_qkbalanced_128k_2gpu_pareto
LOG_ROOT=$RUN_ROOT/logs
DEVICES=${DEVICES:-0,1}
WAIT_FOR_PREREQUISITE=${WAIT_FOR_PREREQUISITE:-1}

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"
trap 'touch "$RUN_ROOT/TERMINAL"' EXIT

if [[ "$WAIT_FOR_PREREQUISITE" == "1" ]]; then
  while [[ ! -e "$PREREQUISITE/ALL_COMPLETE" ]]; do
    if ! pgrep -f \
      '^bash scripts/launch_qkbalanced_longbench_official_middle_5way_5gpu_20260727.sh$' \
      >/dev/null; then
      echo "official-middle LongBench exited without ALL_COMPLETE" >&2
      exit 1
    fi
    sleep 60
  done
fi

common=(
  --model_name_or_path "$MODEL"
  --history_tokens 128000
  --query_tokens 256
  --eval_tokens 256
  --window_stride_tokens 128512
  --index_mode qk_variable
  --qk_metric_query_shrinkage 0.75
  --variable_rate_budget 15
  --candidate_fraction 0.06
  --candidate_min_tokens 256
  --candidate_max_tokens 1280
  --attention_fraction 0.06
  --candidate_selection_mode per_head_stream
  --stream_group_size 1
  --exact_cache_fraction 0.032
  --directory_backend fused
  --prefill_cache_mode quantized_offloaded_exact
  --prefill_quantization_bits 4
  --prefill_quantization_group_size 16
  --prefill_conversion_source transient_quantized_key
  --prefill_chunk_tokens 2048
  --dtype float16
  --device cuda
  --device_map balanced
)

run_case() {
  local topic="$1"
  local window="$2"
  local variant="$3"
  local output="$RUN_ROOT/${topic}_w${window}_${variant}.json"
  local log="$LOG_ROOT/${topic}_w${window}_${variant}.log"
  local variant_args=()

  if [[ -s "$output" ]]; then
    echo "SKIP $topic w$window $variant"
    return
  fi
  case "$variant" in
    auto_fulltopk)
      variant_args+=(--retrieval_backend full_topk)
      ;;
    fixed4421_sampled)
      variant_args+=(
        --fixed_bit_allocation 4,4,2,1,0,0,0,0
        --retrieval_backend sampled_compact
        --sampled_candidate_multiplier 1.5
      )
      ;;
    *)
      echo "unknown variant: $variant" >&2
      return 2
      ;;
  esac

  echo "START $topic w$window $variant on GPUs $DEVICES"
  CUDA_VISIBLE_DEVICES="$DEVICES" \
    "$PYTHON" -u src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$output" \
    --topic "$topic" \
    --window_index "$window" \
    "${common[@]}" \
    "${variant_args[@]}" \
    >"$log" 2>&1
  echo "DONE $topic w$window $variant"
}

# A short target suffix first validates that two-card model/KV placement fits.
SMOKE=$RUN_ROOT/mixed_a_w2_auto_fulltopk_smoke.json
if [[ ! -s "$SMOKE" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICES" \
    "$PYTHON" -u src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$SMOKE" \
    --topic mixed_a \
    --window_index 2 \
    "${common[@]}" \
    --eval_tokens 16 \
    --retrieval_backend full_topk \
    >"$LOG_ROOT/mixed_a_w2_auto_fulltopk_smoke.log" 2>&1
fi

topics=(mixed_a mixed_a mixed_b mixed_b)
windows=(2 3 2 3)
for index in "${!topics[@]}"; do
  run_case "${topics[$index]}" "${windows[$index]}" auto_fulltopk
  run_case "${topics[$index]}" "${windows[$index]}" fixed4421_sampled
done

"$PYTHON" - "$RUN_ROOT" "$REFERENCE" <<'PY'
import csv
import json
import math
import statistics
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
reference = Path(sys.argv[2])
cases = [
    ("mixed_a", 2),
    ("mixed_a", 3),
    ("mixed_b", 2),
    ("mixed_b", 3),
]
variants = ("auto_fulltopk", "fixed4421_sampled")

rows = []
aggregate = {
    variant: {
        "nll": [],
        "prefill": 0.0,
        "conversion": 0.0,
        "online": 0.0,
        "gpu_ratio": [],
        "host_bytes": [],
        "peak_gpu_bytes": [],
    }
    for variant in variants
}
full_nll = []
full_fixed = 0.0
full_online = 0.0

for topic, window in cases:
    reference_rows = json.loads(
        (reference / f"{topic}_w{window}" / "case_summary.json").read_text(
            encoding="utf-8"
        )
    )
    full = next(row for row in reference_rows if row["method"] == "full_attention")
    full_nll.extend([float(full["nll"])] * int(full["tokens"]))
    full_fixed += float(full["dense_prompt_seconds"])
    full_online += float(full["sparse_decode_seconds"])

    payloads = {
        variant: json.loads(
            (run_root / f"{topic}_w{window}_{variant}.json").read_text(
                encoding="utf-8"
            )
        )
        for variant in variants
    }
    if payloads[variants[0]]["target_token_ids"] != payloads[variants[1]][
        "target_token_ids"
    ]:
        raise RuntimeError(f"target-token mismatch for {topic} w{window}")

    for variant, payload in payloads.items():
        values = aggregate[variant]
        token_nll = [float(value) for value in payload["token_nll"]]
        values["nll"].extend(token_nll)
        values["prefill"] += float(payload["prefill_seconds"])
        values["conversion"] += float(payload["cache_conversion_seconds"])
        values["online"] += float(payload["synchronized_model_forward_seconds"])
        values["gpu_ratio"].append(
            float(payload["hierarchical_over_final_length_full_kv"])
        )
        values["host_bytes"].append(float(payload["pinned_host_bytes"]))
        values["peak_gpu_bytes"].append(
            float(payload["process_peak_gpu_allocated_during_prefill_conversion"])
        )
        method_ppl = math.exp(sum(token_nll) / len(token_nll))
        rows.append(
            {
                "case": f"{topic}_w{window}",
                "variant": variant,
                "tokens": len(token_nll),
                "full_ppl": math.exp(float(full["nll"])),
                "method_ppl": method_ppl,
                "quality_retention_percent": (
                    100.0 * math.exp(float(full["nll"])) / method_ppl
                ),
                "decode_speedup": (
                    float(full["sparse_decode_seconds"])
                    / float(payload["synchronized_model_forward_seconds"])
                ),
                "gpu_kv_ratio_percent": (
                    100.0
                    * float(payload["hierarchical_over_final_length_full_kv"])
                ),
                "pinned_host_gib": float(payload["pinned_host_bytes"]) / 2**30,
                "prefill_seconds": float(payload["prefill_seconds"]),
                "conversion_seconds": float(payload["cache_conversion_seconds"]),
                "online_seconds": float(payload["synchronized_model_forward_seconds"]),
            }
        )

full_ppl = math.exp(sum(full_nll) / len(full_nll))
summary_variants = {}
for variant, values in aggregate.items():
    method_ppl = math.exp(sum(values["nll"]) / len(values["nll"]))
    sparse_fixed = values["prefill"] + values["conversion"]
    full_total = full_fixed + full_online
    sparse_total = sparse_fixed + values["online"]
    per_token_saving = (full_online - values["online"]) / len(values["nll"])
    summary_variants[variant] = {
        "tokens": len(values["nll"]),
        "ppl": method_ppl,
        "quality_retention_percent": 100.0 * full_ppl / method_ppl,
        "decode_speedup": full_online / values["online"],
        "total_speedup_at_256_tokens_per_window": full_total / sparse_total,
        "prefill_seconds_sum": values["prefill"],
        "conversion_seconds_sum": values["conversion"],
        "online_seconds_sum": values["online"],
        "break_even_generated_tokens_per_window": (
            max(0.0, sparse_fixed - full_fixed) / per_token_saving / len(cases)
            if per_token_saving > 0.0
            else None
        ),
        "mean_gpu_kv_ratio_percent": 100.0 * statistics.mean(values["gpu_ratio"]),
        "mean_pinned_host_gib": statistics.mean(values["host_bytes"]) / 2**30,
        "max_peak_gpu_allocated_gib_across_two_cards": (
            max(values["peak_gpu_bytes"]) / 2**30
        ),
    }

payload = {
    "schema": "qkbalanced_128k_2gpu_pareto_v1",
    "model": "Qwen3-4B-Instruct",
    "devices_per_run": 2,
    "history_tokens": 128000,
    "cases": len(cases),
    "tokens_per_variant": len(full_nll),
    "full": {
        "ppl": full_ppl,
        "gpu_kv_ratio_percent": 100.0,
        "fixed_seconds_sum": full_fixed,
        "online_seconds_sum": full_online,
    },
    "variants": summary_variants,
}

with (run_root / "per_case.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(run_root / "summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
echo "ALL_COMPLETE"
