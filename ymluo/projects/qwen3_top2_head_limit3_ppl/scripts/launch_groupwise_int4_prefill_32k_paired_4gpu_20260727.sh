#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
RUN_ROOT=$ROOT/results/20260727_groupwise_int4_prefill_32k_paired_4gpu
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
  --history_tokens 32000
  --eval_tokens 64
  --query_tokens 256
  --window_index 0
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

run_topic() {
  local gpu="$1"
  local topic="$2"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$RUN_ROOT/${topic}_exact.json" \
    --topic "$topic" \
    --prefill_cache_mode offloaded_exact \
    "${common[@]}" \
    >"$LOG_ROOT/${topic}_exact.log" 2>&1

  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$RUN_ROOT/${topic}_group16.json" \
    --topic "$topic" \
    --prefill_cache_mode quantized_offloaded_exact \
    --prefill_quantization_bits 4 \
    --prefill_quantization_group_size 16 \
    "${common[@]}" \
    >"$LOG_ROOT/${topic}_group16.log" 2>&1
}

topics=(sports medicine computer religion)
pids=()
for index in 0 1 2 3; do
  run_topic "$index" "${topics[$index]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

"$PYTHON" - "$RUN_ROOT" "${topics[@]}" <<'PY'
import json
import math
import random
import statistics
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
topics = sys.argv[2:]
cases = []
for topic in topics:
    exact = json.loads(
        (run_root / f"{topic}_exact.json").read_text(encoding="utf-8")
    )
    groupwise = json.loads(
        (run_root / f"{topic}_group16.json").read_text(encoding="utf-8")
    )
    if exact["target_token_ids"] != groupwise["target_token_ids"]:
        raise RuntimeError(f"target token mismatch for {topic}")
    exact_nll = [float(value) for value in exact["token_nll"]]
    group_nll = [float(value) for value in groupwise["token_nll"]]
    if len(exact_nll) != len(group_nll):
        raise RuntimeError(f"token count mismatch for {topic}")
    cases.append(
        {
            "topic": topic,
            "exact_nll": exact_nll,
            "group_nll": group_nll,
            "exact_ppl": math.exp(statistics.fmean(exact_nll)),
            "group16_ppl": math.exp(statistics.fmean(group_nll)),
            "quality_percent": (
                100.0
                * math.exp(
                    statistics.fmean(exact_nll)
                    - statistics.fmean(group_nll)
                )
            ),
            "exact_fixed_seconds": float(
                exact["prefill_plus_conversion_seconds"]
            ),
            "group16_fixed_seconds": float(
                groupwise["prefill_plus_conversion_seconds"]
            ),
            "exact_peak_gib": float(
                exact["process_peak_gpu_allocated_during_prefill_conversion"]
            )
            / (1024.0**3),
            "group16_peak_gib": float(
                groupwise[
                    "process_peak_gpu_allocated_during_prefill_conversion"
                ]
            )
            / (1024.0**3),
        }
    )

all_exact = [value for case in cases for value in case["exact_nll"]]
all_group = [value for case in cases for value in case["group_nll"]]
pooled_exact_ppl = math.exp(statistics.fmean(all_exact))
pooled_group_ppl = math.exp(statistics.fmean(all_group))

rng = random.Random(20260727)
replicates = []
for _ in range(10000):
    sampled_exact = []
    sampled_group = []
    for _ in cases:
        case = rng.choice(cases)
        token_count = len(case["exact_nll"])
        for _ in range(token_count):
            token_index = rng.randrange(token_count)
            sampled_exact.append(case["exact_nll"][token_index])
            sampled_group.append(case["group_nll"][token_index])
    replicates.append(
        100.0
        * math.exp(
            statistics.fmean(sampled_exact)
            - statistics.fmean(sampled_group)
        )
    )
replicates.sort()
lower = replicates[int(0.025 * len(replicates))]
upper = replicates[int(0.975 * len(replicates))]

summary = {
    "schema": "groupwise_int4_prefill_32k_paired_v1",
    "history_tokens": 32000,
    "paired_tokens": len(all_exact),
    "topics": topics,
    "exact_ppl": pooled_exact_ppl,
    "group16_ppl": pooled_group_ppl,
    "quality_retention_percent": (
        100.0 * pooled_exact_ppl / pooled_group_ppl
    ),
    "quality_retention_paired_bootstrap_95_ci": [lower, upper],
    "mean_exact_fixed_seconds": statistics.fmean(
        case["exact_fixed_seconds"] for case in cases
    ),
    "mean_group16_fixed_seconds": statistics.fmean(
        case["group16_fixed_seconds"] for case in cases
    ),
    "fixed_stage_speedup": (
        statistics.fmean(case["exact_fixed_seconds"] for case in cases)
        / statistics.fmean(
            case["group16_fixed_seconds"] for case in cases
        )
    ),
    "mean_exact_peak_gib": statistics.fmean(
        case["exact_peak_gib"] for case in cases
    ),
    "mean_group16_peak_gib": statistics.fmean(
        case["group16_peak_gib"] for case in cases
    ),
    "per_topic": cases,
}
(run_root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
