#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
REFERENCE=$ROOT/results/20260727_groupwise_int4_prefill_32k_paired_4gpu
RUN_ROOT=$ROOT/results/20260727_direct_quantized_key_32k_paired_4gpu
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
  --prefill_cache_mode quantized_offloaded_exact
  --prefill_quantization_bits 4
  --prefill_quantization_group_size 16
  --prefill_conversion_source transient_quantized_key
  --dtype float16
  --device cuda
  --device_map auto
)

run_topic() {
  local gpu="$1"
  local topic="$2"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u \
    src/run_hierarchical_physical_cache_ppl_20260715.py \
    --output "$RUN_ROOT/${topic}_direct.json" \
    --topic "$topic" \
    "${common[@]}" \
    >"$LOG_ROOT/${topic}_direct.log" 2>&1
}

topics=(sports medicine computer religion)
gpus=(0 2 3 7)
pids=()
for index in 0 1 2 3; do
  run_topic "${gpus[$index]}" "${topics[$index]}" &
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

"$PYTHON" - "$REFERENCE" "$RUN_ROOT" "${topics[@]}" <<'PY'
import json
import math
import random
import statistics
import sys
from pathlib import Path

reference_root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
topics = sys.argv[3:]
cases = []
for topic in topics:
    exact = json.loads(
        (reference_root / f"{topic}_exact.json").read_text(
            encoding="utf-8"
        )
    )
    direct = json.loads(
        (run_root / f"{topic}_direct.json").read_text(encoding="utf-8")
    )
    if exact["target_token_ids"] != direct["target_token_ids"]:
        raise RuntimeError(f"target token mismatch for {topic}")
    exact_nll = [float(value) for value in exact["token_nll"]]
    direct_nll = [float(value) for value in direct["token_nll"]]
    cases.append(
        {
            "topic": topic,
            "exact_nll": exact_nll,
            "direct_nll": direct_nll,
            "exact_ppl": math.exp(statistics.fmean(exact_nll)),
            "direct_ppl": math.exp(statistics.fmean(direct_nll)),
            "quality_percent": 100.0
            * math.exp(
                statistics.fmean(exact_nll)
                - statistics.fmean(direct_nll)
            ),
            "direct_prefill_seconds": float(direct["prefill_seconds"]),
            "direct_conversion_seconds": float(
                direct["cache_conversion_seconds"]
            ),
            "direct_fixed_seconds": float(
                direct["prefill_plus_conversion_seconds"]
            ),
            "direct_online_seconds": float(
                direct["synchronized_model_forward_seconds"]
            ),
            "direct_peak_gib": float(
                direct[
                    "process_peak_gpu_allocated_during_prefill_conversion"
                ]
            )
            / (1024.0**3),
        }
    )

all_exact = [value for case in cases for value in case["exact_nll"]]
all_direct = [value for case in cases for value in case["direct_nll"]]
exact_ppl = math.exp(statistics.fmean(all_exact))
direct_ppl = math.exp(statistics.fmean(all_direct))
rng = random.Random(20260727)
replicates = []
for _ in range(10000):
    sampled_exact = []
    sampled_direct = []
    for _ in cases:
        case = rng.choice(cases)
        token_count = len(case["exact_nll"])
        for _ in range(token_count):
            token_index = rng.randrange(token_count)
            sampled_exact.append(case["exact_nll"][token_index])
            sampled_direct.append(case["direct_nll"][token_index])
    replicates.append(
        100.0
        * math.exp(
            statistics.fmean(sampled_exact)
            - statistics.fmean(sampled_direct)
        )
    )
replicates.sort()
summary = {
    "schema": "direct_quantized_key_32k_paired_v1",
    "history_tokens": 32000,
    "paired_tokens": len(all_exact),
    "topics": topics,
    "exact_ppl": exact_ppl,
    "direct_ppl": direct_ppl,
    "quality_retention_percent": 100.0 * exact_ppl / direct_ppl,
    "quality_retention_paired_bootstrap_95_ci": [
        replicates[int(0.025 * len(replicates))],
        replicates[int(0.975 * len(replicates))],
    ],
    "mean_direct_prefill_seconds": statistics.fmean(
        case["direct_prefill_seconds"] for case in cases
    ),
    "mean_direct_conversion_seconds": statistics.fmean(
        case["direct_conversion_seconds"] for case in cases
    ),
    "mean_direct_fixed_seconds": statistics.fmean(
        case["direct_fixed_seconds"] for case in cases
    ),
    "mean_direct_online_seconds": statistics.fmean(
        case["direct_online_seconds"] for case in cases
    ),
    "mean_direct_peak_gib": statistics.fmean(
        case["direct_peak_gib"] for case in cases
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
