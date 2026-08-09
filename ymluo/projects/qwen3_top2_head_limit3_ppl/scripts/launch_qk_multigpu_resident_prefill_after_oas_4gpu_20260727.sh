#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
PREREQUISITE=$ROOT/results/20260727_qkbalanced_qscale_oas_dev_m20_5gpu
REFERENCE=$ROOT/results/20260727_qkmetric_qscale_128k_holdout
OFFLOADED=$ROOT/results/20260727_qk_variable_physical_128k_4gpu
RUN_ROOT=$ROOT/results/20260727_qk_multigpu_resident_prefill_4gpu
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"
trap 'touch "$RUN_ROOT/TERMINAL"' EXIT

while [[ ! -e "$PREREQUISITE/ALL_COMPLETE" ]]; do
  if ! pgrep -f \
    '^bash scripts/launch_qkbalanced_qscale_oas_dev_m20_5gpu_20260727.sh$' \
    >/dev/null; then
    echo "OAS prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

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
  --prefill_cache_mode dynamic
  --prefill_chunk_tokens 4096
  --dtype float16
  --device cuda
  --device_map balanced
)

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -u \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --output "$RUN_ROOT/mixed_a_w2_32k_smoke.json" \
  --history_tokens 32000 \
  --eval_tokens 32 \
  "${common[@]}" \
  >"$LOG_ROOT/32k_smoke.log" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -u \
  src/run_hierarchical_physical_cache_ppl_20260715.py \
  --output "$RUN_ROOT/mixed_a_w2_128k.json" \
  --history_tokens 128000 \
  --eval_tokens 256 \
  "${common[@]}" \
  >"$LOG_ROOT/128k.log" 2>&1

"$PYTHON" - "$RUN_ROOT" "$REFERENCE" "$OFFLOADED" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
reference = Path(sys.argv[2])
offloaded_root = Path(sys.argv[3])
resident = json.loads(
    (run_root / "mixed_a_w2_128k.json").read_text(encoding="utf-8")
)
offloaded = json.loads(
    (offloaded_root / "mixed_a_w2_qkfixed4421sampled.json").read_text(
        encoding="utf-8"
    )
)
if resident["target_token_ids"] != offloaded["target_token_ids"]:
    raise RuntimeError("resident/offloaded target token mismatch")

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
payload = {
    "schema": "qk_multigpu_resident_prefill_v1",
    "history_tokens": 128000,
    "eval_tokens": token_count,
    "full_ppl": full_ppl,
    "resident_ppl": resident_ppl,
    "resident_quality_retention_percent": 100.0 * full_ppl / resident_ppl,
    "offloaded_ppl": float(offloaded["ppl"]),
    "full_fixed_seconds": full_fixed,
    "resident_prefill_seconds_including_query": float(
        resident["prefill_seconds"]
    ),
    "resident_conversion_seconds": float(
        resident["cache_conversion_seconds"]
    ),
    "resident_fixed_seconds": resident_fixed,
    "offloaded_fixed_seconds": float(offloaded["prefill_seconds"])
    + float(offloaded["cache_conversion_seconds"]),
    "full_online_seconds": full_online,
    "resident_online_seconds": resident_online,
    "resident_online_speedup": full_online / resident_online,
    "resident_total_speedup_at_eval_length": (
        (full_fixed + full_online) / (resident_fixed + resident_online)
    ),
    "resident_break_even_tokens": break_even,
    "resident_gpu_kv_ratio": float(
        resident["hierarchical_over_final_length_full_kv"]
    ),
    "resident_peak_gpu_allocated_by_device": resident[
        "process_peak_gpu_allocated_by_device"
    ],
}
(run_root / "summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
