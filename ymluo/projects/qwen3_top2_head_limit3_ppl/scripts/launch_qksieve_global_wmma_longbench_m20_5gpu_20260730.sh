#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
TEMPLATE="${TEMPLATE:-$ROOT/results/20260729_qksieve_frozen_template_frontier/templates/global32_3domain_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260730_qksieve_global_wmma_longbench_m20_5gpu}"
LOG_ROOT="$RUN_ROOT/logs"
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=full_kv,qksieve_global_qkbalanced_qmse_wmma_sampled

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"
test -f "$TEMPLATE"

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,4}"
if [[ "${#gpus[@]}" -ne 5 ]]; then
  echo "QKSIEVE_GPUS must contain exactly five GPU ids" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-6]$ ]]; then
    echo "QKSIEVE_GPUS is restricted to physical GPUs 0-6; got $gpu" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[$gpu]+x}" ]]; then
    echo "QKSIEVE_GPUS contains duplicate GPU id $gpu" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

common_args=(
  --model_name_or_path "$MODEL"
  --longbench_data_dir "$DATA"
  --tasks "$TASKS"
  --methods "$METHODS"
  --max_prompt_tokens 7500
  --prompt_truncation_mode official_middle
  --official_query_tail_tokens 8
  --max_context_tokens 0
  --prefill_chunk_tokens 2048
  --prompt_wrapper qwen3
  --qk_metric_query_shrinkage 0.75
  --packed_qmse_template_in "$TEMPLATE"
  --sampled_quantile_sample_count 256
  --dtype float16
  --device cuda
  --device_map auto
)

CUDA_VISIBLE_DEVICES=${gpus[0]} "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  "${common_args[@]}" \
  --output_dir "$RUN_ROOT/smoke" \
  --tasks narrativeqa \
  --max_samples_per_task 1 \
  --num_shards 1 \
  --shard_index 0 \
  --max_new_tokens_override 8 \
  >"$LOG_ROOT/smoke.log" 2>&1

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 2, len(rows)
by_method = {row["method"]: row for row in rows}
full = by_method["full_kv"]
qksieve = by_method["qksieve_global_qkbalanced_qmse_wmma_sampled"]
assert full["executed_path"] == "full_kv"
assert qksieve["executed_path"] == qksieve["method"]
assert qksieve["configured_score_mode"].endswith(
    "qfused_gqa4_wmma_kappend_unbiased_packed_direct"
)
assert float(qksieve["packed_qmse_fixed_template_active"]) == 1.0
assert float(qksieve["packed_qmse_fused_query_prepare_requested"]) == 1.0
assert float(qksieve["packed_qmse_fused_query_prepare_executed"]) == 1.0
assert float(qksieve["packed_qmse_allocation_frozen_before_query"]) == 1.0
assert float(qksieve["packed_qmse_sample_count"]) == 256.0
assert float(qksieve["sampled_quantile_fallback"]) == 0.0
assert float(qksieve["configured_index_bits_per_token"]) == 240.0
assert 0.0 < float(qksieve["selected_history_count_mean"]) <= 1280.0
assert int(qksieve["suffix_tokens"]) == 8
print("global-template WMMA sampled-quantile LongBench smoke passed")
PY

pids=()
for shard in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    "${common_args[@]}" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --max_samples_per_task 20 \
    --num_shards 5 \
    --shard_index "$shard" \
    --max_new_tokens_override 0 \
    >"$LOG_ROOT/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more shards failed; valid CSV rows were preserved" >&2
  exit 1
fi

"$PYTHON" src/summarize_paired_longbench_20260728.py \
  --run_root "$RUN_ROOT" \
  --methods "$METHODS" \
  --reference_method full_kv \
  --expected_pairs 320 \
  --expected_tasks 16 \
  --bootstrap_resamples 10000 \
  --output "$RUN_ROOT/paired_summary.json" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$ROOT" "$RUN_ROOT" "$MODEL" "$TEMPLATE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
template = Path(sys.argv[4])
source_paths = [
    root / "src/run_sample_calibrated_longbench_20260717.py",
    root / "src/run_head_top2_targeted_ppl_20260714.py",
    root / "src/run_critical_position_budget_probe_20260715.py",
    root / "src/mixedblock_spectral_cuda_20260729.py",
    root / "src/qksieve_query_cuda_20260728.py",
]
manifest = {
    "model_path": sys.argv[3],
    "template_path": str(template),
    "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
    "protocol": {
        "prompt_truncation": "official_middle",
        "prompt_limit": 7500,
        "query_tail_tokens": 8,
        "candidate_schedule": "min(N,1280,max(256,ceil(0.06*N)))",
        "sampled_quantile_samples": 256,
        "fallback": False,
        "rerank": False,
        "recent_or_sink_reservation": False,
    },
    "qksieve": {
        "transform": "model-level frozen QK-balanced",
        "allocation": "model-level frozen qMSE mixed bit",
        "score_mode": (
            "pca_hierarchical_autoqmsetotal15z_qkmetric_"
            "qfused_gqa4_wmma_kappend_unbiased_packed_direct"
        ),
        "index_bits_per_token_per_kv_head_contract": 240,
        "exact_kv": "GPU-resident FP16",
    },
    "source_sha256": {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_paths
    },
}
(run_root / "method_contract.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

touch "$RUN_ROOT/ALL_COMPLETE"
