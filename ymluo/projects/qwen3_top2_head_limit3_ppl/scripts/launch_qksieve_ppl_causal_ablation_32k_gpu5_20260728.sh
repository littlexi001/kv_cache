#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
RUN_ROOT=$ROOT/results/20260728_qksieve_ppl_causal_ablation_32k_gpu5
GPU="${QKSIEVE_GPU:-5}"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

declare -a CASES=(
  "qksieve|pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk|240"
  "keypca_uniform1|pca_hierarchical_fixed11111111_packed_fulltopk|256"
  "qkbalanced_uniform1|pca_hierarchical_fixed11111111_qkmetric_packed_fulltopk|256"
  "random_uniform1|pca_hierarchical_fixed11111111_random_packed_fulltopk|256"
  "keypca_keymse|pca_hierarchical_autokeytotal15z_packed_fulltopk|240"
  "qkbalanced_keymse|pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk|240"
  "fier_rtn1_g32|fier_rtn1_g32_packed_fulltopk|256"
)

for spec in "${CASES[@]}"; do
  IFS='|' read -r tag score_mode index_bits <<<"$spec"
  out_dir="$RUN_ROOT/$tag"
  if [[ -f "$out_dir/case_summary.json" ]]; then
    echo "[skip] $tag: case_summary.json exists"
    continue
  fi

  echo "[run] $tag score_mode=$score_mode index_bits=$index_bits"
  "$PYTHON" -u src/run_direct_countcap_denseprompt_ppl_20260725.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$out_dir" \
    --topics sports,medicine \
    --window_indices 0,1 \
    --methods full_attention,direct_countcap \
    --history_tokens 32000 \
    --eval_tokens 128 \
    --window_stride_tokens 32512 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 48 \
    --sample_count 256 \
    --candidate_overfetch 1.0 \
    --protect_recent_tokens 0 \
    --direct_score_mode "$score_mode" \
    --qk_metric_query_shrinkage 0.75 \
    --prefill_chunk_tokens 2048 \
    --cache_mode auto \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --collect_logit_stability \
    >"$RUN_ROOT/logs/$tag.log" 2>&1
done

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
case_meta = {
    "qksieve": {
        "component": "QK-balanced coordinates + Q-aware MSE bit allocation",
        "index_bits_per_token_per_kv_head": 240,
    },
    "keypca_uniform1": {
        "component": "Key-PCA coordinates + uniform 1-bit allocation",
        "index_bits_per_token_per_kv_head": 256,
    },
    "qkbalanced_uniform1": {
        "component": "QK-balanced coordinates + uniform 1-bit allocation",
        "index_bits_per_token_per_kv_head": 256,
    },
    "random_uniform1": {
        "component": "random coordinates + uniform 1-bit allocation",
        "index_bits_per_token_per_kv_head": 256,
    },
    "keypca_keymse": {
        "component": "Key-PCA coordinates + Key-MSE bit allocation",
        "index_bits_per_token_per_kv_head": 240,
    },
    "qkbalanced_keymse": {
        "component": "QK-balanced coordinates + Key-MSE bit allocation",
        "index_bits_per_token_per_kv_head": 240,
    },
    "fier_rtn1_g32": {
        "component": "FIER RTN-1, sequence group size 32",
        "index_bits_per_token_per_kv_head": 256,
    },
}

def weighted_mean(rows, field):
    valid = [r for r in rows if r.get(field) is not None]
    if not valid:
        return None
    denom = sum(int(r["tokens"]) for r in valid)
    return sum(float(r[field]) * int(r["tokens"]) for r in valid) / denom

summary = {
    "scope": "32K Qwen3-4B sports+medicine, two held-out windows per topic",
    "purpose": "causal mechanism/PPL diagnostic; not the final multi-model PPL table",
    "shared_attention_budget": "min(N,1280,max(256,ceil(0.06*N)))",
    "shared_exact_sparse_attention": True,
    "rerank": False,
    "rows": [],
}

for tag, meta in case_meta.items():
    source = run_root / tag / "case_summary.json"
    if not source.exists():
        raise FileNotFoundError(source)
    records = json.loads(source.read_text(encoding="utf-8"))
    full = [r for r in records if r["method"] == "full_attention"]
    sparse = [r for r in records if r["method"] == "direct_countcap"]
    if len(full) != 4 or len(sparse) != 4:
        raise RuntimeError(f"{tag}: expected four strict pairs, got {len(full)}/{len(sparse)}")

    full_nll = weighted_mean(full, "nll")
    sparse_nll = weighted_mean(sparse, "nll")
    full_ppl = math.exp(full_nll)
    sparse_ppl = math.exp(sparse_nll)
    full_step = weighted_mean(full, "steady_sparse_seconds_per_step")
    sparse_step = weighted_mean(sparse, "steady_sparse_seconds_per_step")
    summary["rows"].append({
        "tag": tag,
        **meta,
        "full_ppl": full_ppl,
        "sparse_ppl": sparse_ppl,
        "quality_retention": full_ppl / sparse_ppl,
        "delta_nll": sparse_nll - full_nll,
        "top1_agreement": weighted_mean(sparse, "top1_agreement"),
        "kl_full_to_sparse": weighted_mean(sparse, "kl_full_to_sparse_mean"),
        "js_divergence": weighted_mean(sparse, "js_divergence_mean"),
        "attention_token_ratio": weighted_mean(sparse, "actual_attention_tokens_mean") / 32000.0,
        "packed_index_ratio_of_full_kv": weighted_mean(sparse, "packed_index_ratio_of_full_kv"),
        "steady_decode_speedup": full_step / sparse_step,
        "fixed_index_overhead_seconds": weighted_mean(sparse, "fixed_sparse_overhead_seconds"),
    })

(run_root / "ablation_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

touch "$RUN_ROOT/ALL_COMPLETE"
