#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
TEXT=${TEXT:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

COMMON=(
  --model_name_or_path "$MODEL"
  --text_path "$TEXT"
  --prefill_tokens 2048
  --num_blocks 4
  --calibration_tokens 16
  --eval_tokens_per_block 64
  --dtype float16
  --device cuda:0
  --attn_implementation eager
  --recent_tokens 512
  --landmark_stride 64
  --combos '0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
  --rescue_strategy none
)

run_one() {
  local gpu=$1
  local name=$2
  shift 2
  if [[ -f "outputs/${name}/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py "$@" \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

run_one 0 server_pcic_hardtopic_b4_none_minloss_eager \
  "${COMMON[@]}" \
  --output_dir outputs/server_pcic_hardtopic_b4_none_minloss_eager \
  --combo_select_policy min_loss &

run_one 1 server_pcic_hardtopic_b4_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager \
  "${COMMON[@]}" \
  --output_dir outputs/server_pcic_hardtopic_b4_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager \
  --combo_select_policy risk_memory_confidence_fast \
  --risk_memory_loss_slack 0.2 \
  --risk_memory_seed_tokens 64 \
  --sentinel_tokens 8 \
  --sentinel_loss_slack 0.03 \
  --sentinel_all_min_margin 0.1 \
  --sentinel_pairwise_min_margin 0.05 \
  --confidence_fast_all_min_delta_loss -0.05 &

wait

"$PY" - <<'PY'
import csv
import json
import pathlib

runs = [
    ("hardtopic none", "outputs/server_pcic_hardtopic_b4_none_minloss_eager"),
    ("hardtopic conffast", "outputs/server_pcic_hardtopic_b4_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"),
]

print("| run | blocks | avg_delta_ppl | method/base | combos | routes |")
print("|---|---:|---:|---:|---|---|")
for label, directory in runs:
    path = pathlib.Path(directory) / "pcic_r_blockwise_results.csv"
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg_delta = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
    routes = []
    for row in evals:
        try:
            rule = json.loads(row.get("rescue_rule") or "{}")
        except Exception:
            rule = {}
        route = rule.get("sentinel_route") or rule.get("fast_route") or rule.get("kind") or "-"
        routes.append(f"b{row['block']}:{route}")
    print(
        f"| {label} | {len(evals)} | {avg_delta:.6f} | "
        f"{method_seconds / max(baseline_seconds, 1e-9):.3f} | "
        f"{';'.join(row['combo'] for row in evals)} | {'; '.join(routes)} |"
    )
PY
