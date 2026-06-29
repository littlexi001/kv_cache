#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

run_one() {
  local gpu=$1
  local dataset=$2
  local eval_tokens=$3
  local text_path=$4
  local combos=$5
  local output_dir="outputs/server_pcic_r3_${dataset}_b4_eval${eval_tokens}_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"
  if [[ -f "$output_dir/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $dataset eval$eval_tokens"
    return 0
  fi
  echo "[start] $dataset eval$eval_tokens gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --text_path "$text_path" \
    --output_dir "$output_dir" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block "$eval_tokens" \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos "$combos" \
    --rescue_strategy none --combo_select_policy risk_memory_confidence_fast --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 --sentinel_tokens 8 --sentinel_loss_slack 0.03 \
    --sentinel_all_min_margin 0.1 --sentinel_pairwise_min_margin 0.05 \
    --confidence_fast_all_min_delta_loss -0.05 \
    > "outputs/logs/server_pcic_r3_${dataset}_b4_eval${eval_tokens}_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager.log" 2>&1
  echo "[done] $dataset eval$eval_tokens"
}

run_one 0 war 256 data/war_and_peace_pg2600.txt '7,6;0,6;0,7;0,13' &
run_one 1 monte 256 data/count_monte_cristo_pg1184.txt '2,0;2,7;2,0,7,12;7,13' &
run_one 2 war 512 data/war_and_peace_pg2600.txt '7,6;0,6;0,7;0,13' &
run_one 3 monte 512 data/count_monte_cristo_pg1184.txt '2,0;2,7;2,0,7,12;7,13' &
wait

"$PY" - <<'PY'
import csv
import json
import pathlib

runs = [
    ("War eval128", "outputs/server_pcic_r3_war_b4_eval128_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("Monte eval128", "outputs/server_pcic_r3_monte_b4_eval128_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("War eval256", "outputs/server_pcic_r3_war_b4_eval256_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("Monte eval256", "outputs/server_pcic_r3_monte_b4_eval256_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("War eval512", "outputs/server_pcic_r3_war_b4_eval512_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("Monte eval512", "outputs/server_pcic_r3_monte_b4_eval512_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"),
]

print("| run | blocks | avg_delta_ppl | method/base | combos | triggered_blocks |")
print("|---|---:|---:|---:|---|---|")
for label, directory in runs:
    path = pathlib.Path(directory) / "pcic_r_blockwise_results.csv"
    if not path.exists():
        print(f"| {label} | missing | | | | `{path}` |")
        continue
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg_delta_ppl = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
    triggered = []
    for row in evals:
        try:
            rule = json.loads(row.get("rescue_rule") or "{}")
        except Exception:
            rule = {}
        if int(rule.get("triggered") or 0):
            route = rule.get("sentinel_route") or rule.get("fast_route") or "triggered"
            triggered.append(f"b{row['block']}:{route}")
    print(
        f"| {label} | {len(evals)} | {avg_delta_ppl:.6f} | "
        f"{method_seconds / max(baseline_seconds, 1e-9):.3f} | "
        f"{';'.join(row['combo'] for row in evals)} | {'; '.join(triggered) or '-'} |"
    )
PY
