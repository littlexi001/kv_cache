#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
OFFSETS=${OFFSETS:-"8192 16384 24576 32768"}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

run_one() {
  local gpu=$1
  local dataset=$2
  local offset=$3
  local text_path=$4
  local combos=$5
  local output_dir="outputs/server_pcic_r3_${dataset}_off${offset}_b4_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager"
  if [[ -f "$output_dir/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $dataset off$offset"
    return 0
  fi
  echo "[start] $dataset off$offset gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --text_path "$text_path" \
    --output_dir "$output_dir" \
    --start_token_offset "$offset" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos "$combos" \
    --rescue_strategy none --combo_select_policy risk_memory_confidence_lazy --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 --sentinel_tokens 8 --sentinel_loss_slack 0.03 \
    --sentinel_all_min_margin 0.1 --sentinel_pairwise_min_margin 0.05 \
    --confidence_fast_all_min_delta_loss -0.05 \
    --confidence_lazy_pairwise_min_delta_loss -0.025 \
    --confidence_lazy_pairwise_max_calib_gap 0.08 \
    --confidence_lazy_pairwise_max_memory_delta_loss 0.08 \
    > "outputs/logs/server_pcic_r3_${dataset}_off${offset}_b4_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager.log" 2>&1
  echo "[done] $dataset off$offset"
}

gpu=0
for offset in $OFFSETS; do
  run_one "$gpu" war "$offset" data/war_and_peace_pg2600.txt '7,6;0,6;0,7;0,13' &
  gpu=$(((gpu + 1) % 8))
  run_one "$gpu" monte "$offset" data/count_monte_cristo_pg1184.txt '2,0;2,7;2,0,7,12;7,13' &
  gpu=$(((gpu + 1) % 8))
done

wait

"$PY" - <<'PY'
import csv
import json
import pathlib

offsets = ["8192", "16384", "24576", "32768"]
methods = {
    "conffast_s8": "conffast_s8_seed64_allm01_pairm005_slack03_delta005",
    "conflazy": "conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008",
}
values = {method: [] for method in methods}
speed = {method: [] for method in methods}

def summarize(path):
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
    skipped = triggered = 0
    routes = []
    for row in evals:
        try:
            rule = json.loads(row.get("rescue_rule") or "{}")
        except Exception:
            rule = {}
        triggered += int(rule.get("triggered") or 0)
        if rule.get("fast_route") == "lazy_skip_pairwise_memory_anchor":
            skipped += 1
        routes.append(f"b{row['block']}:{rule.get('fast_route')}:{rule.get('sentinel_route')}")
    return avg, method_seconds / max(baseline_seconds, 1e-9), skipped, triggered, ";".join(row["combo"] for row in evals), "; ".join(routes)

print("| dataset | offset | conffast_delta | conflazy_delta | conffast_ratio | conflazy_ratio | lazy_skipped/triggered | lazy_combos |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for dataset in ["war", "monte"]:
    label = "War" if dataset == "war" else "Monte"
    for offset in offsets:
        row_data = {}
        for method, suffix in methods.items():
            path = pathlib.Path("outputs") / f"server_pcic_r3_{dataset}_off{offset}_b4_{suffix}_eager" / "pcic_r_blockwise_results.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            row_data[method] = summarize(path)
            values[method].append(row_data[method][0])
            speed[method].append(row_data[method][1])
        lazy = row_data["conflazy"]
        print(
            f"| {label} | {offset} | {row_data['conffast_s8'][0]:.6f} | {lazy[0]:.6f} | "
            f"{row_data['conffast_s8'][1]:.3f} | {lazy[1]:.3f} | {lazy[2]}/{lazy[3]} | {lazy[4]} |"
        )

print()
print("| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins_vs_fast |")
print("|---|---:|---:|---:|---:|")
for method in methods:
    wins = 0
    for index, value in enumerate(values[method]):
        best = min(values[name][index] for name in methods)
        if abs(value - best) < 1e-9:
            wins += 1
    print(
        f"| {method} | {sum(values[method]) / len(values[method]):.6f} | "
        f"{max(values[method]):.6f} | {sum(speed[method]) / len(speed[method]):.3f} | {wins} |"
    )
PY
