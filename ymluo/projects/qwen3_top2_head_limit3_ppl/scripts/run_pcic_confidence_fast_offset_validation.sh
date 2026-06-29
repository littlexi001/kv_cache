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
  local output_dir="outputs/server_pcic_r3_${dataset}_off${offset}_b4_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager"
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
    --rescue_strategy none --combo_select_policy risk_memory_confidence_fast --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 --sentinel_tokens 8 --sentinel_loss_slack 0.03 \
    --sentinel_all_min_margin 0.1 --sentinel_pairwise_min_margin 0.05 \
    --confidence_fast_all_min_delta_loss -0.05 \
    > "outputs/logs/server_pcic_r3_${dataset}_off${offset}_b4_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager.log" 2>&1
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
    "confidence_routed": "confroute_s8_seed64_allm01_pairm005_slack03",
    "confidence_fast": "conffast_s8_seed64_allm01_pairm005_slack03_delta005",
}
values = {method: [] for method in methods}
speed = {method: [] for method in methods}

print("| dataset | offset | confidence_routed | confidence_fast | fast method/base | fast combos | fast routes |")
print("|---|---:|---:|---:|---:|---|---|")
for dataset in ["war", "monte"]:
    label = "War" if dataset == "war" else "Monte"
    for offset in offsets:
        avgs = {}
        fast_ratio = 0.0
        fast_combos = ""
        fast_routes = []
        for method, suffix in methods.items():
            path = pathlib.Path("outputs") / f"server_pcic_r3_{dataset}_off{offset}_b4_{suffix}_eager" / "pcic_r_blockwise_results.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = list(csv.DictReader(path.open()))
            evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
            avgs[method] = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
            values[method].append(avgs[method])
            method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
            baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
            ratio = method_seconds / max(baseline_seconds, 1e-9)
            speed[method].append(ratio)
            if method == "confidence_fast":
                fast_ratio = ratio
                fast_combos = ";".join(row["combo"] for row in evals)
                for row in evals:
                    try:
                        rule = json.loads(row.get("rescue_rule") or "{}")
                    except Exception:
                        rule = {}
                    route = rule.get("sentinel_route") or rule.get("fast_route") or "memory_no_sentinel"
                    triggered = int(rule.get("triggered") or 0)
                    fast_routes.append(
                        f"b{row['block']}:{rule.get('selected_combo')}:{route}:t{triggered}"
                    )
        print(
            f"| {label} | {offset} | {avgs['confidence_routed']:.6f} | {avgs['confidence_fast']:.6f} | "
            f"{fast_ratio:.3f} | {fast_combos} | {'; '.join(fast_routes)} |"
        )

print()
print("| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins |")
print("|---|---:|---:|---:|---:|")
for method in methods:
    wins = 0
    for index in range(len(values[method])):
        best = min(values[name][index] for name in methods)
        if abs(values[method][index] - best) < 1e-9:
            wins += 1
    print(
        f"| {method} | {sum(values[method]) / len(values[method]):.6f} | "
        f"{max(values[method]):.6f} | {sum(speed[method]) / len(speed[method]):.3f} | {wins} |"
    )
PY
