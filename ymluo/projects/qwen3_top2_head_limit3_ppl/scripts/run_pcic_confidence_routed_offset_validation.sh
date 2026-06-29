#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
OFFSETS=${OFFSETS:-"8192 16384 24576 32768"}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
mkdir -p outputs/logs

run_one() {
  local gpu=$1
  local dataset=$2
  local offset=$3
  local text_path=$4
  local combos=$5
  local output_dir="outputs/server_pcic_r3_${dataset}_off${offset}_b4_confroute_s8_seed64_allm01_pairm005_slack03_eager"
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
    --rescue_strategy none --combo_select_policy risk_memory_confidence_routed --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 --sentinel_tokens 8 --sentinel_loss_slack 0.03 \
    --sentinel_all_min_margin 0.1 --sentinel_pairwise_min_margin 0.05 \
    > "outputs/logs/server_pcic_r3_${dataset}_off${offset}_b4_confroute_s8_seed64_allm01_pairm005_slack03_eager.log" 2>&1
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
    "min_loss": "minloss_posgate",
    "risk_memory": "riskmemory_monogate",
    "rm_sentinel": "riskmemory_sentinel_s8_seed64_slack03",
    "sentall_conf": "riskmemory_sentinelall_s8_seed64_slack03_margin01",
    "confidence_routed": "confroute_s8_seed64_allm01_pairm005_slack03",
}
values = {method: [] for method in methods}

print("| dataset | offset | min_loss | risk_memory | rm_sentinel | sentall_conf | confidence_routed | routed combos | routes |")
print("|---|---:|---:|---:|---:|---:|---:|---|---|")
for dataset in ["war", "monte"]:
    label = "War" if dataset == "war" else "Monte"
    for offset in offsets:
        avgs = {}
        routed_combos = ""
        routes = []
        for method, suffix in methods.items():
            path = pathlib.Path("outputs") / f"server_pcic_r3_{dataset}_off{offset}_b4_{suffix}_eager" / "pcic_r_blockwise_results.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = list(csv.DictReader(path.open()))
            evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
            avgs[method] = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
            values[method].append(avgs[method])
            if method == "confidence_routed":
                routed_combos = ";".join(row["combo"] for row in evals)
                for row in evals:
                    try:
                        rule = json.loads(row.get("rescue_rule") or "{}")
                    except Exception:
                        rule = {}
                    if rule.get("kind") == "risk_memory_confidence_routed":
                        routes.append(
                            f"b{row['block']}:{rule.get('selected_combo')}:{rule.get('sentinel_route')}"
                            f" bm={float(rule.get('sentinel_best_margin', 0.0)):.3f}"
                            f" pd={float(rule.get('sentinel_pairwise_delta_loss', 0.0)):.3f}"
                        )
        print(
            f"| {label} | {offset} | {avgs['min_loss']:.6f} | {avgs['risk_memory']:.6f} | "
            f"{avgs['rm_sentinel']:.6f} | {avgs['sentall_conf']:.6f} | {avgs['confidence_routed']:.6f} | "
            f"{routed_combos} | {'; '.join(routes)} |"
        )

print()
print("| method | mean | worst | wins |")
print("|---|---:|---:|---:|")
for method in methods:
    wins = 0
    for index in range(len(values[method])):
        best = min(values[name][index] for name in methods)
        if abs(values[method][index] - best) < 1e-9:
            wins += 1
    print(f"| {method} | {sum(values[method]) / len(values[method]):.6f} | {max(values[method]):.6f} | {wins} |")
PY
