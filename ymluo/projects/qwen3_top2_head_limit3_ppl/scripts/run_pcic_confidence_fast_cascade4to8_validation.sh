#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODE=${MODE:-all}
OFFSETS=${OFFSETS:-"8192 16384 24576 32768"}
CASCADE_MARGIN=${CASCADE_MARGIN:-0.15}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

tag="cascade4to8_m${CASCADE_MARGIN/./}"

common_args=(
  --dtype float16
  --device cuda:0
  --attn_implementation eager
  --recent_tokens 512
  --landmark_stride 64
  --rescue_strategy none
  --combo_select_policy risk_memory_confidence_fast
  --risk_memory_loss_slack 0.2
  --risk_memory_seed_tokens 64
  --sentinel_tokens 8
  --sentinel_loss_slack 0.03
  --sentinel_all_min_margin 0.1
  --sentinel_pairwise_min_margin 0.05
  --confidence_fast_all_min_delta_loss -0.05
  --sentinel_cascade_initial_tokens 4
  --sentinel_cascade_accept_margin "$CASCADE_MARGIN"
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

run_offsets() {
  local gpu=0
  for offset in $OFFSETS; do
    run_one "$gpu" "server_pcic_r3_war_off${offset}_b4_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
      --text_path data/war_and_peace_pg2600.txt \
      --output_dir "outputs/server_pcic_r3_war_off${offset}_b4_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
      --start_token_offset "$offset" \
      --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
      --combos '7,6;0,6;0,7;0,13' \
      "${common_args[@]}" &
    gpu=$(((gpu + 1) % 8))
    run_one "$gpu" "server_pcic_r3_monte_off${offset}_b4_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
      --text_path data/count_monte_cristo_pg1184.txt \
      --output_dir "outputs/server_pcic_r3_monte_off${offset}_b4_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
      --start_token_offset "$offset" \
      --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
      --combos '2,0;2,7;2,0,7,12;7,13' \
      "${common_args[@]}" &
    gpu=$(((gpu + 1) % 8))
  done
  wait
}

run_key() {
  run_one 0 "server_pcic_r3_war_b8_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --text_path data/war_and_peace_pg2600.txt \
    --output_dir "outputs/server_pcic_r3_war_b8_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --combos '7,6;0,6;0,7;0,13' \
    "${common_args[@]}" &

  run_one 1 "server_pcic_r3_monte_b8_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --text_path data/count_monte_cristo_pg1184.txt \
    --output_dir "outputs/server_pcic_r3_monte_b8_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --combos '2,0;2,7;2,0,7,12;7,13' \
    "${common_args[@]}" &

  run_one 2 "server_pcic_r3_war_b4_eval128_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --text_path data/war_and_peace_pg2600.txt \
    --output_dir "outputs/server_pcic_r3_war_b4_eval128_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
    --combos '7,6;0,6;0,7;0,13' \
    "${common_args[@]}" &

  run_one 3 "server_pcic_r3_monte_b4_eval128_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --text_path data/count_monte_cristo_pg1184.txt \
    --output_dir "outputs/server_pcic_r3_monte_b4_eval128_conffast_${tag}_seed64_allm01_pairm005_slack03_delta005_eager" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
    --combos '2,0;2,7;2,0,7,12;7,13' \
    "${common_args[@]}" &

  wait
}

if [[ "$MODE" == "offset" || "$MODE" == "all" ]]; then
  run_offsets
fi
if [[ "$MODE" == "key" || "$MODE" == "all" ]]; then
  run_key
fi

"$PY" - "$tag" <<'PY'
import csv
import json
import pathlib
import sys

tag = sys.argv[1]

def read_eval(directory):
    path = pathlib.Path(directory) / "pcic_r_blockwise_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg_delta = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
    early = extended = triggered = 0
    tokens = []
    for row in evals:
        try:
            rule = json.loads(row.get("rescue_rule") or "{}")
        except Exception:
            rule = {}
        triggered += int(rule.get("triggered") or 0)
        early += int(rule.get("sentinel_cascade_accepted_early") or 0)
        extended += int(rule.get("sentinel_cascade_extended") or 0)
        if "sentinel_tokens" in rule:
            tokens.append(str(rule["sentinel_tokens"]))
    return {
        "avg_delta": avg_delta,
        "ratio": method_seconds / max(baseline_seconds, 1e-9),
        "combos": ";".join(row["combo"] for row in evals),
        "triggered": triggered,
        "early": early,
        "extended": extended,
        "blocks": len(evals),
        "tokens": ",".join(tokens) or "-",
    }

offsets = ["8192", "16384", "24576", "32768"]
values = {"s8": [], "cascade": []}
speed = {"s8": [], "cascade": []}

print("## offset")
print("| dataset | offset | s8_delta | cascade_delta | s8_ratio | cascade_ratio | early/extended/triggered | tokens | cascade_combos |")
print("|---|---:|---:|---:|---:|---:|---|---|---|")
for dataset in ["war", "monte"]:
    label = "War" if dataset == "war" else "Monte"
    for offset in offsets:
        s8 = read_eval(f"outputs/server_pcic_r3_{dataset}_off{offset}_b4_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager")
        cascade = read_eval(f"outputs/server_pcic_r3_{dataset}_off{offset}_b4_conffast_{tag}_seed64_allm01_pairm005_slack03_delta005_eager")
        values["s8"].append(s8["avg_delta"])
        values["cascade"].append(cascade["avg_delta"])
        speed["s8"].append(s8["ratio"])
        speed["cascade"].append(cascade["ratio"])
        print(
            f"| {label} | {offset} | {s8['avg_delta']:.6f} | {cascade['avg_delta']:.6f} | "
            f"{s8['ratio']:.3f} | {cascade['ratio']:.3f} | "
            f"{cascade['early']}/{cascade['extended']}/{cascade['triggered']} | {cascade['tokens']} | {cascade['combos']} |"
        )

print()
print("| method | mean_delta_ppl | worst_delta_ppl | mean_method/base | wins_vs_s8 |")
print("|---|---:|---:|---:|---:|")
for method in ["s8", "cascade"]:
    wins = 0
    for index, value in enumerate(values[method]):
        best = min(values["s8"][index], values["cascade"][index])
        if abs(value - best) < 1e-9:
            wins += 1
    print(
        f"| conffast_{method} | {sum(values[method]) / len(values[method]):.6f} | "
        f"{max(values[method]):.6f} | {sum(speed[method]) / len(speed[method]):.3f} | {wins} |"
    )

key_runs = [
    ("War b8", "outputs/server_pcic_r3_war_b8_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager", f"outputs/server_pcic_r3_war_b8_conffast_{tag}_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("Monte b8", "outputs/server_pcic_r3_monte_b8_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager", f"outputs/server_pcic_r3_monte_b8_conffast_{tag}_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("War eval128", "outputs/server_pcic_r3_war_b4_eval128_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager", f"outputs/server_pcic_r3_war_b4_eval128_conffast_{tag}_seed64_allm01_pairm005_slack03_delta005_eager"),
    ("Monte eval128", "outputs/server_pcic_r3_monte_b4_eval128_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager", f"outputs/server_pcic_r3_monte_b4_eval128_conffast_{tag}_seed64_allm01_pairm005_slack03_delta005_eager"),
]

print()
print("## key")
print("| run | s8_delta | cascade_delta | s8_ratio | cascade_ratio | early/extended/triggered | tokens | cascade_combos |")
print("|---|---:|---:|---:|---:|---|---|---|")
for label, s8_dir, cascade_dir in key_runs:
    s8 = read_eval(s8_dir)
    cascade = read_eval(cascade_dir)
    print(
        f"| {label} | {s8['avg_delta']:.6f} | {cascade['avg_delta']:.6f} | "
        f"{s8['ratio']:.3f} | {cascade['ratio']:.3f} | "
        f"{cascade['early']}/{cascade['extended']}/{cascade['triggered']} | {cascade['tokens']} | {cascade['combos']} |"
    )
PY
