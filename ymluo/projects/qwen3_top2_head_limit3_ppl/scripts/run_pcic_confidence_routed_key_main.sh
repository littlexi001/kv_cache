#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
mkdir -p outputs/logs

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

common_args=(
  --dtype float16
  --device cuda:0
  --attn_implementation eager
  --recent_tokens 512
  --landmark_stride 64
  --rescue_strategy none
  --combo_select_policy risk_memory_confidence_routed
  --risk_memory_loss_slack 0.2
  --risk_memory_seed_tokens 64
  --sentinel_tokens 8
  --sentinel_loss_slack 0.03
  --sentinel_all_min_margin 0.1
  --sentinel_pairwise_min_margin 0.05
)

run_one 0 server_pcic_r3_war_b8_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir outputs/server_pcic_r3_war_b8_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
  --combos '7,6;0,6;0,7;0,13' \
  "${common_args[@]}" &

run_one 1 server_pcic_r3_monte_b8_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --text_path data/count_monte_cristo_pg1184.txt \
  --output_dir outputs/server_pcic_r3_monte_b8_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
  --combos '2,0;2,7;2,0,7,12;7,13' \
  "${common_args[@]}" &

run_one 2 server_pcic_r3_war_b4_eval128_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir outputs/server_pcic_r3_war_b4_eval128_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
  --combos '7,6;0,6;0,7;0,13' \
  "${common_args[@]}" &

run_one 3 server_pcic_r3_monte_b4_eval128_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --text_path data/count_monte_cristo_pg1184.txt \
  --output_dir outputs/server_pcic_r3_monte_b4_eval128_confroute_s8_seed64_allm01_pairm005_slack03_eager \
  --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
  --combos '2,0;2,7;2,0,7,12;7,13' \
  "${common_args[@]}" &

wait

"$PY" - <<'PY'
import csv
import json
import pathlib

runs = [
    ("War b8 none", "outputs/server_pcic_r3_war_b8_none_minloss_eager"),
    ("War b8 calib-meta", "outputs/server_pcic_r3_war_b8_calib_meta_eager"),
    ("War b8 confroute", "outputs/server_pcic_r3_war_b8_confroute_s8_seed64_allm01_pairm005_slack03_eager"),
    ("Monte b8 none", "outputs/server_pcic_r3_monte_b8_none_minloss_eager"),
    ("Monte b8 calib-meta", "outputs/server_pcic_r3_monte_b8_calib_meta_eager"),
    ("Monte b8 confroute", "outputs/server_pcic_r3_monte_b8_confroute_s8_seed64_allm01_pairm005_slack03_eager"),
    ("War eval128 none", "outputs/server_pcic_r3_war_b4_eval128_none_minloss_eager"),
    ("War eval128 calib-meta", "outputs/server_pcic_r3_war_b4_eval128_calib_meta_eager"),
    ("War eval128 confroute", "outputs/server_pcic_r3_war_b4_eval128_confroute_s8_seed64_allm01_pairm005_slack03_eager"),
    ("Monte eval128 none", "outputs/server_pcic_r3_monte_b4_eval128_none_minloss_eager"),
    ("Monte eval128 calib-meta", "outputs/server_pcic_r3_monte_b4_eval128_calib_meta_eager"),
    ("Monte eval128 confroute", "outputs/server_pcic_r3_monte_b4_eval128_confroute_s8_seed64_allm01_pairm005_slack03_eager"),
]

print("| run | blocks | avg_delta_ppl | method/base | combos | routed_blocks |")
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
    routed = []
    for row in evals:
        try:
            rule = json.loads(row.get("rescue_rule") or "{}")
        except Exception:
            rule = {}
        if rule.get("kind") == "risk_memory_confidence_routed":
            routed.append(
                f"b{row['block']}:{rule.get('selected_combo')}:{rule.get('sentinel_route')}"
                f" bm={float(rule.get('sentinel_best_margin', 0.0)):.3f}"
                f" pd={float(rule.get('sentinel_pairwise_delta_loss', 0.0)):.3f}"
            )
    print(
        f"| {label} | {len(evals)} | {avg_delta_ppl:.6f} | "
        f"{method_seconds / max(baseline_seconds, 1e-9):.3f} | "
        f"{';'.join(row['combo'] for row in evals)} | {'; '.join(routed) or '-'} |"
    )
PY
