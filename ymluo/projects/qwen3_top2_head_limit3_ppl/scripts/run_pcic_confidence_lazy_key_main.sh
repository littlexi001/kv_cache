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
  --combo_select_policy risk_memory_confidence_lazy
  --risk_memory_loss_slack 0.2
  --risk_memory_seed_tokens 64
  --sentinel_tokens 8
  --sentinel_loss_slack 0.03
  --sentinel_all_min_margin 0.1
  --sentinel_pairwise_min_margin 0.05
  --confidence_fast_all_min_delta_loss -0.05
  --confidence_lazy_pairwise_min_delta_loss -0.025
  --confidence_lazy_pairwise_max_calib_gap 0.08
  --confidence_lazy_pairwise_max_memory_delta_loss 0.08
)

run_one 0 server_pcic_r3_war_b8_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir outputs/server_pcic_r3_war_b8_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
  --combos '7,6;0,6;0,7;0,13' \
  "${common_args[@]}" &

run_one 1 server_pcic_r3_monte_b8_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --text_path data/count_monte_cristo_pg1184.txt \
  --output_dir outputs/server_pcic_r3_monte_b8_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
  --combos '2,0;2,7;2,0,7,12;7,13' \
  "${common_args[@]}" &

run_one 2 server_pcic_r3_war_b4_eval128_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir outputs/server_pcic_r3_war_b4_eval128_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
  --combos '7,6;0,6;0,7;0,13' \
  "${common_args[@]}" &

run_one 3 server_pcic_r3_monte_b4_eval128_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --text_path data/count_monte_cristo_pg1184.txt \
  --output_dir outputs/server_pcic_r3_monte_b4_eval128_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager \
  --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
  --combos '2,0;2,7;2,0,7,12;7,13' \
  "${common_args[@]}" &

wait

"$PY" - <<'PY'
import csv
import json
import pathlib

runs = [
    ("War b8", "war_b8"),
    ("Monte b8", "monte_b8"),
    ("War eval128", "war_b4_eval128"),
    ("Monte eval128", "monte_b4_eval128"),
]

print("| run | conffast_delta | conflazy_delta | conffast_ratio | conflazy_ratio | lazy_skipped/triggered | lazy_combos |")
print("|---|---:|---:|---:|---:|---:|---|")
for label, key in runs:
    fast = pathlib.Path(f"outputs/server_pcic_r3_{key}_conffast_s8_seed64_allm01_pairm005_slack03_delta005_eager/pcic_r_blockwise_results.csv")
    lazy = pathlib.Path(f"outputs/server_pcic_r3_{key}_conflazy_s8_seed64_allm01_pairm005_slack03_delta005_lazym0025_gap008_mem008_eager/pcic_r_blockwise_results.csv")
    out = []
    for path in (fast, lazy):
        rows = list(csv.DictReader(path.open()))
        evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
        avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
        method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
        baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
        skipped = triggered = 0
        for row in evals:
            rule = json.loads(row.get("rescue_rule") or "{}")
            triggered += int(rule.get("triggered") or 0)
            skipped += int(rule.get("fast_route") == "lazy_skip_pairwise_memory_anchor")
        out.append((avg, method_seconds / max(baseline_seconds, 1e-9), skipped, triggered, ";".join(row["combo"] for row in evals)))
    print(
        f"| {label} | {out[0][0]:.6f} | {out[1][0]:.6f} | {out[0][1]:.3f} | {out[1][1]:.3f} | "
        f"{out[1][2]}/{out[1][3]} | {out[1][4]} |"
    )
PY
