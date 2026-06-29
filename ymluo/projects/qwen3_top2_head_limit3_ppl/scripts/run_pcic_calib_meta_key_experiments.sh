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
  echo "[start] $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py "$@" \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

run_one 0 server_pcic_r3_war_b8_calib_meta_eager \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir outputs/server_pcic_r3_war_b8_calib_meta_eager \
  --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
  --dtype float16 --device cuda:0 --attn_implementation eager \
  --recent_tokens 512 --landmark_stride 64 \
  --combos '7,6;0,6;0,7;0,13' \
  --rescue_strategy calib_meta_fallback --combo_select_policy min_loss \
  --block_risk_max_gap 0.2 --block_risk_positive_ratio 0.5 \
  --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
  --meta_max_gap_increase 0.0 \
  --meta_min_original_positive_ratio_if_increase 0.4 &

run_one 1 server_pcic_r3_monte_b8_calib_meta_eager \
  --text_path data/count_monte_cristo_pg1184.txt \
  --output_dir outputs/server_pcic_r3_monte_b8_calib_meta_eager \
  --prefill_tokens 4096 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 \
  --dtype float16 --device cuda:0 --attn_implementation eager \
  --recent_tokens 512 --landmark_stride 64 \
  --combos '2,0;2,7;2,0,7,12;7,13' \
  --rescue_strategy calib_meta_fallback --combo_select_policy min_loss \
  --block_risk_max_gap 0.6 --block_risk_positive_ratio 0.7 \
  --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
  --meta_max_gap_increase 0.0 \
  --meta_min_original_positive_ratio_if_increase 0.4 &

run_one 2 server_pcic_r3_war_b4_eval128_calib_meta_eager \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir outputs/server_pcic_r3_war_b4_eval128_calib_meta_eager \
  --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
  --dtype float16 --device cuda:0 --attn_implementation eager \
  --recent_tokens 512 --landmark_stride 64 \
  --combos '7,6;0,6;0,7;0,13' \
  --rescue_strategy calib_meta_fallback --combo_select_policy min_loss \
  --block_risk_max_gap 0.2 --block_risk_positive_ratio 0.5 \
  --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
  --meta_max_gap_increase 0.0 \
  --meta_min_original_positive_ratio_if_increase 0.4 &

run_one 3 server_pcic_r3_monte_b4_eval128_calib_meta_eager \
  --text_path data/count_monte_cristo_pg1184.txt \
  --output_dir outputs/server_pcic_r3_monte_b4_eval128_calib_meta_eager \
  --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 \
  --dtype float16 --device cuda:0 --attn_implementation eager \
  --recent_tokens 512 --landmark_stride 64 \
  --combos '2,0;2,7;2,0,7,12;7,13' \
  --rescue_strategy calib_meta_fallback --combo_select_policy min_loss \
  --block_risk_max_gap 0.6 --block_risk_positive_ratio 0.7 \
  --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
  --meta_max_gap_increase 0.0 \
  --meta_min_original_positive_ratio_if_increase 0.4 &

wait

"$PY" - <<'PY'
import csv
import json
import pathlib

runs = [
    ("War b8 none", "outputs/server_pcic_r3_war_b8_none_minloss_eager"),
    ("War b8 fixed R3", "outputs/server_pcic_r3_war_b8_block_fallback_minloss_eager"),
    ("War b8 sentinel", "outputs/server_pcic_r3_war_b8_sentinel_s4_reuse_filter_g03_eager"),
    ("War b8 calib-meta", "outputs/server_pcic_r3_war_b8_calib_meta_eager"),
    ("Monte b8 none", "outputs/server_pcic_r3_monte_b8_none_minloss_eager"),
    ("Monte b8 fixed R3", "outputs/server_pcic_r3_monte_b8_r3_gap06_ratio07_eager"),
    ("Monte b8 sentinel", "outputs/server_pcic_r3_monte_b8_sentinel_s4_reuse_filter_g03_eager"),
    ("Monte b8 calib-meta", "outputs/server_pcic_r3_monte_b8_calib_meta_eager"),
    ("War eval128 none", "outputs/server_pcic_r3_war_b4_eval128_none_minloss_eager"),
    ("War eval128 fixed R3", "outputs/server_pcic_r3_war_b4_eval128_r3_gap02_ratio05_eager"),
    ("War eval128 sentinel", "outputs/server_pcic_r3_war_eval128_sentinel_s4_reuse_filter_g03_eager"),
    ("War eval128 calib-meta", "outputs/server_pcic_r3_war_b4_eval128_calib_meta_eager"),
    ("Monte eval128 none", "outputs/server_pcic_r3_monte_b4_eval128_none_minloss_eager"),
    ("Monte eval128 calib-meta", "outputs/server_pcic_r3_monte_b4_eval128_calib_meta_eager"),
]
print("| run | blocks | avg_delta_loss | avg_delta_ppl | method/base | triggered | meta_accept |")
print("|---|---:|---:|---:|---:|---:|---:|")
for label, directory in runs:
    path = pathlib.Path(directory) / "pcic_r_blockwise_results.csv"
    if not path.exists():
        print(f"| {label} | missing | | | | | {path} |")
        continue
    evals = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    avg_delta_loss = sum(float(row["delta_loss"]) for row in evals) / len(evals)
    avg_delta_ppl = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
    triggered = 0
    accepted = 0
    for row in evals:
        try:
            rule = json.loads(row.get("rescue_rule") or "{}")
        except Exception:
            rule = {}
        triggered += int(rule.get("triggered", 0))
        accepted += int(rule.get("meta_accepted", 0))
    print(
        f"| {label} | {len(evals)} | {avg_delta_loss:.6f} | {avg_delta_ppl:.6f} | "
        f"{method_seconds / max(baseline_seconds, 1e-9):.4f} | {triggered} | {accepted} |"
    )
PY
