#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
OFFSETS=${OFFSETS:-"24576 32768"}
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

gpu=0
for offset in $OFFSETS; do
  run_one "$gpu" "server_pcic_r3_war_off${offset}_b4_none_eager" \
    --text_path data/war_and_peace_pg2600.txt \
    --output_dir "outputs/server_pcic_r3_war_off${offset}_b4_none_eager" \
    --start_token_offset "$offset" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '7,6;0,6;0,7;0,13' \
    --rescue_strategy none --combo_select_policy min_loss &
  gpu=$(((gpu + 1) % 8))

  run_one "$gpu" "server_pcic_r3_war_off${offset}_b4_calibmeta_posgate_eager" \
    --text_path data/war_and_peace_pg2600.txt \
    --output_dir "outputs/server_pcic_r3_war_off${offset}_b4_calibmeta_posgate_eager" \
    --start_token_offset "$offset" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '7,6;0,6;0,7;0,13' \
    --rescue_strategy calib_meta_fallback --combo_select_policy min_loss \
    --block_risk_max_gap 0.2 --block_risk_positive_ratio 0.5 \
    --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
    --meta_max_gap_increase 0.0 \
    --meta_min_original_positive_ratio_if_increase 0.4 &
  gpu=$(((gpu + 1) % 8))

  run_one "$gpu" "server_pcic_r3_monte_off${offset}_b4_none_eager" \
    --text_path data/count_monte_cristo_pg1184.txt \
    --output_dir "outputs/server_pcic_r3_monte_off${offset}_b4_none_eager" \
    --start_token_offset "$offset" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '2,0;2,7;2,0,7,12;7,13' \
    --rescue_strategy none --combo_select_policy min_loss &
  gpu=$(((gpu + 1) % 8))

  run_one "$gpu" "server_pcic_r3_monte_off${offset}_b4_calibmeta_posgate_eager" \
    --text_path data/count_monte_cristo_pg1184.txt \
    --output_dir "outputs/server_pcic_r3_monte_off${offset}_b4_calibmeta_posgate_eager" \
    --start_token_offset "$offset" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '2,0;2,7;2,0,7,12;7,13' \
    --rescue_strategy calib_meta_fallback --combo_select_policy min_loss \
    --block_risk_max_gap 0.6 --block_risk_positive_ratio 0.7 \
    --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
    --meta_max_gap_increase 0.0 \
    --meta_min_original_positive_ratio_if_increase 0.4 &
  gpu=$(((gpu + 1) % 8))
done

wait

"$PY" - <<'PY'
import csv
import json
import os
import pathlib

offsets = os.environ.get("OFFSETS", "24576 32768").split()
runs: list[tuple[str, str]] = []
for offset in offsets:
    runs.extend(
        [
            (f"War off{offset} none", f"outputs/server_pcic_r3_war_off{offset}_b4_none_eager"),
            (f"War off{offset} refined", f"outputs/server_pcic_r3_war_off{offset}_b4_calibmeta_posgate_eager"),
            (f"Monte off{offset} none", f"outputs/server_pcic_r3_monte_off{offset}_b4_none_eager"),
            (f"Monte off{offset} refined", f"outputs/server_pcic_r3_monte_off{offset}_b4_calibmeta_posgate_eager"),
        ]
    )

print("| run | blocks | avg_delta_ppl | method/base | accepted | proposal_blocks | combos |")
print("|---|---:|---:|---:|---:|---|---|")
for label, directory in runs:
    path = pathlib.Path(directory) / "pcic_r_blockwise_results.csv"
    if not path.exists():
        print(f"| {label} | missing | | | | | `{path}` |")
        continue
    evals = [row for row in csv.DictReader(path.open()) if row.get("kind") == "pcic_r_eval"]
    avg_delta_ppl = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
    accepted = 0
    proposals = []
    for row in evals:
        try:
            rule = json.loads(row.get("rescue_rule") or "{}")
        except Exception:
            rule = {}
        accepted += int(rule.get("meta_accepted", 0))
        if int(rule.get("proposal_triggered", 0)):
            proposals.append(
                f"b{row['block']}:{rule.get('original_combo')}->{rule.get('selected_combo')}"
                f"/a{rule.get('meta_accepted', 0)}"
            )
    print(
        f"| {label} | {len(evals)} | {avg_delta_ppl:.6f} | "
        f"{method_seconds / max(baseline_seconds, 1e-9):.3f} | {accepted} | "
        f"{';'.join(proposals) or '-'} | {';'.join(row['combo'] for row in evals)} |"
    )
PY
