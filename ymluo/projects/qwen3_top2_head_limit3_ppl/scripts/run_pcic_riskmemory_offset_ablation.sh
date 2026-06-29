#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
OFFSETS=${OFFSETS:-"8192 16384 24576 32768"}
POLICIES=${POLICIES:-"min_loss risk_budget risk_memory"}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
mkdir -p outputs/logs

run_one() {
  local gpu=$1
  local name=$2
  shift 2
  local output_dir=""
  local previous=""
  for arg in "$@"; do
    if [[ "$previous" == "--output_dir" ]]; then
      output_dir="$arg"
      break
    fi
    previous="$arg"
  done
  if [[ -n "$output_dir" && -f "$output_dir/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py "$@" \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

policy_dir_suffix() {
  case "$1" in
    min_loss) echo "minloss_posgate" ;;
    risk_budget) echo "riskbudget_posgate" ;;
    risk_memory) echo "riskmemory_monogate" ;;
    *) echo "$1" ;;
  esac
}

gpu=0
for offset in $OFFSETS; do
  for policy in $POLICIES; do
    suffix=$(policy_dir_suffix "$policy")

    run_one "$gpu" "server_pcic_r3_war_off${offset}_b4_${suffix}_eager" \
      --text_path data/war_and_peace_pg2600.txt \
      --output_dir "outputs/server_pcic_r3_war_off${offset}_b4_${suffix}_eager" \
      --start_token_offset "$offset" \
      --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
      --dtype float16 --device cuda:0 --attn_implementation eager \
      --recent_tokens 512 --landmark_stride 64 \
      --combos '7,6;0,6;0,7;0,13' \
      --rescue_strategy calib_meta_fallback --combo_select_policy "$policy" --risk_memory_loss_slack 0.2 \
      --block_risk_max_gap 0.2 --block_risk_positive_ratio 0.5 \
      --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
      --meta_max_gap_increase 0.0 \
      --meta_min_original_positive_ratio_if_increase 0.4 &
    gpu=$(((gpu + 1) % 8))

    run_one "$gpu" "server_pcic_r3_monte_off${offset}_b4_${suffix}_eager" \
      --text_path data/count_monte_cristo_pg1184.txt \
      --output_dir "outputs/server_pcic_r3_monte_off${offset}_b4_${suffix}_eager" \
      --start_token_offset "$offset" \
      --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
      --dtype float16 --device cuda:0 --attn_implementation eager \
      --recent_tokens 512 --landmark_stride 64 \
      --combos '2,0;2,7;2,0,7,12;7,13' \
      --rescue_strategy calib_meta_fallback --combo_select_policy "$policy" --risk_memory_loss_slack 0.2 \
      --block_risk_max_gap 0.6 --block_risk_positive_ratio 0.7 \
      --meta_min_original_max_gap 0.5 --meta_selected_loss_slack 0.1 \
      --meta_max_gap_increase 0.0 \
      --meta_min_original_positive_ratio_if_increase 0.4 &
    gpu=$(((gpu + 1) % 8))
  done
done

wait

"$PY" - <<'PY'
import csv
import json
import os
import pathlib

offsets = os.environ.get("OFFSETS", "8192 16384 24576 32768").split()
policies = os.environ.get("POLICIES", "min_loss risk_budget risk_memory").split()
suffixes = {
    "min_loss": "minloss_posgate",
    "risk_budget": "riskbudget_posgate",
    "risk_memory": "riskmemory_monogate",
}

print("| dataset | offset | policy | avg_delta_ppl | method/base | accepted | proposals | combos |")
print("|---|---:|---|---:|---:|---:|---|---|")
for offset in offsets:
    for dataset in ["war", "monte"]:
        label = "War" if dataset == "war" else "Monte"
        for policy in policies:
            suffix = suffixes[policy]
            directory = pathlib.Path("outputs") / f"server_pcic_r3_{dataset}_off{offset}_b4_{suffix}_eager"
            path = directory / "pcic_r_blockwise_results.csv"
            if not path.exists():
                print(f"| {label} | {offset} | {policy} | missing | | | | `{path}` |")
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
                f"| {label} | {offset} | {policy} | {avg_delta_ppl:.6f} | "
                f"{method_seconds / max(baseline_seconds, 1e-9):.3f} | {accepted} | "
                f"{';'.join(proposals) or '-'} | {';'.join(row['combo'] for row in evals)} |"
            )
PY
