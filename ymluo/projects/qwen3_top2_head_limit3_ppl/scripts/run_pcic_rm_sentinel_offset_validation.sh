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

gpu=0
for offset in $OFFSETS; do
  run_one "$gpu" "server_pcic_r3_war_off${offset}_b4_riskmemory_sentinel_s8_seed64_slack03_eager" \
    --text_path data/war_and_peace_pg2600.txt \
    --output_dir "outputs/server_pcic_r3_war_off${offset}_b4_riskmemory_sentinel_s8_seed64_slack03_eager" \
    --start_token_offset "$offset" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '7,6;0,6;0,7;0,13' \
    --rescue_strategy none --combo_select_policy risk_memory_sentinel --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 --sentinel_tokens 8 --sentinel_loss_slack 0.03 &
  gpu=$(((gpu + 1) % 8))

  run_one "$gpu" "server_pcic_r3_monte_off${offset}_b4_riskmemory_sentinel_s8_seed64_slack03_eager" \
    --text_path data/count_monte_cristo_pg1184.txt \
    --output_dir "outputs/server_pcic_r3_monte_off${offset}_b4_riskmemory_sentinel_s8_seed64_slack03_eager" \
    --start_token_offset "$offset" \
    --prefill_tokens 4096 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '2,0;2,7;2,0,7,12;7,13' \
    --rescue_strategy none --combo_select_policy risk_memory_sentinel --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 --sentinel_tokens 8 --sentinel_loss_slack 0.03 &
  gpu=$(((gpu + 1) % 8))
done

wait

"$PY" - <<'PY'
import csv
import json
import os
import pathlib

offsets = os.environ.get("OFFSETS", "8192 16384 24576 32768").split()

print("| dataset | offset | method | avg_delta_ppl | method/base | sentinel_blocks | memory_selected | combos |")
print("|---|---:|---|---:|---:|---|---:|---|")
for offset in offsets:
    for dataset in ["war", "monte"]:
        label = "War" if dataset == "war" else "Monte"
        runs = [
            ("min_loss", f"outputs/server_pcic_r3_{dataset}_off{offset}_b4_minloss_posgate_eager"),
            (
                "risk_memory",
                f"outputs/server_pcic_r3_{dataset}_off{offset}_b4_riskmemory_monogate_eager",
            ),
            (
                "rm_sentinel",
                f"outputs/server_pcic_r3_{dataset}_off{offset}_b4_riskmemory_sentinel_s8_seed64_slack03_eager",
            ),
        ]
        for method, directory in runs:
            path = pathlib.Path(directory) / "pcic_r_blockwise_results.csv"
            if not path.exists():
                print(f"| {label} | {offset} | {method} | missing | | | | `{path}` |")
                continue
            rows = list(csv.DictReader(path.open()))
            evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
            avg_delta_ppl = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
            method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
            baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
            blocks = []
            memory_selected = 0
            for row in evals:
                try:
                    rule = json.loads(row.get("rescue_rule") or "{}")
                except Exception:
                    rule = {}
                if rule.get("kind") == "risk_memory_sentinel" and int(rule.get("triggered", 0)):
                    memory_selected += int(rule.get("sentinel_memory_selected", 0))
                    blocks.append(
                        f"b{row['block']}:{rule.get('min_loss_combo')}|{rule.get('memory_combo')}"
                        f"->{rule.get('selected_combo')} md={float(rule.get('sentinel_memory_delta_loss', 0.0)):.4f}"
                    )
            print(
                f"| {label} | {offset} | {method} | {avg_delta_ppl:.6f} | "
                f"{method_seconds / max(baseline_seconds, 1e-9):.3f} | "
                f"{'; '.join(blocks) or '-'} | {memory_selected} | "
                f"{';'.join(row['combo'] for row in evals)} |"
            )
PY
