#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

OUT=outputs/server_pcic_war_b2_horizongate_top2_batchedgate_smoke_seed64_eager
if [[ ! -f "$OUT/pcic_r_blockwise_results.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path data/war_and_peace_pg2600.txt \
    --output_dir "$OUT" \
    --prefill_tokens 2048 --num_blocks 2 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '7,6;0,13;0,7;0,6' \
    --rescue_strategy none \
    --combo_select_policy risk_memory_horizon_gate \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 \
    --sentinel_tokens 64 \
    --sentinel_cascade_initial_tokens 32 \
    --sentinel_cascade_accept_margin 0.01 \
    --sentinel_cascade_extend_topk 2 \
    --sentinel_batched_candidates true \
    --sentinel_loss_slack 0.0 \
    --sentinel_all_min_margin 0.0 \
    --horizon_gate_min_gain 0.0 \
    --horizon_gate_min_ratio 0.0 \
    --horizon_gate_uncertainty_floor 0.005 \
    > outputs/logs/server_pcic_war_b2_horizongate_top2_batchedgate_smoke_seed64_eager.log 2>&1
fi

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    ("serial_top2", pathlib.Path("outputs/server_pcic_war_b2_horizongate_top2_timingv2_seed64_eager/pcic_r_blockwise_results.csv")),
    ("batched_top2", pathlib.Path("outputs/server_pcic_war_b2_horizongate_top2_batchedgate_smoke_seed64_eager/pcic_r_blockwise_results.csv")),
]
print("| run | avg_delta_ppl | selected/base | serial_total/base | gate_s | combos | batched |")
print("|---|---:|---:|---:|---:|---|---:|")
for name, path in cases:
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    selected_ratio = sum(float(row["seconds"]) for row in evals) / sum(float(row["baseline_seconds"]) for row in evals)
    total_ratio = sum(float(row.get("method_seconds") or row["seconds"]) for row in evals) / sum(float(row["baseline_seconds"]) for row in evals)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in evals)
    batched = 0
    for row in evals:
        rule = json.loads(row.get("rescue_rule") or "{}")
        batched = max(batched, int(rule.get("sentinel_batched_candidates", 0) or 0))
    print(
        f"| {name} | {avg:.6f} | {selected_ratio:.3f} | {total_ratio:.3f} | "
        f"{gate:.3f} | {';'.join(row['combo'] for row in evals)} | {batched} |"
    )
PY
