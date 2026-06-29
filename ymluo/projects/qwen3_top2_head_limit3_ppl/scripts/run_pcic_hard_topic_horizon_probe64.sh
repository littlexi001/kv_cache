#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
TEXT=${TEXT:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

OUT=outputs/server_pcic_hardtopic_b4_horizonprobe64_all_seed64_slack03_margin0_eager
if [[ ! -f "$OUT/pcic_r_blockwise_results.csv" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path "$TEXT" \
    --output_dir "$OUT" \
    --prefill_tokens 2048 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 64 \
    --dtype float16 --device cuda:0 --attn_implementation eager \
    --recent_tokens 512 --landmark_stride 64 \
    --combos '0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13' \
    --rescue_strategy none \
    --combo_select_policy risk_memory_sentinel_all \
    --risk_memory_loss_slack 0.2 \
    --risk_memory_seed_tokens 64 \
    --sentinel_tokens 64 \
    --sentinel_loss_slack 0.03 \
    --sentinel_all_min_margin 0.0 \
    > outputs/logs/server_pcic_hardtopic_b4_horizonprobe64_all_seed64_slack03_margin0_eager.log 2>&1
fi

"$PY" - <<'PY'
import csv
import json
import pathlib

path = pathlib.Path("outputs/server_pcic_hardtopic_b4_horizonprobe64_all_seed64_slack03_margin0_eager/pcic_r_blockwise_results.csv")
rows = list(csv.DictReader(path.open()))
evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
print("| block | combo | delta_ppl | route | best_margin | best_loss_combo |")
print("|---:|---|---:|---|---:|---|")
for row in evals:
    rule = json.loads(row.get("rescue_rule") or "{}")
    losses = rule.get("sentinel_all_losses") or {}
    best_combo = min(losses, key=losses.get) if losses else ""
    print(
        f"| {row['block']} | {row['combo']} | {float(row['delta_ppl']):.6f} | "
        f"{rule.get('sentinel_route')} | {float(rule.get('sentinel_best_margin', 0.0)):.3f} | {best_combo} |"
    )
print()
print(
    "| avg | | "
    f"{sum(float(row['delta_ppl']) for row in evals) / len(evals):.6f} | | | "
    f"{';'.join(row['combo'] for row in evals)} |"
)
print()
print("reference:")
print("- none b4_eval64 avg_delta_ppl: 0.030228")
print("- conffast_s8 b4_eval64 avg_delta_ppl: 0.003316")
print("- static oracle 0,6 b4_eval64 avg_delta_ppl: -0.012719")
PY
