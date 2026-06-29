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

COMMON=(
  --model_name_or_path "$MODEL"
  --text_path "$TEXT"
  --dtype float16
  --device cuda:0
  --attn_implementation eager
  --recent_tokens 512
  --landmark_stride 64
  --combos '0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13'
  --rescue_strategy none
  --combo_select_policy risk_memory_confidence_fast
  --risk_memory_loss_slack 0.2
  --sentinel_tokens 8
  --sentinel_loss_slack 0.03
  --sentinel_all_min_margin 0.1
  --sentinel_pairwise_min_margin 0.05
  --confidence_fast_all_min_delta_loss -0.05
)

run_one() {
  local gpu=$1
  local seed=$2
  local setting=$3
  local blocks=$4
  local eval_tokens=$5
  local name="server_pcic_hardtopic_${setting}_conffast_s8_seed${seed}_allm01_pairm005_slack03_delta005_eager"
  if [[ -f "outputs/${name}/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py \
    "${COMMON[@]}" \
    --output_dir "outputs/${name}" \
    --prefill_tokens 2048 \
    --num_blocks "$blocks" \
    --calibration_tokens 16 \
    --eval_tokens_per_block "$eval_tokens" \
    --risk_memory_seed_tokens "$seed" \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

gpu=0
for seed in 0 16 64; do
  run_one "$gpu" "$seed" b4_eval64 4 64 &
  gpu=$(((gpu + 1) % 8))
  run_one "$gpu" "$seed" b4_eval128 4 128 &
  gpu=$(((gpu + 1) % 8))
done
wait

"$PY" - <<'PY'
import csv
import json
import pathlib

print("| setting | seed | avg_delta_ppl | method/base | combos | triggered | routes |")
print("|---|---:|---:|---:|---|---:|---|")
for setting in ["b4_eval64", "b4_eval128"]:
    for seed in [0, 16, 64]:
        path = pathlib.Path("outputs") / f"server_pcic_hardtopic_{setting}_conffast_s8_seed{seed}_allm01_pairm005_slack03_delta005_eager" / "pcic_r_blockwise_results.csv"
        rows = list(csv.DictReader(path.open()))
        evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
        avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
        method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
        baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
        triggered = 0
        routes = []
        for row in evals:
            rule = json.loads(row.get("rescue_rule") or "{}")
            triggered += int(rule.get("triggered") or 0)
            routes.append(f"b{row['block']}:{rule.get('fast_route')}:{rule.get('sentinel_route')}")
        print(
            f"| {setting} | {seed} | {avg:.6f} | {method_seconds / max(baseline_seconds, 1e-9):.3f} | "
            f"{';'.join(row['combo'] for row in evals)} | {triggered} | {'; '.join(routes)} |"
        )
PY
