#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
TEXT=${TEXT:-/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt}
COMBOS=${COMBOS:-"0,6 0,7 0,13 7,6 2,0 2,7 2,0,7,12 7,13"}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p outputs/logs

run_one() {
  local gpu=$1
  local combo=$2
  local setting=$3
  local blocks=$4
  local eval_tokens=$5
  local combo_tag=${combo//,/_}
  local name="server_pcic_hardtopic_static_${setting}_${combo_tag}_eager"
  if [[ -f "outputs/${name}/pcic_r_blockwise_results.csv" ]]; then
    echo "[skip] $name"
    return 0
  fi
  echo "[start] $name gpu=$gpu combo=$combo"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/run_pcic_rescue_blockwise_local.py \
    --model_name_or_path "$MODEL" \
    --text_path "$TEXT" \
    --output_dir "outputs/${name}" \
    --prefill_tokens 2048 \
    --num_blocks "$blocks" \
    --calibration_tokens 16 \
    --eval_tokens_per_block "$eval_tokens" \
    --dtype float16 \
    --device cuda:0 \
    --attn_implementation eager \
    --recent_tokens 512 \
    --landmark_stride 64 \
    --combos "$combo" \
    --rescue_strategy none \
    --combo_select_policy min_loss \
    > "outputs/logs/${name}.log" 2>&1
  echo "[done] $name"
}

gpu=0
for combo in $COMBOS; do
  run_one "$gpu" "$combo" b4_eval64 4 64 &
  gpu=$(((gpu + 1) % 8))
done
wait

gpu=0
for combo in $COMBOS; do
  run_one "$gpu" "$combo" b4_eval128 4 128 &
  gpu=$(((gpu + 1) % 8))
done
wait

"$PY" - <<'PY'
import csv
import pathlib

combos = "0,6 0,7 0,13 7,6 2,0 2,7 2,0,7,12 7,13".split()
settings = ["b4_eval64", "b4_eval128"]

print("| setting | combo | avg_delta_ppl | method/base |")
print("|---|---|---:|---:|")
for setting in settings:
    rows_out = []
    for combo in combos:
        combo_tag = combo.replace(",", "_")
        path = pathlib.Path("outputs") / f"server_pcic_hardtopic_static_{setting}_{combo_tag}_eager" / "pcic_r_blockwise_results.csv"
        rows = list(csv.DictReader(path.open()))
        evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
        avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
        method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
        baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
        rows_out.append((avg, combo, method_seconds / max(baseline_seconds, 1e-9)))
    for avg, combo, ratio in sorted(rows_out):
        print(f"| {setting} | {combo} | {avg:.6f} | {ratio:.3f} |")

print()
print("| setting | oracle_combo | oracle_delta |")
print("|---|---|---:|")
for setting in settings:
    best = None
    for combo in combos:
        combo_tag = combo.replace(",", "_")
        path = pathlib.Path("outputs") / f"server_pcic_hardtopic_static_{setting}_{combo_tag}_eager" / "pcic_r_blockwise_results.csv"
        rows = list(csv.DictReader(path.open()))
        evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
        avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
        if best is None or avg < best[0]:
            best = (avg, combo)
    print(f"| {setting} | {best[1]} | {best[0]:.6f} |")
PY
