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
  --combo_select_policy min_loss
)

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

run_one 0 server_pcic_hardtopic_b8_none_minloss_eager \
  "${COMMON[@]}" \
  --output_dir outputs/server_pcic_hardtopic_b8_none_minloss_eager \
  --prefill_tokens 2048 --num_blocks 8 --calibration_tokens 16 --eval_tokens_per_block 64 &

run_one 1 server_pcic_hardtopic_b4_eval128_none_minloss_eager \
  "${COMMON[@]}" \
  --output_dir outputs/server_pcic_hardtopic_b4_eval128_none_minloss_eager \
  --prefill_tokens 2048 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block 128 &

wait

"$PY" - <<'PY'
import csv
import pathlib

runs = [
    ("hardtopic none b4", "outputs/server_pcic_hardtopic_b4_none_minloss_eager"),
    ("hardtopic none b8", "outputs/server_pcic_hardtopic_b8_none_minloss_eager"),
    ("hardtopic none eval128", "outputs/server_pcic_hardtopic_b4_eval128_none_minloss_eager"),
]

print("| run | blocks | avg_delta_ppl | method/base | combos |")
print("|---|---:|---:|---:|---|")
for label, directory in runs:
    rows = list(csv.DictReader((pathlib.Path(directory) / "pcic_r_blockwise_results.csv").open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    method_seconds = sum(float(row.get("method_seconds") or row.get("seconds") or 0.0) for row in evals)
    baseline_seconds = sum(float(row.get("baseline_seconds") or 0.0) for row in evals)
    print(
        f"| {label} | {len(evals)} | {avg:.6f} | {method_seconds / max(baseline_seconds, 1e-9):.3f} | "
        f"{';'.join(row['combo'] for row in evals)} |"
    )
PY
