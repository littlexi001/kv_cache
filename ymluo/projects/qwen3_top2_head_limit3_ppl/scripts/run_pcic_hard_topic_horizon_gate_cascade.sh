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

run_case() {
  local name=$1
  local eval_tokens=$2
  local cascade_margin=$3
  local out="outputs/${name}"
  if [[ ! -f "$out/pcic_r_blockwise_results.csv" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" src/run_pcic_rescue_blockwise_local.py \
      --model_name_or_path "$MODEL" \
      --text_path "$TEXT" \
      --output_dir "$out" \
      --prefill_tokens 2048 --num_blocks 4 --calibration_tokens 16 --eval_tokens_per_block "$eval_tokens" \
      --dtype float16 --device cuda:0 --attn_implementation eager \
      --recent_tokens 512 --landmark_stride 64 \
      --combos '0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13' \
      --rescue_strategy none \
      --combo_select_policy risk_memory_horizon_gate \
      --risk_memory_loss_slack 0.2 \
      --risk_memory_seed_tokens 64 \
      --sentinel_tokens 64 \
      --sentinel_cascade_initial_tokens 32 \
      --sentinel_cascade_accept_margin "$cascade_margin" \
      --sentinel_loss_slack 0.0 \
      --sentinel_all_min_margin 0.0 \
      --horizon_gate_min_gain 0.0 \
      --horizon_gate_min_ratio 0.0 \
      --horizon_gate_uncertainty_floor 0.005 \
      > "outputs/logs/${name}.log" 2>&1
  fi
}

run_case server_pcic_hardtopic_b4_horizongate_cascade32to64_m02_eval64_seed64_g0_r0_eager 64 0.02
run_case server_pcic_hardtopic_b4_horizongate_cascade32to64_m02_eval128_seed64_g0_r0_eager 128 0.02
run_case server_pcic_hardtopic_b4_horizongate_cascade32to64_m01_eval64_seed64_g0_r0_eager 64 0.01
run_case server_pcic_hardtopic_b4_horizongate_cascade32to64_m01_eval128_seed64_g0_r0_eager 128 0.01
run_case server_pcic_hardtopic_b4_horizongate_cascade32to64_m04_eval64_seed64_g0_r0_eager 64 0.04
run_case server_pcic_hardtopic_b4_horizongate_cascade32to64_m04_eval128_seed64_g0_r0_eager 128 0.04

"$PY" - <<'PY'
import csv
import json
import pathlib

cases = [
    ("eval64 raw_s64", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_s64_eval64_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("eval64 cascade_m01", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_cascade32to64_m01_eval64_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("eval64 cascade", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_cascade32to64_m02_eval64_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("eval64 cascade_m04", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_cascade32to64_m04_eval64_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("eval128 raw_s64", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_s64_eval128_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("eval128 cascade_m01", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_cascade32to64_m01_eval128_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("eval128 cascade", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_cascade32to64_m02_eval128_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
    ("eval128 cascade_m04", pathlib.Path("outputs/server_pcic_hardtopic_b4_horizongate_cascade32to64_m04_eval128_seed64_g0_r0_eager/pcic_r_blockwise_results.csv")),
]
print("| run | avg_delta_ppl | selected_method/base | gate_s | cascade_extended | cascade_early | combos |")
print("|---|---:|---:|---:|---:|---:|---|")
for name, path in cases:
    rows = list(csv.DictReader(path.open()))
    evals = [row for row in rows if row.get("kind") == "pcic_r_eval"]
    avg = sum(float(row["delta_ppl"]) for row in evals) / len(evals)
    ratio = sum(float(row["seconds"]) for row in evals) / sum(float(row["baseline_seconds"]) for row in evals)
    gate = sum(float(row.get("gate_seconds") or 0.0) for row in evals)
    extended = 0
    early = 0
    for row in evals:
        rule = json.loads(row.get("rescue_rule") or "{}")
        extended += int(rule.get("sentinel_cascade_extended", 0) or 0)
        early += int(rule.get("sentinel_cascade_accepted_early", 0) or 0)
    print(
        f"| {name} | {avg:.6f} | {ratio:.3f} | {gate:.3f} | "
        f"{extended} | {early} | {';'.join(row['combo'] for row in evals)} |"
    )
print()
print("reference:")
print("- hardtopic eval64 none: 0.030228; conffast_s8: 0.003316; static 0,6: -0.012719")
print("- hardtopic eval128 none: 0.009629; conffast_s8: 0.038679; static 0,6: -0.020744")
PY
