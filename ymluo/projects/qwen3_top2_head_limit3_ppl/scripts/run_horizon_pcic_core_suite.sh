#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}
cd "$ROOT"

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

RUN_EXPERIMENTS=${RUN_EXPERIMENTS:-0}
RUN_SMOKE=${RUN_SMOKE:-0}

required_files=(
  "src/run_pcic_rescue_blockwise_local.py"
  "src/evaluate_qwen3_top2_head_limit3_ppl.py"
  "scripts/summarize_horizon_pcic_results.py"
  "scripts/run_pcic_hard_topic_horizon_gate_topk_cascade.sh"
  "scripts/run_pcic_warmonte_horizon_gate_topk_cascade.sh"
  "scripts/run_pcic_horizon_top2_timing_refresh.sh"
  "scripts/run_pcic_batched_candidate_smoke.sh"
  "scripts/run_pcic_war_batched_gate_smoke.sh"
  "data/war_and_peace_pg2600.txt"
  "data/count_monte_cristo_pg1184.txt"
)

for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -e "$MODEL" ]]; then
  echo "missing model path: $MODEL" >&2
  exit 1
fi

"$PY" -m py_compile \
  src/evaluate_qwen3_top2_head_limit3_ppl.py \
  src/run_pcic_rescue_blockwise_local.py \
  scripts/summarize_horizon_pcic_results.py

echo "Horizon-PCIC core files OK"
echo "RUN_EXPERIMENTS=$RUN_EXPERIMENTS RUN_SMOKE=$RUN_SMOKE HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

if [[ "$RUN_EXPERIMENTS" == "1" ]]; then
  echo "Running key Horizon-PCIC experiments if outputs are missing..."
  bash scripts/run_pcic_hard_topic_horizon_gate_topk_cascade.sh
  bash scripts/run_pcic_warmonte_horizon_gate_topk_cascade.sh
  bash scripts/run_pcic_horizon_top2_timing_refresh.sh
else
  echo "Skipping heavy experiments. Set RUN_EXPERIMENTS=1 to run guarded experiment scripts."
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "Running batched candidate smoke tests..."
  bash scripts/run_pcic_batched_candidate_smoke.sh
  bash scripts/run_pcic_war_batched_gate_smoke.sh
else
  echo "Skipping smoke tests. Set RUN_SMOKE=1 to run batched gate smoke scripts."
fi

"$PY" scripts/summarize_horizon_pcic_results.py

echo
echo "Generated:"
echo "  docs/horizon_pcic_key_results_2026_06_29.md"
echo "  docs/horizon_pcic_key_results_2026_06_29.csv"
echo
grep -n "top2 cascade" docs/horizon_pcic_key_results_2026_06_29.md || true
