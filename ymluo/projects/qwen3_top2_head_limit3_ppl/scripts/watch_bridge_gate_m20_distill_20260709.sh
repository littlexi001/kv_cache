#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

NO_BRIDGE="${NO_BRIDGE:-outputs/riskkv_v13_multiscale_flow_m20_20260709_bridge_ablation_m20_b512_p128/task_results.csv}"
ALL_BRIDGE="${ALL_BRIDGE:-outputs/riskkv_v16_multiscale_bridge_m20_20260709_bridge_ablation_m20_b512_p128/task_results.csv}"
V18_BROAD="${V18_BROAD:-outputs/riskkv_v18_task_bridge_m20_20260709_task_bridge_v18_m20_b512_p128/summary.csv}"
V18_QM="${V18_QM:-outputs/riskkv_v18_task_bridge_qm_m20_20260709_task_bridge_v18_qm_m20_b512_p128/summary.csv}"
OUT="${OUT:-outputs/bridge_gate_distill_m20_20260709}"

while [[ ! -f "$NO_BRIDGE" || ! -f "$ALL_BRIDGE" ]]; do
  echo "WAIT_PAIR no_bridge=$([[ -f "$NO_BRIDGE" ]] && echo yes || echo no) all_bridge=$([[ -f "$ALL_BRIDGE" ]] && echo yes || echo no) $(date -Is)"
  sleep 300
done

python3 scripts/distill_bridge_gate_from_paired_results_20260709.py \
  --no_bridge_results "$NO_BRIDGE" \
  --bridge_results "$ALL_BRIDGE" \
  --output_dir "$OUT"

python3 scripts/train_bridge_gate_from_labels_20260709.py \
  --labels_csv "$OUT/bridge_gate_labels.csv" \
  --output_dir "${OUT}_train" \
  --feature_results "$NO_BRIDGE"

{
  echo
  echo "# Available v18 m20 summaries"
  for path in "$V18_BROAD" "$V18_QM"; do
    if [[ -f "$path" ]]; then
      echo
      echo "## $path"
      cat "$path"
    else
      echo
      echo "## $path"
      echo "pending"
    fi
  done
} >> "$OUT/bridge_gate_report.md"

echo "DONE bridge gate m20 distill $(date -Is)"
cat "$OUT/bridge_gate_report.md"
