#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl"
ordinary="${ROOT}/results/20260801_qksieve_keyonly_native128k_same6_v25_6gpu"
alpha="${ROOT}/results/20260801_qksieve_valuesketch16_alpha_computer_medicine128k_6gpu"
mkdir -p "${ordinary}/launcher_logs" "${alpha}"

env \
  ROOT="${ROOT}" \
  RUN_ROOT="${ordinary}" \
  GPU_IDS="4,5" \
  TOPICS="mixed_b:20260836" \
  VARIANTS="qksieve_keymse_requestlocal_sampled_k1280_c32" \
  PROTECT_RECENT_TOKENS=0 \
  bash "${ROOT}/scripts/run_qksieve_native128k_valuesketch_topics_20260801.sh" \
  >"${ordinary}/launcher_logs/mixed_b.log" 2>&1

env \
  ROOT="${ROOT}" \
  RUN_ROOT="${alpha}" \
  GPU_IDS="4,5" \
  ALPHA=0.5 \
  TOPICS="sports_both:20260831" \
  VARIANT="qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280" \
  bash "${ROOT}/scripts/run_qksieve_valuesketch_alpha_pair_20260801.sh" \
  >"${alpha}/alpha050_sports_launcher.log" 2>&1

env \
  ROOT="${ROOT}" \
  GPU_IDS="4,5" \
  bash "${ROOT}/scripts/complete_qksieve_keyonly_ruler_hard64k128k_gpu0123_20260801.sh" \
  >"${ROOT}/results/20260801_qksieve_ruler_keyonly_gpu45_launcher.log" 2>&1

touch "${ROOT}/results/20260801_qksieve_mixed_alpha_sports_ruler_gpu45_complete"
