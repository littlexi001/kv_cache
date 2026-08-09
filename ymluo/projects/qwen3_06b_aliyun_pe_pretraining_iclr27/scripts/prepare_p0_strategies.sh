#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMON=(
  --model-root "${MODEL_ROOT}"
  --start-layer-fraction 0.5
  --frequency-pairs 16
  --alpha-min 0.25
  --alpha-max 1.0
  --remote-min 8192
  --remote-max 131072
  --remote-points 33
  --content-phase-probes 32
  --local-max 2048
  --local-points 17
  --temperature 0.08
  --steps 2000
  --learning-rate 0.03
  --seed 20260808
)

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/optimize_phase_profile.py" \
  "${COMMON[@]}" \
  --name optimized_phase_complementary \
  --local-preservation-weight 0.0 \
  --output "${PROJECT_ROOT}/configs/strategies/optimized_phase_complementary.json"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/optimize_phase_profile.py" \
  "${COMMON[@]}" \
  --name optimized_phase_complementary_local \
  --local-preservation-weight 0.5 \
  --output "${PROJECT_ROOT}/configs/strategies/optimized_phase_complementary_local.json"

echo "prepared both optimized phase profiles from ${MODEL_ROOT}"
