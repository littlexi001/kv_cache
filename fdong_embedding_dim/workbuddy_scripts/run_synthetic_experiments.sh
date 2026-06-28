#!/bin/bash
# ==============================================================================
# Synthetic CRS experiment: baseline + 3 alpha values
# Model: 1 TransformerBlock, d=128, 4 heads, 268K params
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=/Users/bytedance/.workbuddy/binaries/python/versions/3.13.12/bin/python3
SCRIPT=workbuddy_scripts/synthetic_crs_experiment.py
STEPS=2000
BATCH=32
P=8
LR=3e-4

echo "========================================="
echo "  Synthetic CRS Experiment Suite"
echo "  ${STEPS} steps, p=${P}, lr=${LR}"
echo "========================================="

# --- Baseline ---
echo ""
echo "### BASELINE ###"
${PYTHON} ${SCRIPT} --mode baseline --max_steps ${STEPS} --batch_size ${BATCH} --lr ${LR}

# --- CRS with different alphas ---
for ALPHA in 0.1 0.3 0.5; do
    echo ""
    echo "### CRS alpha=${ALPHA} p=${P} ###"
    ${PYTHON} ${SCRIPT} --mode crs --alpha ${ALPHA} --p ${P} \
        --max_steps ${STEPS} --batch_size ${BATCH} --lr ${LR}
done

echo ""
echo "========================================="
echo "  All done. Results in outputs/synthetic_crs/"
echo "========================================="
