#!/bin/bash
# ==============================================================================
# 先跑 baseline 5000 步，再续训 CRS 从 3000 → 5000 步
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

PYTHON=/Users/bytedance/.workbuddy/binaries/python/versions/3.13.12/bin/python3

BATCH_SIZE=32
SEQ_LEN=256
GRAD_ACCUM=4
LR=3e-4
MAX_STEPS=5000
SAVE_EVERY=200
LOG_EVERY=25
NUM_WORKERS=4

echo "========================================"
echo "  Phase 1: Baseline — 5000 steps (fresh)"
echo "  Phase 2: CRS — resume 3000 → 5000"
echo "========================================"
echo "  Batch:   ${BATCH_SIZE} × ${SEQ_LEN} seqlen"
echo "  GradAccum: ${GRAD_ACCUM}"
echo "  LR:      ${LR}"
echo "========================================"

# ============================================================================
# Phase 1: Baseline
# ============================================================================
echo ""
echo "########################################"
echo "#  PHASE 1: Baseline — 5000 steps"
echo "########################################"
echo ""

${PYTHON} workbuddy_scripts/small_lm_crs.py \
    --mode baseline \
    --batch_size ${BATCH_SIZE} \
    --max_seq_len ${SEQ_LEN} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --max_steps ${MAX_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --num_workers ${NUM_WORKERS}

echo ""
echo "Baseline finished (exit code $?)."

# ============================================================================
# Phase 2: Resume CRS from step 3000
# ============================================================================
echo ""
echo "########################################"
echo "#  PHASE 2: CRS — resume 3000 → 5000"
echo "########################################"
echo ""

${PYTHON} workbuddy_scripts/small_lm_crs.py \
    --mode crs \
    --alpha 0.3 \
    --batch_size ${BATCH_SIZE} \
    --max_seq_len ${SEQ_LEN} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --max_steps ${MAX_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --num_workers ${NUM_WORKERS} \
    --resume_from outputs/small_lm_crs/crs_alpha0_3/checkpoints/step_3000.pt

echo ""
echo "========================================"
echo "  ALL DONE"
echo "========================================"
echo ""
echo "  Baseline: outputs/small_lm_crs/baseline/"
echo "  CRS:      outputs/small_lm_crs/crs_alpha0_3/"
