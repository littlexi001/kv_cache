#!/bin/bash
# ==============================================================================
# run_5k_both.sh
#
# Sequential training: 5000 steps CRS (alpha=0.3) → 5000 steps baseline.
# Each run saves checkpoints (model + optimizer state) for exact resume.
#
# Usage:
#   chmod +x run_5k_both.sh
#   ./run_5k_both.sh
#
# To resume a partially-finished run (e.g. CRS crashed at step 3000):
#   python3 workbuddy_scripts/small_lm_crs.py \
#       --mode crs --alpha 0.3 --max_steps 5000 \
#       --resume_from outputs/small_lm_crs/crs_alpha0_3/checkpoints/step_3000.pt
#
# Or resume baseline:
#   python3 workbuddy_scripts/small_lm_crs.py \
#       --mode baseline --max_steps 5000 \
#       --resume_from outputs/small_lm_crs/baseline/checkpoints/step_3000.pt
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_DIR"

# ---------- Common settings ----------
BATCH_SIZE=32
SEQ_LEN=256
GRAD_ACCUM=4
LR=3e-4
MAX_STEPS=5000
SAVE_EVERY=200
LOG_EVERY=25
NUM_WORKERS=4

echo "========================================"
echo "  CRS + Baseline: 5000 steps each"
echo "========================================"
echo "  Batch:   ${BATCH_SIZE} × ${SEQ_LEN} seqlen"
echo "  GradAccum: ${GRAD_ACCUM} (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  LR:      ${LR}"
echo "  Save:    every ${SAVE_EVERY} steps"
echo "  Log:     every ${LOG_EVERY} steps"
echo "========================================"

# ---------- Phase 1: CRS model ----------
echo ""
echo "########################################"
echo "#  PHASE 1: CRS (alpha=0.3) — 5000 steps"
echo "########################################"
echo ""

/Users/bytedance/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
    workbuddy_scripts/small_lm_crs.py \
    --mode crs \
    --alpha 0.3 \
    --batch_size ${BATCH_SIZE} \
    --max_seq_len ${SEQ_LEN} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --max_steps ${MAX_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --num_workers ${NUM_WORKERS}

CRS_EXIT=$?
echo ""
echo "CRS training finished with exit code ${CRS_EXIT}"

if [ ${CRS_EXIT} -ne 0 ]; then
    echo "CRS training failed. Check logs above."
    echo "To resume from last checkpoint, rerun with --resume_from:"
    echo "  python3 workbuddy_scripts/small_lm_crs.py --mode crs --alpha 0.3 --max_steps 5000 \\"
    echo "      --resume_from outputs/small_lm_crs/crs_alpha0_3/checkpoints/step_XXXX.pt"
    exit ${CRS_EXIT}
fi

# ---------- Phase 2: Baseline model ----------
echo ""
echo "########################################"
echo "#  PHASE 2: Baseline — 5000 steps"
echo "########################################"
echo ""

/Users/bytedance/.workbuddy/binaries/python/versions/3.13.12/bin/python3 \
    workbuddy_scripts/small_lm_crs.py \
    --mode baseline \
    --batch_size ${BATCH_SIZE} \
    --max_seq_len ${SEQ_LEN} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --max_steps ${MAX_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --num_workers ${NUM_WORKERS}

BASE_EXIT=$?
echo ""
echo "Baseline training finished with exit code ${BASE_EXIT}"

if [ ${BASE_EXIT} -ne 0 ]; then
    echo "Baseline training failed. Check logs above."
    echo "To resume from last checkpoint, rerun with --resume_from:"
    echo "  python3 workbuddy_scripts/small_lm_crs.py --mode baseline --max_steps 5000 \\"
    echo "      --resume_from outputs/small_lm_crs/baseline/checkpoints/step_XXXX.pt"
    exit ${BASE_EXIT}
fi

# ---------- Done ----------
echo ""
echo "========================================"
echo "  ALL DONE — Both models trained 5000 steps"
echo "========================================"
echo ""
echo "Output structure:"
echo "  outputs/small_lm_crs/"
echo "    ├── crs_alpha0_3/"
echo "    │   ├── checkpoints/"
echo "    │   │   ├── step_0200.pt"
echo "    │   │   ├── step_0400.pt"
echo "    │   │   ├── ..."
echo "    │   │   └── step_5000_final.pt"
echo "    │   └── metrics.jsonl"
echo "    └── baseline/"
echo "        ├── checkpoints/"
echo "        │   ├── step_1000.pt"
echo "        │   ├── step_2000.pt"
echo "        │   ├── step_3000.pt"
echo "        │   ├── step_4000.pt"
echo "        │   └── step_5000_final.pt"
echo "        └── metrics.jsonl"
echo ""
echo "To resume CRS training from step 5000 → 10000:"
echo "  python3 workbuddy_scripts/small_lm_crs.py --mode crs --alpha 0.3 --max_steps 10000 \\"
echo "      --resume_from outputs/small_lm_crs/crs_alpha0_3/checkpoints/step_5000_final.pt"
echo ""
echo "To resume baseline training from step 5000 → 10000:"
echo "  python3 workbuddy_scripts/small_lm_crs.py --mode baseline --max_steps 10000 \\"
echo "      --resume_from outputs/small_lm_crs/baseline/checkpoints/step_5000_final.pt"
