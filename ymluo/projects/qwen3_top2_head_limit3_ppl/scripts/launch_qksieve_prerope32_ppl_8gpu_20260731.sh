#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
BASE_SCRIPT="${PROJECT_ROOT}/scripts/launch_qksieve_lowbit_ppl_8gpu_20260731.sh"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_prerope32_ppl_8gpu}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_prerope32int2_b8_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_prerope32int4_b8_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_prerope32adaptive_b8_fulltopk_k1280}"

export PROJECT_ROOT
export RUN_ROOT
export VARIANTS
exec bash "${BASE_SCRIPT}"
