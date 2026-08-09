#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"

export PROJECT_ROOT
export RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_dualmass_ppl_8gpu}"
export VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_fixedalloc_post2xdualmass_i112_41_l00to08_fulltopk_k1280}"

exec bash "${SCRIPT_DIR}/launch_qksieve_post2x_prerope_ppl_8gpu_20260731.sh"
