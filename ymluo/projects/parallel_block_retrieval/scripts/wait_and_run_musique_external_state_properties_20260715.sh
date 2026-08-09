#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
MIN_GPUS="${MIN_GPUS:-6}"
MAX_USED_MB="${MAX_USED_MB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-30}"

cd "${ROOT}"
while true; do
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, -v limit="${MAX_USED_MB}" \
          '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($2 + 0) <= limit) print $1}'
  )
  if [[ "${#free_gpus[@]}" -ge "${MIN_GPUS}" ]]; then
    devices="$(IFS=,; echo "${free_gpus[*]}")"
    echo "[musique-properties] starting on GPUs ${devices}" >&2
    exec env GPUS="${devices}" scripts/run_musique_external_state_properties_10m_20260715.sh
  fi
  echo "[musique-properties] waiting: ${#free_gpus[@]} free GPUs" >&2
  sleep "${POLL_SECONDS}"
done
