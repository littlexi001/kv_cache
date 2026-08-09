#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/home/fdong/ymluo/external/musique_official}"
GDOWN="${GDOWN:-/home/fdong/miniconda3/envs/moe/bin/gdown}"

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"
downloaded=false
for _attempt in $(seq 1 12); do
  if "${GDOWN}" --continue 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h -O musique_v1.0.zip; then
    downloaded=true
    break
  fi
  sleep 2
done
if [[ "${downloaded}" != true ]]; then
  echo "MuSiQue download did not complete after retries" >&2
  exit 1
fi
unzip -q -o musique_v1.0.zip
