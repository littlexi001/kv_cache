#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/LongBench}"
CACHE_DIR="${CACHE_DIR:-/home/fdong/qksieve_iclr2027/cache/huggingface}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "${OUT_DIR}" "${ROOT}/logs"

"${PYTHON}" - "${OUT_DIR}" "${CACHE_DIR}" <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

out_dir = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
tasks = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

try:
    zip_path = Path(
        hf_hub_download(
            repo_id="THUDM/LongBench",
            filename="data.zip",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )
except Exception as error:
    # Some mirrors serve the file but reject the metadata HEAD request used by
    # huggingface_hub. A resumable direct GET avoids changing the data source.
    direct_dir = cache_dir / "direct" / "THUDM_LongBench"
    direct_dir.mkdir(parents=True, exist_ok=True)
    zip_path = direct_dir / "data.zip"
    print(f"huggingface_hub failed, using direct mirror GET: {error}")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--output",
            str(zip_path),
            "https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data.zip",
        ],
        check=True,
    )
with zipfile.ZipFile(zip_path) as archive:
    members = set(archive.namelist())
    for task in tasks:
        member = f"data/{task}.jsonl"
        if member not in members:
            raise FileNotFoundError(f"{member} is absent from {zip_path}")
        destination = out_dir / f"{task}.jsonl"
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)

counts = {}
total = 0
for task in tasks:
    path = out_dir / f"{task}.jsonl"
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty LongBench task: {task}")
    first = json.loads(rows[0])
    missing = {"context", "input", "answers"} - set(first)
    if missing:
        raise RuntimeError(f"{task} is missing fields: {sorted(missing)}")
    counts[task] = len(rows)
    total += len(rows)

archive_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
manifest = {
    "repo_id": "THUDM/LongBench",
    "archive": str(zip_path),
    "archive_sha256": archive_hash,
    "data_dir": str(out_dir),
    "task_counts": counts,
    "total_examples": total,
}
(out_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY
