#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL_ID="${MODEL_ID:-LLM-Research/Meta-Llama-3.1-8B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/home/fdong/qksieve_iclr2027/models/Meta-Llama-3.1-8B-Instruct-ms}"
CACHE_DIR="${CACHE_DIR:-/home/fdong/qksieve_iclr2027/cache/modelscope}"

export MODELSCOPE_CACHE="${CACHE_DIR}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "${MODEL_DIR}" "${CACHE_DIR}"

"${PYTHON}" - "${MODEL_ID}" "${MODEL_DIR}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from modelscope import snapshot_download

model_id, model_dir_raw = sys.argv[1:]
model_dir = Path(model_dir_raw)
snapshot_download(
    model_id,
    local_dir=str(model_dir),
    revision="master",
    ignore_patterns=["*.pth"],
)

config_path = model_dir / "config.json"
if not config_path.is_file():
    raise FileNotFoundError(config_path)
config = json.loads(config_path.read_text(encoding="utf-8"))
expected = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}
for name, value in expected.items():
    if config.get(name) != value:
        raise RuntimeError(
            f"unexpected Llama-3.1-8B config {name}={config.get(name)!r}"
        )

weight_files = sorted(
    path
    for pattern in ("*.safetensors", "pytorch_model*.bin")
    for path in model_dir.glob(pattern)
)
weight_bytes = sum(path.stat().st_size for path in weight_files)
if weight_bytes < 14_000_000_000:
    raise RuntimeError(
        f"incomplete Llama-3.1-8B weights: {weight_bytes} bytes"
    )

hashes = {}
for name in (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
):
    path = model_dir / name
    if path.is_file():
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "source": "ModelScope",
    "model_id": model_id,
    "model_dir": str(model_dir),
    "weight_files": [path.name for path in weight_files],
    "weight_bytes": weight_bytes,
    "config_contract": expected,
    "metadata_sha256": hashes,
}
(model_dir / "qksieve_download_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY
