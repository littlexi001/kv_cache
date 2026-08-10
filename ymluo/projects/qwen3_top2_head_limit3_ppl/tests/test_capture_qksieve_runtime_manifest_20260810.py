from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import capture_qksieve_runtime_manifest_20260810 as runtime  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_files_follow_index_and_include_metadata(tmp_path: Path) -> None:
    files = {
        "config.json": b"{}",
        "generation_config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model-00001-of-00002.safetensors": b"one",
        "model-00002-of-00002.safetensors": b"two",
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    index = {
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
            "layer.2": "model-00001-of-00002.safetensors",
        }
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    observed = runtime.model_files(tmp_path)

    assert [path.name for path in observed] == sorted(
        [*files, "model.safetensors.index.json"]
    )
    assert runtime.sha256(tmp_path / "config.json") == digest(
        tmp_path / "config.json"
    )
