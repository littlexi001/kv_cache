from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "download_hf_snapshot_direct_20260810.py"
)
SPEC = importlib.util.spec_from_file_location("direct_snapshot", MODULE_PATH)
assert SPEC and SPEC.loader
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def test_selects_sharded_weights_and_omits_duplicate_consolidated(
    tmp_path: Path,
) -> None:
    siblings = {
        name: {"rfilename": name}
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "consolidated.safetensors",
            "tokenizer.json",
            "README.md",
        )
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    assert download.metadata_files(siblings) == [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ]
    assert download.weight_files(tmp_path, siblings) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_existing_lfs_file_requires_size_and_sha(tmp_path: Path) -> None:
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"valid")
    expected = download.sha256_file(path)
    assert download.valid_existing_file(path, 5, expected)
    assert not download.valid_existing_file(path, 6, expected)
    assert not download.valid_existing_file(path, 5, "0" * 64)
