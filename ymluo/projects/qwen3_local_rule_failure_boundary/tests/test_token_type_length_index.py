from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from build_token_type_length_index import build_index, encode_array, write_gzip_json  # noqa: E402


def test_index_merges_token_ids_without_losing_tiny_probability_mass(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    logits = np.stack(
        [
            np.full((2, 2), 1.0, dtype=np.float16),
            np.full((2, 2), 4.0, dtype=np.float16),
        ]
    )
    mass = np.stack(
        [
            np.full((2, 2), 1e-12, dtype=np.float32),
            np.full((2, 2), 2e-12, dtype=np.float32),
        ]
    )
    write_gzip_json(
        input_dir / "length_500.json.gz",
        {
            "schema_version": 1,
            "key_length": 3,
            "num_layers": 2,
            "num_attention_heads": 2,
            "shape": [2, 2, 2],
            "token_ids_u32_b64": encode_array(np.array([10, 11]), np.dtype("<u4")),
            "token_counts_u32_b64": encode_array(np.array([2, 1]), np.dtype("<u4")),
            "token_text": {"10": " river", "11": "River"},
            "mean_logits_f16_b64": encode_array(logits, np.dtype("<f2")),
            "probability_mass_f32_b64": encode_array(mass, np.dtype("<f4")),
        },
    )

    manifest = build_index(input_dir, output_dir)

    assert manifest["token_count"] == 1
    assert manifest["lengths"] == [500]
    entry = manifest["tokens"]["river"]
    assert entry["token_ids"] == [10, 11]
    with gzip.open(output_dir / entry["file"], "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    counts = np.frombuffer(
        base64.b64decode(payload["occurrence_counts_u32_b64"]), dtype="<u4"
    )
    merged_logits = np.frombuffer(
        base64.b64decode(payload["mean_logits_f16_b64"]), dtype="<f2"
    )
    merged_mass = np.frombuffer(
        base64.b64decode(payload["probability_mass_f32_b64"]), dtype="<f4"
    )
    assert counts.tolist() == [3]
    assert merged_logits.tolist() == pytest.approx([2.0] * 4, abs=1e-3)
    assert merged_mass.tolist() == pytest.approx([3e-12] * 4, rel=1e-6)
