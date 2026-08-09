from __future__ import annotations

import base64
import gzip
import json
import struct
import sys
from pathlib import Path

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from run_attention_confidence_sweep_8b import (  # noqa: E402
    tensor_u32_base64,
    write_full_pre_softmax_scope,
)


def decode_f16(payload: str) -> tuple[float, ...]:
    raw = base64.b64decode(payload)
    return struct.unpack(f"<{len(raw) // 2}e", raw)


def test_full_pre_softmax_head_shard_round_trips(tmp_path: Path) -> None:
    logits = torch.tensor([1.25, -2.5, 0.0, 3.75], dtype=torch.float32)
    destination = tmp_path / "heads" / "layer_00_head_01.json.gz"
    logsumexp = float(torch.logsumexp(logits, dim=-1).item())

    write_full_pre_softmax_scope(
        destination,
        scope="head",
        layer=0,
        head=1,
        key_length=4,
        logits=logits,
        top_positions=[3, 0],
        logsumexp=logsumexp,
    )

    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["scope"] == "head"
    assert payload["layer"] == 0
    assert payload["head"] == 1
    assert payload["key_length"] == 4
    assert payload["top_logit_positions"] == [3, 0]
    assert decode_f16(payload["logits_f16_b64"]) == pytest.approx(logits.tolist(), abs=1e-3)
    assert payload["logsumexp"] == pytest.approx(logsumexp, abs=1e-8)
    reconstructed = [
        torch.exp(torch.tensor(value - payload["logsumexp"])).item()
        for value in decode_f16(payload["logits_f16_b64"])
    ]
    assert sum(reconstructed) == pytest.approx(1.0, abs=5e-4)


def test_full_pre_softmax_aggregate_shard_saves_exact_mean_probability(tmp_path: Path) -> None:
    logits = torch.tensor([0.5, 0.25, -1.0], dtype=torch.float32)
    probabilities = torch.tensor([0.2, 0.7, 0.1], dtype=torch.float32)
    destination = tmp_path / "overall.json.gz"

    write_full_pre_softmax_scope(
        destination,
        scope="overall",
        key_length=3,
        logits=logits,
        probabilities=probabilities,
        top_positions=[0, 1],
    )

    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert decode_f16(payload["logits_f16_b64"]) == pytest.approx(logits.tolist(), abs=1e-3)
    assert decode_f16(payload["probabilities_f16_b64"]) == pytest.approx(
        probabilities.tolist(), abs=1e-3
    )


def test_token_ids_are_uint32_little_endian() -> None:
    raw = base64.b64decode(tensor_u32_base64([0, 151643, 42]))
    assert struct.unpack("<3I", raw) == (0, 151643, 42)
