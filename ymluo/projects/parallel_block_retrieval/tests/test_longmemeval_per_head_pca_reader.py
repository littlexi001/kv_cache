from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from evaluate_longmemeval_10m_per_head_pca_reader import (  # noqa: E402
    pack_context,
    quantize_dequantize_int4,
    quota_union,
    selective_head_rrf,
)


class TinyTokenizer:
    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def test_int4_quantization_is_bounded_and_shape_preserving() -> None:
    values = torch.tensor([[[-8.0, -1.0, 0.0, 7.0]]], dtype=torch.float16)
    recovered = quantize_dequantize_int4(values)
    assert recovered.shape == values.shape
    assert torch.isfinite(recovered).all()
    assert float((recovered.float() - values.float()).abs().max()) <= 8.0 / 7.0


def test_selective_head_rrf_preserves_independent_head_votes() -> None:
    scores = np.array(
        [
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.9, 1.0],
        ]
    )
    ranking, diagnostics = selective_head_rrf(
        scores,
        output_depth=2,
        active_fraction=0.5,
        per_head_depth=1,
        rrf_constant=20.0,
    )
    assert set(ranking) == {0, 1}
    assert diagnostics["active_channels"] == 2.0


def test_quota_union_is_stable_and_unique() -> None:
    assert quota_union([1, 2, 3], [3, 4, 5], 5) == [1, 2, 3, 4, 5]


def test_context_packing_never_exceeds_budget() -> None:
    context, selected, tokens = pack_context(
        [0, 1, 2],
        {0: "a b c", 1: "d e f", 2: "g h i"},
        np.array([-1, -1, -1]),
        TinyTokenizer(),
        token_budget=14,
        include_dates=False,
    )
    assert tokens <= 14
    assert selected
    assert context
