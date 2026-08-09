from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from attention_selectors import (  # noqa: E402
    SelectorSpec,
    actual_history_fraction,
    build_keep_mask,
    historical_budget,
    parse_selector,
)


def test_historical_budget_rounds_up_and_caps() -> None:
    assert historical_budget(0, 0.02) == 0
    assert historical_budget(10, 0.02) == 1
    assert historical_budget(101, 0.02) == 3
    assert historical_budget(10, 1.0) == 10


def test_sink_recent_is_equal_budget_and_keeps_self() -> None:
    scores = torch.arange(12, dtype=torch.float32).view(1, 1, 1, 12)
    keep, history = build_keep_mask(scores, parse_selector("sink_recent_s1"), 0.25)
    assert torch.nonzero(history[0, 0, 0], as_tuple=False).flatten().tolist() == [0, 9, 10]
    assert torch.nonzero(keep[0, 0, 0], as_tuple=False).flatten().tolist() == [0, 9, 10, 11]


def test_top_attention_uses_each_head_scores() -> None:
    scores = torch.tensor([[[[9.0, 1.0, 0.0, 5.0]], [[0.0, 8.0, 1.0, 5.0]]]])
    _, history = build_keep_mask(scores, SelectorSpec("top_attention"), 0.34)
    assert torch.nonzero(history[0, 0, 0], as_tuple=False).flatten().tolist() == [0, 1]
    assert torch.nonzero(history[0, 1, 0], as_tuple=False).flatten().tolist() == [1, 2]


def test_ratio_one_matches_full_causal_mask() -> None:
    scores = torch.randn(1, 2, 3, 7)
    keep, history = build_keep_mask(scores, SelectorSpec("top_attention"), 1.0)
    selected, eligible = actual_history_fraction(history)
    assert selected == eligible
    for query_idx, current in enumerate([4, 5, 6]):
        assert keep[0, :, query_idx, : current + 1].all()
        assert not keep[0, :, query_idx, current + 1 :].any()


def test_drop_remote_is_subset_of_oracle_top() -> None:
    scores = torch.arange(20, dtype=torch.float32).view(1, 1, 1, 20)
    _, oracle = build_keep_mask(scores, SelectorSpec("top_attention"), 0.5)
    _, local = build_keep_mask(
        scores,
        SelectorSpec("top_attention_drop_remote"),
        0.5,
        role_sink_tokens=2,
        role_recent_tokens=3,
    )
    assert torch.all(local <= oracle)
    assert torch.nonzero(local[0, 0, 0], as_tuple=False).flatten().tolist() == [16, 17, 18]


def test_random_control_is_deterministic() -> None:
    scores = torch.randn(1, 3, 2, 10)
    first = build_keep_mask(scores, SelectorSpec("random"), 0.3, layer_idx=4)[1]
    second = build_keep_mask(scores, SelectorSpec("random"), 0.3, layer_idx=4)[1]
    assert torch.equal(first, second)
