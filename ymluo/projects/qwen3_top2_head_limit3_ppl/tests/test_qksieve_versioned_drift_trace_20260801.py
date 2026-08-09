from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qksieve_versioned_drift_trace_20260801 import (  # noqa: E402
    covariance_drift,
    proxy_scores,
    selection_rows,
)


def test_versioned_identity_segments_recover_exact_scores() -> None:
    generator = torch.Generator().manual_seed(20260801)
    queries = torch.randn(3, 4, generator=generator)
    first = torch.randn(5, 4, generator=generator)
    second = torch.randn(7, 4, generator=generator)
    identity = torch.eye(4)
    state = {"query_factor": identity}
    scores = proxy_scores(
        queries,
        [first, second],
        [state, state],
        0.5,
    )
    expected = queries @ torch.cat([first, second], dim=0).T * 0.5
    torch.testing.assert_close(scores, expected)


def test_exact_selector_has_unit_recall_and_oracle_mass() -> None:
    scores = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
    rows = selection_rows(scores, {"exact": scores}, topk=2)
    assert rows[0]["topk_recall"] == 1.0
    assert rows[0]["selected_to_oracle_mass"] == 1.0


def test_covariance_drift_is_zero_for_identical_samples() -> None:
    values = torch.randn(16, 8, generator=torch.Generator().manual_seed(7))
    assert covariance_drift(values, values) == 0.0
