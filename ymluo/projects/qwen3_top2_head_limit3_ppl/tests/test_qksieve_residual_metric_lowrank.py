from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qksieve_residual_metric_lowrank_20260804 import (  # noqa: E402
    correlation,
    rankdata,
    risk_metrics,
    top_fraction_recall,
)


def test_rankdata_and_correlation_are_order_consistent() -> None:
    values = torch.tensor([3.0, 1.0, 2.0])
    assert torch.equal(rankdata(values), torch.tensor([2.0, 0.0, 1.0]))
    assert correlation(values, 2.0 * values + 1.0) > 0.999999


def test_identical_risk_has_exact_metrics() -> None:
    squared = torch.tensor([1.0, 4.0, 9.0, 16.0])
    metrics = risk_metrics(squared, squared)
    assert metrics["risk_relative_l2"] == 0.0
    assert metrics["log_risk_rmse"] == 0.0
    assert metrics["pearson"] > 0.999999
    assert metrics["spearman"] > 0.999999
    assert metrics["top10_recall"] == 1.0


def test_top_fraction_recall_detects_disjoint_selection() -> None:
    exact = torch.arange(100, dtype=torch.float32)
    approximate = -exact
    assert top_fraction_recall(approximate, exact, 0.10) == 0.0
