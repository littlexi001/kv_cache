from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_block_partition_estimator_probe_20260715 import (  # noqa: E402
    block_partition_metrics,
    sampled_tail_partition_metrics,
)


def test_constant_score_blocks_are_exact_for_both_estimators() -> None:
    scores = torch.zeros((1, 2, 8), dtype=torch.float32)
    rows = block_partition_metrics(scores, block_size=4, thresholds=(0.75,))

    assert len(rows) == 2
    for row in rows:
        assert float(row["block_logz_mae"]) < 1e-6
        assert float(row["total_logz_mae"]) < 1e-6
        assert abs(float(row["mean_actual_mass"]) - 1.0) < 1e-6
        assert abs(float(row["mean_selected_token_ratio"]) - 1.0) < 1e-6


def test_partition_probe_handles_partial_last_block() -> None:
    scores = torch.arange(10, dtype=torch.float32).view(1, 1, 10)
    rows = block_partition_metrics(scores, block_size=4, thresholds=(0.75, 0.9))

    assert len(rows) == 4
    assert all(torch.isfinite(torch.tensor(float(row["block_logz_mae"]))) for row in rows)
    assert all(0.0 < float(row["mean_selected_token_ratio"]) <= 1.0 for row in rows)


def test_tail_sample_estimator_returns_valid_budget_and_mass() -> None:
    torch.manual_seed(3)
    scores = torch.randn((1, 2, 101), dtype=torch.float32)
    rows = sampled_tail_partition_metrics(
        scores,
        budget_fractions=(0.01, 0.02, 0.04),
        thresholds=(0.75, 0.9),
        sample_fraction=0.1,
        seed=7,
    )

    assert len(rows) == 2
    assert all(0.0 < float(row["mean_selected_token_ratio"]) <= 0.04 for row in rows)
    assert all(0.0 < float(row["mean_actual_mass"]) <= 1.0 for row in rows)
    assert all(float(row["total_logz_mae"]) >= 0.0 for row in rows)
