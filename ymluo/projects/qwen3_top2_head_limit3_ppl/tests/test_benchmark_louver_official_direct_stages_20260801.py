from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_louver_official_direct_stages_20260801 import (  # noqa: E402
    parse_lengths,
    sampled_threshold_rank,
    selected_count,
)


def test_louver_length_and_budget_contract() -> None:
    assert parse_lengths("32768,8192,32768") == [8192, 32768]
    assert selected_count(8192) == 492
    assert selected_count(32768) == 1280
    assert selected_count(131072) == 1280


def test_sampled_threshold_rank_tracks_target_fraction() -> None:
    assert sampled_threshold_rank(8192, 492, 256) == 16
    assert sampled_threshold_rank(32768, 1280, 256) == 10
    assert sampled_threshold_rank(131072, 1280, 256) == 3
