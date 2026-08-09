from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_qkbalanced_longbench_paired_20260727 import optional_mean


def test_optional_mean_preserves_missing_diagnostics() -> None:
    assert optional_mean([None, None]) is None
    assert optional_mean([None, 0.05, 0.07]) == pytest.approx(0.06)
