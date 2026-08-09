from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_k_svd_conflict_geometry import energy_fraction, find_subsequence, projected_metrics  # noqa: E402

import torch


def test_find_subsequence() -> None:
    assert find_subsequence([1, 2, 3, 2, 3], [2, 3]) == 1
    assert find_subsequence([1, 2], [3]) is None


def test_projection_metrics_are_separated() -> None:
    a = torch.tensor([1.0, 0.0, 0.0, 1.0])
    b = torch.tensor([1.0, 0.0, 0.0, -1.0])
    metrics = projected_metrics("x", a, b, 2)
    assert metrics["x_cos_top"] == 1.0
    assert metrics["x_cos_tail"] == -1.0
    assert energy_fraction(a, 2) == 0.5
