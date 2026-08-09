from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_causal_seed_sampling import weighted_sample_without_replacement  # noqa: E402


def test_weighted_sampling_is_unique_and_prefers_nonzero_support() -> None:
    weights = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    sampled = weighted_sample_without_replacement(weights, 3, np.random.default_rng(9))

    assert np.unique(sampled).size == 3
    assert set(sampled.tolist()) == {2, 3, 4}
