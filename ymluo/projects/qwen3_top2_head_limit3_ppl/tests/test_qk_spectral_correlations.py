import numpy as np

from analyze_qk_spectral_correlations_20260726 import (
    average_ranks,
    correlation,
)


def test_average_ranks_handles_ties() -> None:
    values = np.asarray([3.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(average_ranks(values), [4.0, 1.5, 1.5, 3.0])


def test_spearman_detects_monotonic_relation() -> None:
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = np.asarray([1.0, 4.0, 9.0, 16.0])
    pearson, spearman = correlation(x, y)
    assert pearson < 1.0
    assert spearman == 1.0
