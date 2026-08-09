from __future__ import annotations

import itertools

import numpy as np


BIT_LEVELS = (0, 1, 2, 4, 8)
NORMALIZED_RATE = {0: 0, 1: 2, 2: 3, 4: 5, 8: 9}


def _feasible_allocations() -> list[tuple[int, ...]]:
    return [
        allocation
        for allocation in itertools.product(BIT_LEVELS, repeat=8)
        if sum(NORMALIZED_RATE[bit] for bit in allocation) <= 15
    ]


def test_paper_feasible_allocation_count_matches_physical_rate() -> None:
    allocations = _feasible_allocations()

    assert len(allocations) == 13_817
    assert (0,) * 8 in allocations
    assert all(
        sum(NORMALIZED_RATE[bit] for bit in allocation) <= 15
        for allocation in allocations
    )


def test_uniform_calibration_gap_implies_two_epsilon_selection_regret() -> None:
    rng = np.random.default_rng(20260728)
    population = rng.uniform(0.0, 4.0, size=13_817)
    calibration_error = rng.uniform(-0.17, 0.17, size=population.shape)
    calibration = population + calibration_error
    epsilon = float(np.max(np.abs(calibration_error)))

    selected = int(np.argmin(calibration))
    oracle = int(np.argmin(population))

    assert population[selected] - population[oracle] <= 2.0 * epsilon + 1e-12


def test_full_score_regret_adds_only_cross_band_residuals() -> None:
    rng = np.random.default_rng(728)
    diagonal = rng.uniform(0.0, 3.0, size=4096)
    calibration_error = rng.uniform(-0.08, 0.08, size=diagonal.shape)
    cross_band = rng.uniform(-0.03, 0.03, size=diagonal.shape)
    calibration = diagonal + calibration_error
    full = diagonal + cross_band
    epsilon = float(np.max(np.abs(calibration_error)))

    selected = int(np.argmin(calibration))
    full_oracle = int(np.argmin(full))
    right_hand_side = (
        2.0 * epsilon
        + abs(float(cross_band[selected]))
        + abs(float(cross_band[full_oracle]))
    )

    assert full[selected] - full[full_oracle] <= right_hand_side + 1e-12
