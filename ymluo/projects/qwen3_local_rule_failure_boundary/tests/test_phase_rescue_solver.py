from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phase_rescue_solver import phase_lift, phase_scores, solve_phase_rescue  # noqa: E402


def test_exact_noop_for_zero_target() -> None:
    a = np.asarray([1.0, -0.4, 0.7])
    b = np.asarray([0.2, 0.8, -0.1])
    phase = np.asarray([0.3, -2.1, 7.4])

    result = solve_phase_rescue(
        a,
        b,
        phase,
        target_lift=0.0,
        budget=2,
        phase_cap=0.25,
    )

    assert result.feasible
    assert result.status == "no_op"
    assert result.support == ()
    assert result.norm == 0.0
    assert result.achieved_lift == 0.0
    np.testing.assert_array_equal(result.shifts, np.zeros_like(phase))
    np.testing.assert_array_equal(
        phase_scores(a, b, phase, result.shifts),
        phase_scores(a, b, phase),
    )


def test_positive_lift_respects_strict_budget_and_per_plane_caps() -> None:
    a = np.asarray([1.0, 0.8, 1.2, 0.4])
    b = np.asarray([0.0, 0.2, -0.1, 0.5])
    phase = np.asarray([1.3, -1.0, 0.9, -2.0])
    cap = np.asarray([0.50, 0.45, 0.12, 0.30])

    result = solve_phase_rescue(
        a,
        b,
        phase,
        target_lift=0.25,
        budget=2,
        phase_cap=cap,
    )

    assert result.feasible
    assert result.achieved_lift >= 0.25 - 1e-8
    assert len(result.support) <= 2
    assert np.count_nonzero(np.abs(result.shifts) > 1e-9) <= 2
    assert np.all(np.abs(result.shifts) <= cap + 1e-12)
    assert math.isclose(
        result.achieved_lift,
        phase_lift(a, b, phase, result.shifts),
        abs_tol=1e-12,
    )


def test_budget_one_selects_only_one_frequency_plane() -> None:
    a = np.asarray([2.0, 1.0, 0.5])
    b = np.zeros(3)
    phase = np.asarray([1.2, 1.2, 1.2])

    result = solve_phase_rescue(
        a,
        b,
        phase,
        target_lift=0.20,
        budget=1,
        phase_cap=0.4,
    )

    assert result.feasible
    assert len(result.support) == 1
    assert result.support == (0,)
    assert np.count_nonzero(result.shifts) == 1


def test_infeasible_problem_returns_max_lift_fallback() -> None:
    a = np.asarray([1.0, 2.0])
    b = np.zeros(2)
    phase = np.asarray([math.pi / 2.0, math.pi / 2.0])
    cap = np.asarray([0.1, 0.2])

    result = solve_phase_rescue(
        a,
        b,
        phase,
        target_lift=10.0,
        budget=1,
        phase_cap=cap,
    )

    expected = 2.0 * math.sin(0.2)
    assert not result.feasible
    assert result.status == "infeasible_max_lift"
    assert result.support == (1,)
    assert math.isclose(result.shifts[1], 0.2, abs_tol=1e-12)
    assert math.isclose(result.achieved_lift, expected, abs_tol=1e-12)
    assert math.isclose(result.max_recoverable_lift, expected, abs_tol=1e-12)


def test_solution_matches_small_dimensional_bruteforce_grid() -> None:
    """The nonlinear water filling agrees with an independent dense search."""

    a = np.asarray([1.0, 0.8])
    b = np.asarray([0.0, 0.2])
    phase = np.asarray([1.3, -1.0])
    cap = np.asarray([0.50, 0.45])
    target = 0.25

    result = solve_phase_rescue(
        a,
        b,
        phase,
        target_lift=target,
        budget=2,
        phase_cap=cap,
    )
    assert result.feasible

    first = np.linspace(-cap[0], cap[0], 501)
    second = np.linspace(-cap[1], cap[1], 501)
    first_grid, second_grid = np.meshgrid(first, second, indexing="ij")
    native = float(np.sum(phase_scores(a, b, phase)))
    lift_grid = (
        a[0] * np.cos(phase[0] - first_grid)
        + b[0] * np.sin(phase[0] - first_grid)
        + a[1] * np.cos(phase[1] - second_grid)
        + b[1] * np.sin(phase[1] - second_grid)
        - native
    )
    norm_grid = np.hypot(first_grid, second_grid)
    brute_norm = float(np.min(norm_grid[lift_grid >= target]))

    # A grid can only overestimate the continuous optimum.  Two grid cells are
    # allowed for discretisation, while a large gap would expose a bad KKT
    # branch or an incorrect nonlinear line search.
    grid_diagonal = math.hypot(first[1] - first[0], second[1] - second[0])
    assert result.norm <= brute_norm + 2.0 * grid_diagonal
    assert result.norm >= brute_norm - 2.0 * grid_diagonal
    assert abs(result.achieved_lift - target) <= 2e-8


def test_nonconcave_phase_minimum_uses_exact_nonlinear_rescue() -> None:
    """At phase pi the first derivative is zero, so linear rescue would fail."""

    result = solve_phase_rescue(
        np.asarray([1.0]),
        np.asarray([0.0]),
        np.asarray([math.pi]),
        target_lift=0.05,
        budget=1,
        phase_cap=0.5,
    )

    expected_shift = math.acos(-0.95)
    expected_shift = math.pi - expected_shift
    assert result.feasible
    assert math.isclose(result.shifts[0], expected_shift, abs_tol=2e-8)
    assert math.isclose(result.achieved_lift, 0.05, abs_tol=2e-8)

