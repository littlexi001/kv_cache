from __future__ import annotations

"""Strict-sparse, box-constrained phase rescue.

For one rotary frequency plane, write the native contribution as

    f_i(phi_i) = A_i cos(phi_i) + B_i sin(phi_i).

The solver changes the phase to ``phi_i - delta_i`` and solves the following
problem approximately (exactly for the one-dimensional subproblems):

    minimise    ||delta||_2
    subject to  sum_i [f_i(phi_i-delta_i)-f_i(phi_i)] >= target_lift,
                ||delta||_0 <= budget,
                |delta_i| <= phase_cap_i.

Only movement along the shortest arc to the closest maximum is useful to a
minimum-norm positive rescue.  This turns every selected plane into a bounded,
monotone one-dimensional lift curve.  A fixed support is solved with a
separable Lagrange multiplier (nonlinear water filling).  All trigonometric
scores are re-evaluated exactly; bisection on a discontinuous water-filling
jump supplies the final line search instead of trusting a linearised lift.

Selecting the globally optimal sparse support is combinatorial.  Small
problems are enumerated exactly.  Larger problems use a heuristic support
ranking computed from exact recoverable lift and phase risk, with a capacity
guard.  The returned solution always obeys the strict budget and never reports
feasibility when the chosen support cannot provide the requested lift; it is
not claimed to be the globally minimum-norm sparse solution in this regime.
"""

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, Sequence

import numpy as np


_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class PhaseRescueResult:
    """Result of :func:`solve_phase_rescue`.

    ``shifts`` follows the convention ``new_phase = phase - shifts``.
    ``support`` contains precisely the non-zero entries of ``shifts`` rather
    than unused members of a candidate support.
    """

    shifts: np.ndarray
    achieved_lift: float
    target_lift: float
    support: tuple[int, ...]
    norm: float
    feasible: bool
    max_recoverable_lift: float
    status: str
    iterations: int


@dataclass(frozen=True)
class _PlaneGeometry:
    amplitude: np.ndarray
    distance: np.ndarray
    direction: np.ndarray
    radius: np.ndarray
    max_lift: np.ndarray


def _as_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Convert NumPy, CPU/GPU torch-like, or Python inputs to float64."""

    candidate = values
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    array = np.asarray(candidate, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalise_cap(
    phase_cap: float | Sequence[float] | np.ndarray,
    size: int,
) -> np.ndarray:
    cap = np.asarray(phase_cap, dtype=np.float64)
    if cap.ndim == 0:
        cap = np.full(size, float(cap), dtype=np.float64)
    elif cap.ndim == 1 and cap.shape[0] == size:
        cap = cap.copy()
    else:
        raise ValueError(
            f"phase_cap must be scalar or shape ({size},), got {cap.shape}"
        )
    if not np.all(np.isfinite(cap)) or np.any(cap < 0.0):
        raise ValueError("phase_cap must be finite and non-negative")
    return cap


def phase_scores(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    phase: Sequence[float] | np.ndarray,
    shifts: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return exact per-plane scores under ``new_phase = phase - shifts``."""

    a_array = _as_1d_float(a, "a")
    b_array = _as_1d_float(b, "b")
    phase_array = _as_1d_float(phase, "phase")
    if not (a_array.shape == b_array.shape == phase_array.shape):
        raise ValueError("a, b, and phase must have identical shapes")
    if shifts is None:
        shift_array = np.zeros_like(phase_array)
    else:
        shift_array = _as_1d_float(shifts, "shifts")
        if shift_array.shape != phase_array.shape:
            raise ValueError("shifts must have the same shape as phase")
    effective = phase_array - shift_array
    return a_array * np.cos(effective) + b_array * np.sin(effective)


def phase_lift(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    phase: Sequence[float] | np.ndarray,
    shifts: Sequence[float] | np.ndarray,
) -> float:
    """Return the exact total score lift caused by ``shifts``."""

    native = phase_scores(a, b, phase)
    rescued = phase_scores(a, b, phase, shifts)
    return float(np.sum(rescued - native, dtype=np.float64))


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    wrapped = (angle + math.pi) % _TWO_PI - math.pi
    # Both directions are equally short at -pi.  Choosing +pi makes the
    # direction deterministic and avoids a negative-zero corner case.
    return np.where(np.isclose(wrapped, -math.pi, atol=1e-14), math.pi, wrapped)


def _plane_geometry(
    a: np.ndarray,
    b: np.ndarray,
    phase: np.ndarray,
    cap: np.ndarray,
) -> _PlaneGeometry:
    amplitude = np.hypot(a, b)
    preferred_phase = np.arctan2(b, a)
    error = _wrap_to_pi(phase - preferred_phase)
    distance = np.abs(error)
    direction = np.sign(error)
    direction = np.where(distance > 0.0, direction, 0.0)

    # Passing the closest maximum cannot help a positive minimum-norm rescue.
    radius = np.minimum(cap, distance)
    max_lift = amplitude * (
        np.cos(distance - radius) - np.cos(distance)
    )
    max_lift = np.maximum(max_lift, 0.0)
    return _PlaneGeometry(amplitude, distance, direction, radius, max_lift)


def _lift_for_magnitudes(
    amplitude: np.ndarray,
    distance: np.ndarray,
    magnitudes: np.ndarray,
) -> float:
    value = amplitude * (
        np.cos(distance - magnitudes) - np.cos(distance)
    )
    return float(np.sum(value, dtype=np.float64))


def _magnitude_for_lift(
    amplitude: float,
    distance: float,
    requested_lift: float,
) -> float:
    """Exact shortest movement that supplies one plane's requested lift."""

    if requested_lift <= 0.0 or amplitude <= 0.0 or distance <= 0.0:
        return 0.0
    cosine = math.cos(distance) + requested_lift / amplitude
    cosine = min(1.0, max(-1.0, cosine))
    return max(0.0, distance - math.acos(cosine))


def _single_plane_lagrange_optimum(
    amplitude: float,
    distance: float,
    radius: float,
    multiplier: float,
    root_iterations: int = 70,
) -> float:
    """Globally maximise ``lambda*lift(u) - u**2/2`` on ``[0, radius]``.

    The derivative is

        lambda*rho*sin(distance-u) - u.

    Its derivative is monotone decreasing on the useful shortest arc, hence
    the original derivative is concave.  Its descending zero (if present),
    together with both endpoints, contains every possible global maximum.
    """

    if multiplier <= 0.0 or amplitude <= 0.0 or radius <= 0.0:
        return 0.0

    def derivative(value: float) -> float:
        return multiplier * amplitude * math.sin(distance - value) - value

    def objective(value: float) -> float:
        lift = amplitude * (
            math.cos(distance - value) - math.cos(distance)
        )
        return multiplier * lift - 0.5 * value * value

    candidates = [0.0, radius]

    # Locate the maximum of the concave derivative.  Its own derivative is
    # -lambda*rho*cos(distance-u)-1 and decreases with u on [0, distance].
    second_at_zero = -multiplier * amplitude * math.cos(distance) - 1.0
    second_at_radius = (
        -multiplier * amplitude * math.cos(distance - radius) - 1.0
    )
    if second_at_zero <= 0.0:
        derivative_peak = 0.0
    elif second_at_radius >= 0.0:
        derivative_peak = radius
    else:
        ratio = -1.0 / (multiplier * amplitude)
        ratio = min(1.0, max(-1.0, ratio))
        derivative_peak = distance - math.acos(ratio)
        derivative_peak = min(radius, max(0.0, derivative_peak))

    peak_value = derivative(derivative_peak)
    right_value = derivative(radius)
    if derivative_peak < radius and peak_value > 0.0 and right_value < 0.0:
        low, high = derivative_peak, radius
        for _ in range(root_iterations):
            middle = 0.5 * (low + high)
            if derivative(middle) > 0.0:
                low = middle
            else:
                high = middle
        candidates.append(0.5 * (low + high))

    values = np.asarray([objective(value) for value in candidates])
    maximum = float(np.max(values))
    # At a first-order jump, choosing the largest tied maximiser makes the
    # water-filling solution monotone in lambda.  The outer exact line search
    # then interpolates the jump to the requested lift.
    tied = [
        value
        for value, objective_value in zip(candidates, values)
        if objective_value >= maximum - 1e-13 * max(1.0, abs(maximum))
    ]
    return float(max(tied))


def _waterfill(
    amplitude: np.ndarray,
    distance: np.ndarray,
    radius: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    return np.asarray(
        [
            _single_plane_lagrange_optimum(rho, dist, cap, multiplier)
            for rho, dist, cap in zip(amplitude, distance, radius)
        ],
        dtype=np.float64,
    )


def _solve_fixed_support(
    geometry: _PlaneGeometry,
    support: np.ndarray,
    target_lift: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, bool, int]:
    """Solve the continuous nonlinear problem on one fixed support."""

    size = geometry.amplitude.shape[0]
    result = np.zeros(size, dtype=np.float64)
    if target_lift <= tolerance:
        return result, 0.0, True, 0
    if support.size == 0:
        return result, 0.0, False, 0

    rho = geometry.amplitude[support]
    distance = geometry.distance[support]
    radius = geometry.radius[support]
    maximum = float(np.sum(geometry.max_lift[support], dtype=np.float64))
    if maximum + tolerance < target_lift:
        result[support] = radius
        return result, maximum, False, 0

    low_multiplier = 0.0
    low_magnitude = np.zeros_like(radius)
    low_lift = 0.0
    high_multiplier = 1.0
    high_magnitude = _waterfill(rho, distance, radius, high_multiplier)
    high_lift = _lift_for_magnitudes(rho, distance, high_magnitude)
    expansion_iterations = 0
    while high_lift + tolerance < target_lift and high_multiplier < 1e16:
        low_multiplier = high_multiplier
        low_magnitude = high_magnitude
        low_lift = high_lift
        high_multiplier *= 2.0
        high_magnitude = _waterfill(rho, distance, radius, high_multiplier)
        high_lift = _lift_for_magnitudes(rho, distance, high_magnitude)
        expansion_iterations += 1

    if high_lift + tolerance < target_lift:
        # Numerical guard: the exact capacity test above says this should be
        # feasible, so return the cap solution instead of a false success.
        result[support] = radius
        return result, maximum, maximum + tolerance >= target_lift, expansion_iterations

    bisection_iterations = 0
    for _ in range(max_iterations):
        middle_multiplier = 0.5 * (low_multiplier + high_multiplier)
        middle_magnitude = _waterfill(
            rho, distance, radius, middle_multiplier
        )
        middle_lift = _lift_for_magnitudes(rho, distance, middle_magnitude)
        if middle_lift + tolerance >= target_lift:
            high_multiplier = middle_multiplier
            high_magnitude = middle_magnitude
            high_lift = middle_lift
        else:
            low_multiplier = middle_multiplier
            low_magnitude = middle_magnitude
            low_lift = middle_lift
        bisection_iterations += 1
        if (
            abs(high_lift - target_lift) <= tolerance
            or high_multiplier - low_multiplier
            <= 1e-13 * max(1.0, high_multiplier)
        ):
            break

    # A non-concave plane can enter the active set through a finite jump.  Do
    # not accept the overshooting Lagrangian point: interpolate monotonically
    # between the last rejected and accepted vectors and evaluate exact lift.
    if high_lift > target_lift + tolerance:
        alpha_low, alpha_high = 0.0, 1.0
        for _ in range(max_iterations):
            alpha = 0.5 * (alpha_low + alpha_high)
            candidate = low_magnitude + alpha * (
                high_magnitude - low_magnitude
            )
            candidate_lift = _lift_for_magnitudes(rho, distance, candidate)
            if candidate_lift >= target_lift:
                alpha_high = alpha
                high_magnitude = candidate
                high_lift = candidate_lift
            else:
                alpha_low = alpha
                low_magnitude = candidate
                low_lift = candidate_lift
            bisection_iterations += 1
            if abs(high_lift - target_lift) <= tolerance:
                break

    result[support] = np.minimum(radius, np.maximum(0.0, high_magnitude))
    achieved = _lift_for_magnitudes(
        geometry.amplitude, geometry.distance, result
    )
    return (
        result,
        achieved,
        achieved + tolerance >= target_lift,
        expansion_iterations + bisection_iterations,
    )


def _rank_support(
    geometry: _PlaneGeometry,
    active: np.ndarray,
    target_lift: float,
    budget: int,
    tolerance: float,
) -> np.ndarray:
    """Rank by exact target-relevant lift/risk while preserving capacity."""

    if active.size <= budget:
        return active.copy()

    scores: dict[int, float] = {}
    for index in active.tolist():
        amount = min(float(geometry.max_lift[index]), target_lift)
        required = _magnitude_for_lift(
            float(geometry.amplitude[index]),
            float(geometry.distance[index]),
            amount,
        )
        # Squared phase is the intervention risk because it is exactly the
        # objective being minimised by the trust-region solve.
        risk = max(required * required, np.finfo(np.float64).eps)
        scores[index] = amount / risk

    maximum_budget_capacity = float(
        np.sum(
            np.sort(geometry.max_lift[active])[-budget:], dtype=np.float64
        )
    )
    if maximum_budget_capacity + tolerance < target_lift:
        order = sorted(
            active.tolist(),
            key=lambda index: (geometry.max_lift[index], scores[index]),
            reverse=True,
        )
        return np.asarray(order[:budget], dtype=np.int64)

    chosen: list[int] = []
    remaining = set(active.tolist())
    for slot in range(budget):
        slots_after = budget - slot - 1
        viable: list[int] = []
        chosen_capacity = float(np.sum(geometry.max_lift[chosen])) if chosen else 0.0
        for candidate in remaining:
            other_capacity = sorted(
                (
                    float(geometry.max_lift[index])
                    for index in remaining
                    if index != candidate
                ),
                reverse=True,
            )
            possible = (
                chosen_capacity
                + float(geometry.max_lift[candidate])
                + sum(other_capacity[:slots_after])
            )
            if possible + tolerance >= target_lift:
                viable.append(candidate)
        pool = viable if viable else list(remaining)
        selected = max(
            pool,
            key=lambda index: (
                scores[index],
                geometry.max_lift[index],
                -index,
            ),
        )
        chosen.append(selected)
        remaining.remove(selected)
    return np.asarray(chosen, dtype=np.int64)


def _enumerable_supports(
    active: np.ndarray,
    budget: int,
    maximum_combinations: int,
) -> Iterable[np.ndarray] | None:
    cardinality = min(budget, int(active.size))
    count = math.comb(int(active.size), cardinality)
    if count > maximum_combinations:
        return None
    return (
        np.asarray(indices, dtype=np.int64)
        for indices in combinations(active.tolist(), cardinality)
    )


def solve_phase_rescue(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    phase: Sequence[float] | np.ndarray,
    target_lift: float,
    budget: int,
    phase_cap: float | Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
    max_support_combinations: int = 4096,
) -> PhaseRescueResult:
    """Find a strict-constrained phase rescue.

    Sparse-support selection is globally enumerated only when the number of
    combinations is within ``max_support_combinations``.  Otherwise the
    support is heuristic; the nonlinear solve and score re-evaluation on that
    chosen support still use the exact trigonometric objective.

    Args:
        a, b, phase: Per-frequency-plane coefficients and native phases.
        target_lift: Requested increase in the *sum* of plane contributions.
        budget: Maximum number of non-zero frequency-plane shifts.
        phase_cap: Scalar or per-plane absolute phase-shift cap in radians.
        tolerance: Absolute feasibility/lift tolerance.
        max_iterations: Bisection iterations for water filling and exact line
            search.
        max_support_combinations: Enumerate candidate supports when their count
            is at most this value; otherwise use guarded lift/risk ranking.

    Returns:
        :class:`PhaseRescueResult`.  In an infeasible problem the returned
        shifts maximise recoverable lift under the sparse budget and caps,
        ``feasible`` is false, and ``status`` is ``"infeasible_max_lift"``.
    """

    a_array = _as_1d_float(a, "a")
    b_array = _as_1d_float(b, "b")
    phase_array = _as_1d_float(phase, "phase")
    if not (a_array.shape == b_array.shape == phase_array.shape):
        raise ValueError("a, b, and phase must have identical shapes")
    if not math.isfinite(float(target_lift)) or target_lift < 0.0:
        raise ValueError("target_lift must be finite and non-negative")
    if isinstance(budget, bool) or int(budget) != budget or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0 or max_support_combinations <= 0:
        raise ValueError("iteration and enumeration limits must be positive")

    size = int(a_array.size)
    budget = min(int(budget), size)
    cap = _normalise_cap(phase_cap, size)
    geometry = _plane_geometry(a_array, b_array, phase_array, cap)
    zero = np.zeros(size, dtype=np.float64)

    if target_lift <= tolerance:
        return PhaseRescueResult(
            shifts=zero,
            achieved_lift=0.0,
            target_lift=float(target_lift),
            support=(),
            norm=0.0,
            feasible=True,
            max_recoverable_lift=0.0,
            status="no_op",
            iterations=0,
        )

    active = np.flatnonzero(
        (geometry.max_lift > tolerance) & (geometry.radius > tolerance)
    )
    if budget == 0 or active.size == 0:
        return PhaseRescueResult(
            shifts=zero,
            achieved_lift=0.0,
            target_lift=float(target_lift),
            support=(),
            norm=0.0,
            feasible=False,
            max_recoverable_lift=0.0,
            status="infeasible_max_lift",
            iterations=0,
        )

    capacity_order = active[
        np.argsort(geometry.max_lift[active], kind="stable")[::-1]
    ]
    capacity_support = capacity_order[:budget]
    sparse_capacity = float(
        np.sum(geometry.max_lift[capacity_support], dtype=np.float64)
    )
    if sparse_capacity + tolerance < target_lift:
        magnitudes = np.zeros(size, dtype=np.float64)
        magnitudes[capacity_support] = geometry.radius[capacity_support]
        shifts = geometry.direction * magnitudes
        support = tuple(np.flatnonzero(np.abs(shifts) > tolerance).tolist())
        achieved = phase_lift(a_array, b_array, phase_array, shifts)
        return PhaseRescueResult(
            shifts=shifts,
            achieved_lift=achieved,
            target_lift=float(target_lift),
            support=support,
            norm=float(np.linalg.norm(shifts)),
            feasible=False,
            max_recoverable_lift=sparse_capacity,
            status="infeasible_max_lift",
            iterations=0,
        )

    support_iterator = _enumerable_supports(
        active, budget, max_support_combinations
    )
    used_enumeration = support_iterator is not None
    if support_iterator is None:
        support_iterator = [
            _rank_support(
                geometry, active, float(target_lift), budget, tolerance
            )
        ]

    best_magnitudes: np.ndarray | None = None
    best_lift = -math.inf
    best_norm = math.inf
    total_iterations = 0
    for candidate_support in support_iterator:
        candidate_capacity = float(
            np.sum(geometry.max_lift[candidate_support], dtype=np.float64)
        )
        if candidate_capacity + tolerance < target_lift:
            continue
        magnitudes, achieved, feasible, iterations = _solve_fixed_support(
            geometry,
            candidate_support,
            float(target_lift),
            tolerance,
            max_iterations,
        )
        total_iterations += iterations
        candidate_norm = float(np.linalg.norm(magnitudes))
        if feasible and (
            candidate_norm < best_norm - tolerance
            or (
                abs(candidate_norm - best_norm) <= tolerance
                and achieved < best_lift
            )
        ):
            best_magnitudes = magnitudes
            best_lift = achieved
            best_norm = candidate_norm

    if best_magnitudes is None:
        # This is a defensive numerical fallback.  It obeys every hard
        # constraint and makes the failure explicit rather than fabricating a
        # feasible result.
        best_magnitudes = np.zeros(size, dtype=np.float64)
        best_magnitudes[capacity_support] = geometry.radius[capacity_support]
        best_lift = _lift_for_magnitudes(
            geometry.amplitude, geometry.distance, best_magnitudes
        )
        best_norm = float(np.linalg.norm(best_magnitudes))
        feasible = best_lift + tolerance >= target_lift
        status = "capacity_fallback" if feasible else "infeasible_max_lift"
    else:
        feasible = best_lift + tolerance >= target_lift
        status = "solved_enumerated" if used_enumeration else "solved_ranked"

    shifts = geometry.direction * best_magnitudes
    # Re-evaluate from A/B/phase rather than trusting any transformed formula.
    exact_achieved = phase_lift(a_array, b_array, phase_array, shifts)
    support = tuple(np.flatnonzero(np.abs(shifts) > tolerance).tolist())
    if len(support) > budget:
        raise RuntimeError("internal error: sparse phase budget was exceeded")
    feasible = bool(exact_achieved + tolerance >= target_lift and feasible)
    return PhaseRescueResult(
        shifts=shifts,
        achieved_lift=exact_achieved,
        target_lift=float(target_lift),
        support=support,
        norm=float(np.linalg.norm(shifts)),
        feasible=feasible,
        max_recoverable_lift=sparse_capacity,
        status=status,
        iterations=total_iterations,
    )


__all__ = [
    "PhaseRescueResult",
    "phase_lift",
    "phase_scores",
    "solve_phase_rescue",
]
