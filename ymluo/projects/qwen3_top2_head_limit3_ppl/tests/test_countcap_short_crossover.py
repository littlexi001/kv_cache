from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analyze_countcap_short_crossover_20260723 import (
    break_even_steps,
    choose_path,
    estimate_crossover,
    fit_decode_cost,
)


def point(tokens, qk_retention, qk_speed, key_retention, key_speed):
    return {
        "mean_prompt_tokens": tokens,
        "methods": {
            "countcap_fullprompt": {
                "quality_retention": qk_retention,
                "online_speedup": qk_speed,
            },
            "countcap_fullprompt_keypca": {
                "quality_retention": key_retention,
                "online_speedup": key_speed,
            },
        },
    }


def test_quality_floor_prevents_fast_but_unsafe_switch():
    row = point(8192, 0.90, 1.5, 0.97, 1.02)
    assert choose_path(row, quality_floor=0.95, speed_margin=1.03) == (
        "full_kv",
        1.0,
    )


def test_choose_fastest_eligible_sparse_path():
    row = point(32768, 0.98, 1.25, 0.96, 1.40)
    assert choose_path(row, quality_floor=0.95, speed_margin=1.03) == (
        "countcap_fullprompt_keypca",
        1.40,
    )


def test_log_interpolated_crossover_lies_between_measurements():
    rows = [
        point(8192, 0.98, 0.8, 0.98, 0.9),
        point(32768, 0.98, 1.2, 0.98, 1.1),
    ]
    crossover = estimate_crossover(rows, quality_floor=0.95)
    assert crossover is not None
    assert 8192 < crossover < 32768


def test_decode_cost_fit_separates_fixed_and_per_step_cost():
    rows = [
        {"generated_tokens": str(g + 1), "decode_seconds": str(1.5 + 0.04 * g)}
        for g in (4, 8, 16, 32)
    ]
    model = fit_decode_cost(rows)
    assert abs(model["fixed_seconds"] - 1.5) < 1.0e-9
    assert abs(model["step_seconds"] - 0.04) < 1.0e-9
    assert abs(model["r_squared"] - 1.0) < 1.0e-9


def test_break_even_is_undefined_when_sparse_step_is_slower():
    dense = {"fixed_seconds": 0.0, "step_seconds": 0.04}
    sparse = {"fixed_seconds": 1.0, "step_seconds": 0.06}
    assert break_even_steps(dense, sparse) is None


def test_break_even_includes_sparse_fixed_cost():
    dense = {"fixed_seconds": 0.0, "step_seconds": 0.10}
    sparse = {"fixed_seconds": 1.0, "step_seconds": 0.05}
    assert break_even_steps(dense, sparse) == 20
