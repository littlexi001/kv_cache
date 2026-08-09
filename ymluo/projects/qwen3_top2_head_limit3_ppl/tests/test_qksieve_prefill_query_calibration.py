from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_qksieve_prefill_query_calibration_20260804 import (
    clipped_noninferior_gain,
    evaluate_tail_outputs,
    fit_conditional_map,
)


def test_noninferior_gain_recovers_safe_half_scale() -> None:
    prediction = torch.tensor([[2.0, -4.0], [1.0, 3.0]])
    target = 0.5 * prediction

    gain, reduction = clipped_noninferior_gain(prediction, target)

    torch.testing.assert_close(gain, torch.tensor(0.5))
    torch.testing.assert_close(reduction, torch.tensor(1.0))


def test_noninferior_gain_rejects_one_harmful_query() -> None:
    prediction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    gain, _ = clipped_noninferior_gain(prediction, target)

    torch.testing.assert_close(gain, torch.tensor(0.0))


def test_fit_conditional_map_recovers_linear_residual() -> None:
    generator = torch.Generator().manual_seed(7)
    coordinates = torch.randn(64, 3, generator=generator)
    linear_map = torch.randn(5, 3, generator=generator)
    residual = coordinates @ linear_map.T + torch.tensor(
        [1.0, -2.0, 0.5, 0.0, 3.0]
    )

    fitted = fit_conditional_map(coordinates, residual, fit_stride=1)

    torch.testing.assert_close(fitted, linear_map, rtol=2.0e-3, atol=2.0e-3)


def test_tail_outputs_equal_full_when_every_token_is_selected() -> None:
    generator = torch.Generator().manual_seed(11)
    tokens, dimension = 7, 16
    key = torch.randn(tokens, dimension, generator=generator)
    value = torch.randn(tokens, dimension, generator=generator)
    queries = torch.randn(2, 3, dimension, generator=generator)
    coordinates = key[:, :2].clone()
    outputs = evaluate_tail_outputs(
        queries,
        torch.full((2,), tokens, dtype=torch.long),
        key,
        value,
        key.clone(),
        value.clone(),
        coordinates,
        torch.zeros(dimension, 2),
        torch.eye(dimension),
        scaling=dimension**-0.5,
        top_k=tokens,
    )

    for method in ("valuesketch", "residual_mean", "conditional"):
        torch.testing.assert_close(outputs[method], outputs["full"])
