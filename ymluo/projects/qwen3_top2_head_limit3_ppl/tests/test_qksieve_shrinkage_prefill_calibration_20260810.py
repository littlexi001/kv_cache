from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_qk_balanced_spectral_rate_20260727 import (  # noqa: E402
    resolve_calibration_and_evaluation,
)


def records(count: int) -> list[dict[str, object]]:
    return [
        {
            "step": step,
            "query": torch.full((1, 4, 1, 3), float(step)),
        }
        for step in range(count)
    ]


def test_prefill_tail_uses_prompt_queries_and_keeps_all_decode_queries() -> None:
    prompt = torch.arange(1 * 4 * 6 * 3, dtype=torch.float32).reshape(
        1, 4, 6, 3
    )
    calibration, evaluation, start = resolve_calibration_and_evaluation(
        {"prefill_queries": {7: prompt}},
        7,
        records(5),
        4,
        "prefill_tail",
        torch.device("cpu"),
    )

    expected = prompt[0, :, -4:, :].permute(1, 0, 2)
    assert torch.equal(calibration, expected)
    assert len(evaluation) == 5
    assert start == 0


def test_decode_prefix_preserves_legacy_split() -> None:
    calibration, evaluation, start = resolve_calibration_and_evaluation(
        {},
        7,
        records(5),
        2,
        "decode_prefix",
        torch.device("cpu"),
    )

    assert calibration.shape == (2, 4, 3)
    assert [int(row["step"]) for row in evaluation] == [2, 3, 4]
    assert start == 2


def test_prefill_tail_rejects_missing_prompt_queries() -> None:
    with pytest.raises(ValueError, match="no captured prefill"):
        resolve_calibration_and_evaluation(
            {},
            7,
            records(5),
            2,
            "prefill_tail",
            torch.device("cpu"),
        )
