from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _sync_dual_mass_value_metadata,
    _value_residual_squared,
)


def test_metadata_sync_refreshes_open_and_new_blocks_without_reallocation() -> None:
    minimum = torch.full((1, 2, 4, 3), float("nan"), dtype=torch.float16)
    scale = torch.full_like(minimum, float("nan"))
    minimum[..., :2, :] = 1.0
    scale[..., :2, :] = 2.0
    state: dict[str, object] = {}

    cached_minimum, cached_scale = _sync_dual_mass_value_metadata(
        minimum, scale, state, active_block_count=2
    )
    torch.testing.assert_close(
        cached_minimum[..., :2, :], minimum[..., :2, :].float()
    )
    torch.testing.assert_close(
        cached_scale[..., :2, :], scale[..., :2, :].float()
    )

    minimum[..., 1, :] = 3.0
    scale[..., 1, :] = 4.0
    minimum[..., 2, :] = 5.0
    scale[..., 2, :] = 6.0
    second_minimum, second_scale = _sync_dual_mass_value_metadata(
        minimum, scale, state, active_block_count=3
    )

    assert second_minimum.data_ptr() == cached_minimum.data_ptr()
    assert second_scale.data_ptr() == cached_scale.data_ptr()
    torch.testing.assert_close(
        second_minimum[..., 1, :], minimum[..., 1, :].float()
    )
    torch.testing.assert_close(second_scale[..., 1, :], scale[..., 1, :].float())
    torch.testing.assert_close(
        second_minimum[..., 2, :], minimum[..., 2, :].float()
    )
    torch.testing.assert_close(second_scale[..., 2, :], scale[..., 2, :].float())


def test_diagonal_residual_metric_matches_full_metric_for_diagonal_gram() -> None:
    generator = torch.Generator().manual_seed(17)
    residual = torch.randn(2, 3, 11, 5, generator=generator)
    diagonal = torch.rand(3, 5, generator=generator)
    gram = torch.diag_embed(diagonal)

    exact = _value_residual_squared(residual, gram, "full")
    approximate = _value_residual_squared(residual, gram, "diagonal")

    torch.testing.assert_close(approximate, exact)
