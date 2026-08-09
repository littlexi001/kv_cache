from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qksieve_valuesketch_cuda_20260801 import (  # noqa: E402
    allocate_attention_workspace,
    attention_split_count,
)


@pytest.mark.parametrize(
    ("candidate_capacity", "expected"),
    [(1, 8), (4_096, 8), (4_097, 4), (49_000, 8)],
)
def test_attention_split_count_matches_cuda_policy(
    candidate_capacity: int,
    expected: int,
) -> None:
    assert attention_split_count(candidate_capacity) == expected


def test_allocate_attention_workspace_has_stable_contiguous_views() -> None:
    query = torch.empty(2, 32, 128, dtype=torch.float16)

    workspace = allocate_attention_workspace(query, 5_000)

    assert workspace["output"].shape == (2, 32, 128)
    assert workspace["output_view"].shape == (2, 1, 32, 128)
    assert workspace["output"].data_ptr() == workspace["output_view"].data_ptr()
    assert workspace["output"].is_contiguous()
    assert workspace["output_view"].is_contiguous()
    assert workspace["partial_output"].shape == (64, 4, 128)
    assert workspace["partial_maximum"].shape == (64, 4)
    assert workspace["partial_sum"].shape == (64, 4)
    assert workspace["partial_output"].dtype == torch.float32


def test_attention_split_count_rejects_empty_capacity() -> None:
    with pytest.raises(ValueError, match="positive"):
        attention_split_count(0)
