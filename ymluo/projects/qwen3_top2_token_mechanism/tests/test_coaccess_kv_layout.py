from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_coaccess_kv_layout import (  # noqa: E402
    complete_query_head_groups,
    coaccess_physical_positions,
    page_counts,
    union_query_heads,
)


def test_union_query_heads_combines_shared_kv_accesses() -> None:
    indices = np.asarray(
        [
            [[[0, 1], [2, 3]]],
            [[[1, 4], [3, 5]]],
        ],
        dtype=np.int32,
    ).reshape(2, 2, 2)
    rows = union_query_heads(indices)

    assert rows[0].tolist() == [0, 1, 4]
    assert rows[1].tolist() == [2, 3, 5]


def test_complete_query_head_groups_ignores_incomplete_groups() -> None:
    groups = complete_query_head_groups(np.asarray([1, 10, 12, 13]), group_size=2)

    assert len(groups) == 1
    assert groups[0].tolist() == [2, 3]


def test_coaccess_layout_is_permutation_and_reduces_repeated_far_pair_pages() -> None:
    training = [np.asarray([0, 8]), np.asarray([0, 8]), np.asarray([0, 8]), np.asarray([4, 12])]
    positions = coaccess_physical_positions(
        training,
        token_count=16,
        page_size=4,
        microblock_size=1,
        neighbor_count=4,
    )
    test = [np.asarray([0, 8]), np.asarray([0, 8])]

    assert np.unique(positions).size == 16
    assert page_counts(test, positions, 4).mean() < page_counts(
        test, np.arange(16), 4
    ).mean()
