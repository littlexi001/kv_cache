from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_longbench_prompt_protocol_20260727 import (
    common_prefix_length,
    common_suffix_length,
)


def test_common_edge_lengths_stop_at_first_mismatch() -> None:
    left = [1, 2, 3, 4, 5]
    right = [1, 2, 9, 4, 5]

    assert common_prefix_length(left, right) == 2
    assert common_suffix_length(left, right) == 2


def test_common_edge_lengths_handle_unequal_sequences() -> None:
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert common_suffix_length([2, 3], [1, 2, 3]) == 2
