from pathlib import Path
import sys

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_head_top2_targeted_ppl_20260714 import _pca_int4_partial_scores


def test_pca_int4_index_grows_past_initial_decode_reserve():
    torch.manual_seed(7)
    state = {}
    query = torch.randn(1, 1, 1, 16)
    first_key = torch.randn(1, 1, 4, 16)
    first = _pca_int4_partial_scores(query, first_key, state, projection_dim=16)
    initial_capacity = int(state["capacity"])

    grown_key = torch.randn(1, 1, initial_capacity + 1, 16)
    second = _pca_int4_partial_scores(query, grown_key, state, projection_dim=16)

    assert first.shape == (1, 1, 4)
    assert second.shape == (1, 1, initial_capacity + 1)
    assert int(state["capacity"]) >= initial_capacity + 1
    assert int(state["indexed_count"]) == initial_capacity + 1
    assert int(state["capacity_growth_count"]) == 1
