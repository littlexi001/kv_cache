from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_top_head_category_nll_ablation import matched_random_keys, merge_probes


def test_merge_probes_unions_keys_and_drops_terminal_query() -> None:
    probes = [(2, (0, 1)), (2, (1, 2)), (4, (0,))]
    assert merge_probes(probes, token_count=5) == {2: (0, 1, 2)}


def test_matched_random_is_deterministic_disjoint_and_size_matched() -> None:
    links = {5: (0, 2), 8: (1, 3, 5)}
    first = matched_random_keys(links, seed=17)
    second = matched_random_keys(links, seed=17)
    assert first == second
    for query, target in links.items():
        assert len(first[query]) == len(target)
        assert set(first[query]).isdisjoint(target)
        assert all(0 <= key <= query for key in first[query])
