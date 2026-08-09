from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_coselection_expansion import (  # noqa: E402
    build_affinity_graph,
    expand_from_seeds,
    frequency_from_seeds,
    recall,
)


def test_graph_expansion_recovers_held_out_pair_beyond_frequency() -> None:
    training = np.asarray([[0, 7]] * 20 + [[1, 2]] * 10 + [[3, 4]] * 10)
    graph = build_affinity_graph(training, token_count=10, neighbor_count=3)

    graph_candidate = expand_from_seeds(np.asarray([0]), graph, candidate_count=2)
    frequency_candidate = frequency_from_seeds(np.asarray([0]), graph, candidate_count=2)

    assert 7 in graph_candidate
    assert recall(graph_candidate, np.asarray([0, 7])) == 1.0
    assert graph_candidate.size == 2
    assert np.unique(graph_candidate).size == 2
    assert frequency_candidate.size == 2


def test_graph_expansion_has_fixed_budget_and_keeps_seeds() -> None:
    training = np.asarray([[0, 1], [0, 1], [2, 3], [2, 3]], dtype=np.int64)
    graph = build_affinity_graph(training, token_count=8, neighbor_count=2)
    candidate = expand_from_seeds(np.asarray([0, 2]), graph, candidate_count=5)

    assert candidate.size == 5
    assert np.unique(candidate).size == 5
    assert {0, 2}.issubset(candidate.tolist())
