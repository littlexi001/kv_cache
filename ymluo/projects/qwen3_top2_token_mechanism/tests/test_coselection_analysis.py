from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coselection_analysis import (  # noqa: E402
    benjamini_hochberg,
    cooccurrence_from_incidence,
    extract_pair_edges,
    incidence_from_indices,
    pair_metric_matrices,
    summarize_head,
)


def test_pair_conditional_and_lift_capture_deterministic_coselection() -> None:
    indices = np.asarray(
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [2, 3],
            [2, 3],
        ],
        dtype=np.int64,
    )
    incidence = incidence_from_indices(indices, token_count=4)
    cooccurrence = cooccurrence_from_incidence(incidence)
    count = np.diag(cooccurrence)
    metrics = pair_metric_matrices(count, cooccurrence, observations=8)

    assert cooccurrence[0, 1] == 4
    assert metrics["conditional"][0, 1] == 1.0
    assert metrics["conditional"][1, 0] == 1.0
    assert metrics["lift"][0, 1] == 2.0
    assert metrics["phi"][0, 1] == 1.0


def test_edge_significance_and_components_find_repeated_pair_groups() -> None:
    rows = []
    rows.extend([[0, 1]] * 32)
    rows.extend([[2, 3]] * 32)
    incidence = incidence_from_indices(np.asarray(rows), token_count=8)
    cooccurrence = cooccurrence_from_incidence(incidence)
    count = np.diag(cooccurrence)
    edges = extract_pair_edges(
        count,
        cooccurrence,
        observations=64,
        min_token_count=4,
        min_pair_count=4,
        fdr_alpha=0.01,
    )
    summary = summarize_head(count, cooccurrence, 64, 2, edges)

    significant_pairs = {(edge.token_a, edge.token_b) for edge in edges if edge.significant}
    assert significant_pairs == {(0, 1), (2, 3)}
    assert summary["component_count"] == 2
    assert summary["largest_component_tokens"] == 2
    assert summary["significant_conditional_median"] == 1.0
    assert summary["uniform_fixed_budget_conditional"] == 1 / 7


def test_bh_can_account_for_unmaterialized_unit_p_values() -> None:
    adjusted = benjamini_hochberg(np.asarray([0.0001, 0.01]), total_hypotheses=100)
    assert np.allclose(adjusted, [0.01, 0.5])


def test_near_position_enrichment_detects_local_pairing() -> None:
    indices = np.asarray([[0, 1]] * 20 + [[4, 5]] * 20, dtype=np.int64)
    incidence = incidence_from_indices(indices, token_count=8)
    cooccurrence = cooccurrence_from_incidence(incidence)
    count = np.diag(cooccurrence)
    edges = extract_pair_edges(
        count,
        cooccurrence,
        observations=40,
        min_token_count=4,
        min_pair_count=4,
    )
    summary = summarize_head(count, cooccurrence, 40, 2, edges)
    assert summary["distance_le_1_enrichment"] > 1.0
