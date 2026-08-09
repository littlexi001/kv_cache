from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.stats import hypergeom


@dataclass(frozen=True)
class PairEdge:
    token_a: int
    token_b: int
    count_a: int
    count_b: int
    cooccurrence: int
    conditional_b_given_a: float
    conditional_a_given_b: float
    lift: float
    phi: float
    jaccard: float
    p_value: float
    q_value: float
    significant: bool


def incidence_from_indices(indices: np.ndarray, token_count: int) -> np.ndarray:
    """Build a query-by-token selection matrix from fixed-budget Top-k indices."""
    values = np.asarray(indices)
    if values.ndim != 2:
        raise ValueError("indices must have shape [observations, budget].")
    if token_count <= 0:
        raise ValueError("token_count must be positive.")
    if values.size and (values.min() < 0 or values.max() >= token_count):
        raise ValueError("selection index is outside token_count.")
    incidence = np.zeros((values.shape[0], token_count), dtype=np.uint8)
    if values.size:
        rows = np.repeat(np.arange(values.shape[0]), values.shape[1])
        incidence[rows, values.reshape(-1)] = 1
    return incidence


def cooccurrence_from_incidence(incidence: np.ndarray) -> np.ndarray:
    values = np.asarray(incidence)
    if values.ndim != 2:
        raise ValueError("incidence must have shape [observations, tokens].")
    integer = values.astype(np.int32, copy=False)
    return integer.T @ integer


def pair_metric_matrices(
    selection_count: np.ndarray,
    cooccurrence: np.ndarray,
    observations: int,
) -> dict[str, np.ndarray]:
    """Return directed conditional probability and symmetric association matrices."""
    count = np.asarray(selection_count, dtype=np.float64)
    coocc = np.asarray(cooccurrence, dtype=np.float64)
    if coocc.shape != (count.size, count.size):
        raise ValueError("cooccurrence shape must match selection_count.")
    if observations <= 0:
        raise ValueError("observations must be positive.")

    denom_cond = count[:, None]
    conditional = np.divide(coocc, denom_cond, out=np.zeros_like(coocc), where=denom_cond > 0)

    product = count[:, None] * count[None, :]
    lift = np.divide(
        coocc * observations,
        product,
        out=np.zeros_like(coocc),
        where=product > 0,
    )
    union = count[:, None] + count[None, :] - coocc
    jaccard = np.divide(coocc, union, out=np.zeros_like(coocc), where=union > 0)

    phi_denom = np.sqrt(
        count[:, None]
        * (observations - count[:, None])
        * count[None, :]
        * (observations - count[None, :])
    )
    phi = np.divide(
        observations * coocc - product,
        phi_denom,
        out=np.zeros_like(coocc),
        where=phi_denom > 0,
    )
    np.fill_diagonal(conditional, 0.0)
    np.fill_diagonal(lift, 0.0)
    np.fill_diagonal(jaccard, 0.0)
    np.fill_diagonal(phi, 0.0)
    return {
        "conditional": conditional,
        "lift": lift,
        "jaccard": jaccard,
        "phi": phi,
    }


def benjamini_hochberg(p_values: np.ndarray, total_hypotheses: int | None = None) -> np.ndarray:
    """BH adjusted p-values; total_hypotheses may include omitted p=1 tests."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if p.size == 0:
        return p.copy()
    total = int(total_hypotheses if total_hypotheses is not None else p.size)
    if total < p.size:
        raise ValueError("total_hypotheses cannot be smaller than observed p-values.")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * total / np.arange(1, p.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def extract_pair_edges(
    selection_count: np.ndarray,
    cooccurrence: np.ndarray,
    observations: int,
    *,
    min_token_count: int = 8,
    min_pair_count: int = 4,
    fdr_alpha: float = 0.01,
) -> list[PairEdge]:
    """Find positively associated token pairs using a marginal-conditioned null."""
    count = np.asarray(selection_count, dtype=np.int64)
    coocc = np.asarray(cooccurrence, dtype=np.int64)
    if coocc.shape != (count.size, count.size):
        raise ValueError("cooccurrence shape must match selection_count.")

    candidate = np.triu(coocc >= min_pair_count, k=1)
    active = count >= min_token_count
    candidate &= active[:, None] & active[None, :]
    token_a, token_b = np.nonzero(candidate)
    if token_a.size == 0:
        return []

    pair_count = coocc[token_a, token_b]
    count_a = count[token_a]
    count_b = count[token_b]
    p_values = hypergeom.sf(pair_count - 1, observations, count_a, count_b)
    total_pairs = count.size * (count.size - 1) // 2
    q_values = benjamini_hochberg(p_values, total_pairs)

    conditional_b_given_a = pair_count / count_a
    conditional_a_given_b = pair_count / count_b
    lift = pair_count * observations / (count_a * count_b)
    union = count_a + count_b - pair_count
    jaccard = pair_count / union
    denom = np.sqrt(count_a * (observations - count_a) * count_b * (observations - count_b))
    phi = np.divide(
        observations * pair_count - count_a * count_b,
        denom,
        out=np.zeros_like(pair_count, dtype=np.float64),
        where=denom > 0,
    )

    edges = [
        PairEdge(
            token_a=int(token_a[index]),
            token_b=int(token_b[index]),
            count_a=int(count_a[index]),
            count_b=int(count_b[index]),
            cooccurrence=int(pair_count[index]),
            conditional_b_given_a=float(conditional_b_given_a[index]),
            conditional_a_given_b=float(conditional_a_given_b[index]),
            lift=float(lift[index]),
            phi=float(phi[index]),
            jaccard=float(jaccard[index]),
            p_value=float(p_values[index]),
            q_value=float(q_values[index]),
            significant=bool(q_values[index] <= fdr_alpha and phi[index] > 0.0),
        )
        for index in range(token_a.size)
    ]
    edges.sort(key=lambda edge: (not edge.significant, edge.q_value, -edge.phi, -edge.cooccurrence))
    return edges


class _UnionFind:
    def __init__(self, nodes: Iterable[int]) -> None:
        self.parent = {int(node): int(node) for node in nodes}
        self.size = {int(node): 1 for node in nodes}

    def find(self, node: int) -> int:
        parent = self.parent[node]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while node != parent:
            next_node = self.parent[node]
            self.parent[node] = parent
            node = next_node
        return parent

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]

    def component_sizes(self) -> list[int]:
        counts: dict[int, int] = {}
        for node in self.parent:
            root = self.find(node)
            counts[root] = counts.get(root, 0) + 1
        return sorted(counts.values(), reverse=True)


def component_summary(edges: Iterable[PairEdge]) -> dict[str, int | float]:
    significant = [edge for edge in edges if edge.significant]
    nodes = {edge.token_a for edge in significant} | {edge.token_b for edge in significant}
    if not nodes:
        return {
            "graph_nodes": 0,
            "graph_edges": 0,
            "component_count": 0,
            "largest_component_tokens": 0,
            "largest_component_fraction": 0.0,
        }
    union_find = _UnionFind(nodes)
    for edge in significant:
        union_find.union(edge.token_a, edge.token_b)
    sizes = union_find.component_sizes()
    return {
        "graph_nodes": len(nodes),
        "graph_edges": len(significant),
        "component_count": len(sizes),
        "largest_component_tokens": sizes[0],
        "largest_component_fraction": sizes[0] / len(nodes),
    }


def _near_pair_mass(
    cooccurrence: np.ndarray,
    selection_count: np.ndarray,
    observations: int,
    window: int,
) -> tuple[float, float, float]:
    token_count = selection_count.size
    max_offset = min(max(0, window), token_count - 1)
    observed_near = 0.0
    expected_near = 0.0
    for offset in range(1, max_offset + 1):
        observed_near += float(np.diagonal(cooccurrence, offset=offset).sum())
        expected_near += float(
            np.dot(selection_count[:-offset].astype(np.float64), selection_count[offset:].astype(np.float64))
            / observations
        )
    upper = np.triu(cooccurrence, k=1)
    observed_total = float(upper.sum())
    event_total = float(selection_count.sum())
    expected_total = (
        event_total * event_total - float(np.dot(selection_count, selection_count))
    ) / (2.0 * observations)
    observed_share = observed_near / observed_total if observed_total > 0 else 0.0
    expected_share = expected_near / expected_total if expected_total > 0 else 0.0
    enrichment = observed_share / expected_share if expected_share > 0 else 0.0
    return observed_share, expected_share, enrichment


def summarize_head(
    selection_count: np.ndarray,
    cooccurrence: np.ndarray,
    observations: int,
    budget: int,
    edges: list[PairEdge],
) -> dict[str, Any]:
    count = np.asarray(selection_count, dtype=np.int64)
    coocc = np.asarray(cooccurrence, dtype=np.int64)
    token_count = count.size
    if observations <= 0 or budget <= 0:
        raise ValueError("observations and budget must be positive.")

    upper_i, upper_j = np.nonzero(np.triu(coocc > 0, k=1))
    observed_pair_count = coocc[upper_i, upper_j].astype(np.float64)
    expected_pair_count = count[upper_i] * count[upper_j] / observations
    positive_excess = np.maximum(observed_pair_count - expected_pair_count, 0.0)
    observed_pair_mass = float(observed_pair_count.sum())
    excess_fraction = float(positive_excess.sum() / observed_pair_mass) if observed_pair_mass else 0.0

    selection_probability = count / observations
    event_distribution = count.astype(np.float64)
    if event_distribution.sum() > 0:
        event_distribution /= event_distribution.sum()
        nonzero = event_distribution[event_distribution > 0]
        effective_support = float(math.exp(-float(np.sum(nonzero * np.log(nonzero)))))
    else:
        effective_support = 0.0

    significant = [edge for edge in edges if edge.significant]
    component = component_summary(edges)
    significant_phi = np.asarray([edge.phi for edge in significant], dtype=np.float64)
    significant_lift = np.asarray([edge.lift for edge in significant], dtype=np.float64)
    significant_conditional = np.asarray(
        [max(edge.conditional_b_given_a, edge.conditional_a_given_b) for edge in significant],
        dtype=np.float64,
    )

    row: dict[str, Any] = {
        "observations": observations,
        "token_count": token_count,
        "budget": budget,
        "uniform_fixed_budget_conditional": (budget - 1) / (token_count - 1) if token_count > 1 else 0.0,
        "active_tokens": int(np.count_nonzero(count)),
        "always_selected_tokens": int(np.count_nonzero(count == observations)),
        "selected_ge_50pct_tokens": int(np.count_nonzero(selection_probability >= 0.5)),
        "max_token_selection_probability": float(selection_probability.max(initial=0.0)),
        "token_effective_support": effective_support,
        "unique_coselected_pairs": int(upper_i.size),
        "pair_events": int(observed_pair_mass),
        "positive_excess_pair_mass_fraction": excess_fraction,
        "tested_candidate_pairs": len(edges),
        "significant_positive_pairs": len(significant),
        "significant_fraction_of_coobserved_pairs": len(significant) / upper_i.size if upper_i.size else 0.0,
        "significant_phi_mean": float(significant_phi.mean()) if significant_phi.size else 0.0,
        "significant_phi_max": float(significant_phi.max()) if significant_phi.size else 0.0,
        "significant_lift_median": float(np.median(significant_lift)) if significant_lift.size else 0.0,
        "significant_lift_max": float(significant_lift.max()) if significant_lift.size else 0.0,
        "significant_conditional_median": float(np.median(significant_conditional))
        if significant_conditional.size
        else 0.0,
        "significant_conditional_max": float(significant_conditional.max())
        if significant_conditional.size
        else 0.0,
        **component,
    }
    for window in (1, 4, 16, 64):
        observed_share, expected_share, enrichment = _near_pair_mass(
            coocc, count, observations, window
        )
        row[f"distance_le_{window}_observed_pair_share"] = observed_share
        row[f"distance_le_{window}_marginal_null_share"] = expected_share
        row[f"distance_le_{window}_enrichment"] = enrichment

    row["cluster_score"] = excess_fraction * math.log1p(len(significant))
    return row


def edge_to_dict(edge: PairEdge) -> dict[str, Any]:
    return {
        "token_a": edge.token_a,
        "token_b": edge.token_b,
        "count_a": edge.count_a,
        "count_b": edge.count_b,
        "cooccurrence": edge.cooccurrence,
        "conditional_b_given_a": edge.conditional_b_given_a,
        "conditional_a_given_b": edge.conditional_a_given_b,
        "lift": edge.lift,
        "phi": edge.phi,
        "jaccard": edge.jaccard,
        "p_value": edge.p_value,
        "q_value": edge.q_value,
        "significant": int(edge.significant),
    }
