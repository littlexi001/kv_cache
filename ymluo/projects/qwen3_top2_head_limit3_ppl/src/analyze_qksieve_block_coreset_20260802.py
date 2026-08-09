#!/usr/bin/env python
"""Small blockwise joint K/V coresets for omitted-attention completion."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from analyze_qksieve_conditional_value_moments_20260802 import symmetric_quantize


def _farthest_point_initialization(
    features: torch.Tensor,
    cluster_count: int,
) -> torch.Tensor:
    first = torch.argmax(features.square().sum(dim=-1))
    chosen = [int(first)]
    minimum_distance = (features - features[first]).square().sum(dim=-1)
    for _ in range(1, cluster_count):
        index = int(torch.argmax(minimum_distance))
        chosen.append(index)
        distance = (features - features[index]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    return features[torch.tensor(chosen, device=features.device)].clone()


def _lloyd_assignments(
    features: torch.Tensor,
    cluster_count: int,
    iterations: int,
) -> torch.Tensor:
    if cluster_count <= 0 or cluster_count > features.shape[0]:
        raise ValueError("cluster count must be in [1, block token count]")
    centroids = _farthest_point_initialization(features, cluster_count)
    assignments = torch.zeros(
        features.shape[0], device=features.device, dtype=torch.long
    )
    for _ in range(iterations):
        distances = torch.cdist(features.float(), centroids.float()).square()
        assignments = distances.argmin(dim=-1)
        minimum_distance = distances.amin(dim=-1)
        updated = []
        for cluster in range(cluster_count):
            members = assignments == cluster
            if members.any():
                updated.append(features[members].mean(dim=0))
            else:
                replacement = int(torch.argmax(minimum_distance))
                updated.append(features[replacement])
                minimum_distance[replacement] = -1.0
        next_centroids = torch.stack(updated)
        if torch.equal(next_centroids, centroids):
            break
        centroids = next_centroids
    return assignments


def fit_block_coreset(
    score_coordinates: torch.Tensor,
    values: torch.Tensor,
    block_size: int,
    cluster_count: int,
    moment_bits: int = 8,
    iterations: int = 6,
    value_projection_dim: int = 0,
    value_weight: float = 0.0,
    full_score_coordinates: torch.Tensor | None = None,
    value_moment_bits: int | None = None,
    full_score_moment_bits: int | None = None,
) -> dict[str, torch.Tensor | int | float]:
    """Fit deterministic per-block K or joint K/V prototypes."""
    if score_coordinates.shape[0] != values.shape[0]:
        raise ValueError("score coordinates and Values must share token count")
    if block_size <= 0 or cluster_count <= 0 or iterations <= 0:
        raise ValueError("block, cluster, and iteration counts must be positive")
    token_count, score_dim = score_coordinates.shape
    value_dim = values.shape[-1]
    if (
        full_score_coordinates is not None
        and full_score_coordinates.shape[0] != token_count
    ):
        raise ValueError("full score coordinates must share token count")
    full_score_dim = (
        int(full_score_coordinates.shape[-1])
        if full_score_coordinates is not None
        else 0
    )
    value_moment_bits = (
        moment_bits if value_moment_bits is None else value_moment_bits
    )
    full_score_moment_bits = (
        moment_bits
        if full_score_moment_bits is None
        else full_score_moment_bits
    )
    for name, bits in (
        ("score", moment_bits),
        ("Value", value_moment_bits),
        ("full-Key", full_score_moment_bits),
    ):
        if bits < 2:
            raise ValueError(f"{name} prototype quantization needs at least 2 bits")
    block_count = math.ceil(token_count / block_size)
    value_basis: torch.Tensor | None = None
    if value_weight > 0.0 and value_projection_dim > 0:
        value_projection_dim = min(value_projection_dim, value_dim)
        gram = values.float().T @ values.float()
        _, eigenvectors = torch.linalg.eigh(gram)
        value_basis = eigenvectors[:, -value_projection_dim:].flip(dims=(1,))
        value_features = values.float() @ value_basis
    else:
        value_features = values.new_zeros((token_count, 0)).float()
        value_projection_dim = 0

    assignments = torch.empty(token_count, dtype=torch.long, device=values.device)
    means_x = torch.zeros(
        block_count,
        cluster_count,
        score_dim,
        dtype=torch.float32,
        device=values.device,
    )
    means_v = torch.zeros(
        block_count,
        cluster_count,
        value_dim,
        dtype=torch.float32,
        device=values.device,
    )
    means_full_score = (
        torch.zeros(
            block_count,
            cluster_count,
            full_score_dim,
            dtype=torch.float32,
            device=values.device,
        )
        if full_score_coordinates is not None
        else None
    )
    counts = torch.zeros(
        block_count,
        cluster_count,
        dtype=torch.float32,
        device=values.device,
    )
    for block in range(block_count):
        start = block * block_size
        stop = min(token_count, start + block_size)
        block_x = score_coordinates[start:stop].float()
        x_scale = block_x.square().mean().sqrt().clamp_min(1.0e-8)
        features = [block_x / x_scale]
        if value_projection_dim:
            block_v_feature = value_features[start:stop]
            v_scale = block_v_feature.square().mean().sqrt().clamp_min(1.0e-8)
            features.append(float(value_weight) * block_v_feature / v_scale)
        block_features = torch.cat(features, dim=-1)
        block_assignments = _lloyd_assignments(
            block_features, min(cluster_count, stop - start), iterations
        )
        assignments[start:stop] = block_assignments
        block_values = values[start:stop].float()
        for cluster in range(cluster_count):
            members = block_assignments == cluster
            count = int(members.sum())
            if count == 0:
                continue
            counts[block, cluster] = float(count)
            means_x[block, cluster] = block_x[members].mean(dim=0)
            means_v[block, cluster] = block_values[members].mean(dim=0)
            if means_full_score is not None:
                assert full_score_coordinates is not None
                means_full_score[block, cluster] = full_score_coordinates[
                    start:stop
                ][members].float().mean(dim=0)

    means_x = symmetric_quantize(means_x, moment_bits, (2,))
    means_v = symmetric_quantize(means_v, value_moment_bits, (2,))
    if means_full_score is not None:
        means_full_score = symmetric_quantize(
            means_full_score, full_score_moment_bits, (2,)
        )
    assignment_bits = math.ceil(math.log2(cluster_count))
    prototype_bits = (
        block_count
        * cluster_count
        * (
            moment_bits * score_dim
            + value_moment_bits * value_dim
            + full_score_moment_bits * full_score_dim
        )
    )
    score_scale_bits = (
        16 * block_count * cluster_count if moment_bits < 16 else 0
    )
    value_scale_bits = (
        16 * block_count * cluster_count if value_moment_bits < 16 else 0
    )
    full_score_scale_bits = (
        16 * block_count * cluster_count
        if full_score_dim > 0 and full_score_moment_bits < 16
        else 0
    )
    scale_bits = score_scale_bits + value_scale_bits + full_score_scale_bits
    count_bits = (
        math.ceil(math.log2(block_size + 1)) * block_count * cluster_count
    )
    projection_bits = (
        value_dim * value_projection_dim * 16 if value_basis is not None else 0
    )
    stored_bits = (
        assignment_bits * token_count
        + prototype_bits
        + scale_bits
        + count_bits
        + projection_bits
    )
    shared_metadata_bits = (
        assignment_bits * token_count + count_bits + projection_bits
    )
    proxy_tail_bits = (
        shared_metadata_bits
        + block_count
        * cluster_count
        * (moment_bits * score_dim + value_moment_bits * value_dim)
        + score_scale_bits
        + value_scale_bits
    )
    full_tail_bits = (
        shared_metadata_bits
        + block_count
        * cluster_count
        * (
            full_score_moment_bits * full_score_dim
            + value_moment_bits * value_dim
        )
        + full_score_scale_bits
        + value_scale_bits
    )
    result: dict[str, torch.Tensor | int | float] = {
        "assignments": assignments,
        "mean_x": means_x,
        "mean_v": means_v,
        "counts": counts,
        "block_size": block_size,
        "block_count": block_count,
        "cluster_count": cluster_count,
        "score_dim": score_dim,
        "value_dim": value_dim,
        "moment_bits": moment_bits,
        "value_moment_bits": value_moment_bits,
        "full_score_moment_bits": full_score_moment_bits,
        "assignment_bits": assignment_bits,
        "value_projection_dim": value_projection_dim,
        "value_weight": float(value_weight),
        "full_score_dim": full_score_dim,
        "bits_per_token": stored_bits / token_count,
        "proxy_tail_bits_per_token": proxy_tail_bits / token_count,
        "full_tail_bits_per_token": full_tail_bits / token_count,
    }
    if value_basis is not None:
        result["value_basis"] = value_basis
    if means_full_score is not None:
        result["mean_full_score"] = means_full_score
    return result


def coreset_corrected_proxy_scores(
    proxy_scores: torch.Tensor,
    proxy_direction: torch.Tensor,
    full_score_direction: torch.Tensor,
    model: dict[str, torch.Tensor | int | float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Correct token proxy scores with a query-dependent cluster residual."""
    assignments = model["assignments"]
    mean_x = model["mean_x"]
    mean_full_score = model.get("mean_full_score")
    assert isinstance(assignments, torch.Tensor)
    assert isinstance(mean_x, torch.Tensor)
    if not isinstance(mean_full_score, torch.Tensor):
        raise ValueError("coreset does not store full-score prototypes")
    block_size = int(model["block_size"])
    cluster_count = int(model["cluster_count"])
    token_count = proxy_scores.numel()
    block_ids = torch.arange(token_count, device=proxy_scores.device) // block_size
    proxy_centroid_scores = torch.einsum(
        "bcr,r->bc", mean_x.float(), proxy_direction.float()
    )
    full_centroid_scores = torch.einsum(
        "bcd,d->bc", mean_full_score.float(), full_score_direction.float()
    )
    correction = full_centroid_scores - proxy_centroid_scores
    flat_ids = block_ids * cluster_count + assignments.long()
    token_correction = correction.reshape(-1).index_select(0, flat_ids)
    diagnostics = {
        "correction_abs_mean": float(token_correction.abs().mean()),
        "correction_abs_maximum": float(token_correction.abs().max()),
    }
    return proxy_scores.float() + token_correction, diagnostics


def block_coreset_tail_statistics(
    score_coordinates: torch.Tensor,
    values: torch.Tensor,
    score_direction: torch.Tensor,
    selected: torch.Tensor,
    reference: torch.Tensor | float,
    model: dict[str, torch.Tensor | int | float],
    selected_conditioned: bool = True,
    full_score_coordinates: torch.Tensor | None = None,
    full_score_direction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Approximate omitted softmax totals from bounded-support prototypes."""
    assignments = model["assignments"]
    mean_x = model["mean_x"]
    mean_v = model["mean_v"]
    counts = model["counts"]
    assert isinstance(assignments, torch.Tensor)
    assert isinstance(mean_x, torch.Tensor)
    assert isinstance(mean_v, torch.Tensor)
    assert isinstance(counts, torch.Tensor)
    block_size = int(model["block_size"])
    block_count = int(model["block_count"])
    cluster_count = int(model["cluster_count"])
    selected = selected.long()
    selected_blocks = selected // block_size
    selected_clusters = assignments.index_select(0, selected).long()
    selected_flat = selected_blocks * cluster_count + selected_clusters
    remaining_counts = counts.float().reshape(-1).clone()
    remaining_counts.index_add_(
        0,
        selected_flat,
        -torch.ones_like(selected_flat, dtype=torch.float32),
    )
    remaining_counts = remaining_counts.reshape(block_count, cluster_count)
    tail_mean_x = mean_x.float().clone()
    tail_mean_v = mean_v.float().clone()
    mean_full_score = model.get("mean_full_score")
    tail_mean_full_score = (
        mean_full_score.float().clone()
        if isinstance(mean_full_score, torch.Tensor)
        else None
    )
    if selected_conditioned:
        sum_x = counts.float()[..., None] * tail_mean_x
        sum_v = counts.float()[..., None] * tail_mean_v
        flat_sum_x = sum_x.reshape(-1, sum_x.shape[-1])
        flat_sum_v = sum_v.reshape(-1, sum_v.shape[-1])
        flat_sum_x.index_add_(
            0,
            selected_flat,
            -score_coordinates.index_select(0, selected).float(),
        )
        flat_sum_v.index_add_(
            0,
            selected_flat,
            -values.index_select(0, selected).float(),
        )
        tail_mean_x = flat_sum_x.reshape_as(sum_x) / remaining_counts[
            ..., None
        ].clamp_min(1.0)
        tail_mean_v = flat_sum_v.reshape_as(sum_v) / remaining_counts[
            ..., None
        ].clamp_min(1.0)
        if tail_mean_full_score is not None and full_score_direction is not None:
            if full_score_coordinates is None:
                raise ValueError(
                    "selected conditioning of full-score prototypes needs full coordinates"
                )
            sum_full = counts.float()[..., None] * tail_mean_full_score
            flat_sum_full = sum_full.reshape(-1, sum_full.shape[-1])
            flat_sum_full.index_add_(
                0,
                selected_flat,
                -full_score_coordinates.index_select(0, selected).float(),
            )
            tail_mean_full_score = flat_sum_full.reshape_as(sum_full) / (
                remaining_counts[..., None].clamp_min(1.0)
            )
    reference_tensor = torch.as_tensor(
        reference, dtype=torch.float32, device=values.device
    )
    if tail_mean_full_score is not None and full_score_direction is not None:
        prototype_scores = torch.einsum(
            "bcd,d->bc", tail_mean_full_score, full_score_direction.float()
        )
    else:
        prototype_scores = torch.einsum(
            "bcr,r->bc", tail_mean_x, score_direction.float()
        )
    weights = remaining_counts * torch.exp(
        (prototype_scores - reference_tensor).clamp(-80.0, 40.0)
    )
    denominator = weights.sum()
    numerator = torch.einsum("bc,bcd->d", weights, tail_mean_v)
    uses_full_score = (
        tail_mean_full_score is not None and full_score_direction is not None
    )
    diagnostics = {
        "nonempty_prototypes": float((remaining_counts > 0).sum()),
        "maximum_log_weight": float(
            (prototype_scores - reference_tensor).max()
        ),
        "bits_per_token": float(
            model[
                "full_tail_bits_per_token"
                if uses_full_score
                else "proxy_tail_bits_per_token"
            ]
        ),
    }
    return denominator, numerator, diagnostics
