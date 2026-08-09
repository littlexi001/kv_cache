from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _pca_int4_partial_scores,
    _qk_metric_projection_factors,
    _symmetric_covariance_factors,
)


def _spd_matrix(dimension: int) -> torch.Tensor:
    matrix = torch.randn(dimension, dimension)
    return matrix @ matrix.T + 0.2 * torch.eye(dimension)


def test_qk_metric_full_rank_preserves_inner_product() -> None:
    torch.manual_seed(17)
    dimension = 8
    key_covariance = _spd_matrix(dimension).unsqueeze(0).unsqueeze(0)
    query_covariance = _spd_matrix(dimension).unsqueeze(0).unsqueeze(0)
    query_factor, key_factor = _qk_metric_projection_factors(
        key_covariance,
        query_covariance,
        dimension,
        query_shrinkage=0.0,
    )
    identity = query_factor @ key_factor.transpose(-1, -2)
    torch.testing.assert_close(
        identity,
        torch.eye(dimension).reshape(1, 1, dimension, dimension),
        atol=2.0e-4,
        rtol=2.0e-4,
    )


def test_qk_metric_spectral_floor_preserves_biorthogonal_identity() -> None:
    dimension = 8
    key_diagonal = torch.tensor(
        [4.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    )
    query_diagonal = torch.tensor(
        [0.0, 0.0, 3.0, 1.5, 0.75, 0.0, 0.0, 0.0]
    )
    key_covariance = torch.diag(key_diagonal).reshape(
        1, 1, dimension, dimension
    )
    query_covariance = torch.diag(query_diagonal).reshape(
        1, 1, dimension, dimension
    )
    query_factor, key_factor = _qk_metric_projection_factors(
        key_covariance,
        query_covariance,
        dimension,
        query_shrinkage=0.75,
    )
    identity = query_factor @ key_factor.transpose(-1, -2)
    torch.testing.assert_close(
        identity,
        torch.eye(dimension).reshape(1, 1, dimension, dimension),
        atol=4.0e-4,
        rtol=4.0e-4,
    )

    isotropic_scale = query_covariance.diagonal(
        dim1=-2, dim2=-1
    ).mean(dim=-1)
    shrunk_query = (
        0.25 * query_covariance
        + 0.75
        * isotropic_scale[..., None, None]
        * torch.eye(dimension)
    )
    key_sqrt, _ = _symmetric_covariance_factors(key_covariance)
    query_sqrt, _ = _symmetric_covariance_factors(shrunk_query)
    floored_key_covariance = key_sqrt @ key_sqrt
    floored_query_covariance = query_sqrt @ query_sqrt
    transformed_key = (
        key_factor.transpose(-1, -2)
        @ floored_key_covariance
        @ key_factor
    )
    transformed_query = (
        query_factor.transpose(-1, -2)
        @ floored_query_covariance
        @ query_factor
    )
    torch.testing.assert_close(
        transformed_query,
        transformed_key,
        atol=4.0e-4,
        rtol=4.0e-4,
    )


def test_qk_metric_rank_is_optimal_for_weighted_score_error() -> None:
    torch.manual_seed(23)
    dimension = 12
    rank = 5
    key_covariance = _spd_matrix(dimension).unsqueeze(0).unsqueeze(0)
    query_covariance = _spd_matrix(dimension).unsqueeze(0).unsqueeze(0)
    query_factor, key_factor = _qk_metric_projection_factors(
        key_covariance,
        query_covariance,
        rank,
        query_shrinkage=0.0,
    )
    qk_projection = query_factor @ key_factor.transpose(-1, -2)

    _, key_eigenvectors = torch.linalg.eigh(key_covariance)
    pca_basis = key_eigenvectors[..., -rank:]
    pca_projection = pca_basis @ pca_basis.transpose(-1, -2)

    key_sqrt, _ = _symmetric_covariance_factors(key_covariance)
    query_sqrt, _ = _symmetric_covariance_factors(query_covariance)
    identity = torch.eye(dimension).reshape(1, 1, dimension, dimension)

    def weighted_error(projection: torch.Tensor) -> torch.Tensor:
        residual = query_sqrt @ (identity - projection) @ key_sqrt
        return residual.square().sum()

    assert weighted_error(qk_projection) <= weighted_error(pca_projection) + 1.0e-4


def test_qk_metric_runtime_rebuilds_once_after_warmup() -> None:
    torch.manual_seed(31)
    key = torch.randn(1, 2, 48, 32)
    state: dict[str, object] = {}
    for step in range(4):
        query = torch.randn(1, 4, 1, 32)
        scores = _pca_int4_partial_scores(
            query,
            key,
            state,
            projection_dim=16,
            use_chunked_layout=True,
            logscale16=True,
            qk_metric_shrinkage=0.5,
            qk_metric_warmup_steps=2,
        )
        assert scores.shape == (1, 4, 48)
        assert torch.isfinite(scores).all()

    assert state["qk_metric_active"] is True
    assert state["qk_metric_query_count"] == 2
    assert state["qk_metric_rebuild_count"] == 1
    assert state["indexed_count"] == 48
    assert state["query_basis"].dtype == query.dtype
    assert state["basis"].dtype == key.dtype
    assert "key_second_moment" not in state
    assert "qk_metric_query_second_moment_sum" not in state
