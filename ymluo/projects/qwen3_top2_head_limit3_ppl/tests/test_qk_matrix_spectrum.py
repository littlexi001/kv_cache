from __future__ import annotations

import pytest
import torch

from analyze_qk_matrix_spectrum_20260726 import (
    projection_score_metrics,
    spectral_metrics,
)


def test_projector_fidelity_uses_residual_not_approximate_norm() -> None:
    key = torch.diag(torch.tensor([1.0, 2.0]))
    queries = torch.tensor([[1.0, 0.0]])
    basis = torch.tensor([[1.0], [1.0]]) / (2.0**0.5)
    head_dim = 2

    metrics = projection_score_metrics(
        key.transpose(0, 1) @ key,
        queries.transpose(0, 1) @ queries,
        basis,
        head_dim,
    )
    exact = queries @ key.transpose(0, 1) / (head_dim**0.5)
    approximate = (
        (queries @ basis)
        @ (key @ basis).transpose(0, 1)
        / (head_dim**0.5)
    )
    relative_residual = (
        (exact - approximate).square().sum() / exact.square().sum()
    )
    approximate_norm_fraction = (
        approximate.square().sum() / exact.square().sum()
    )

    assert metrics["qk_fidelity"] == pytest.approx(
        float((1.0 - relative_residual).item())
    )
    assert metrics["qk_fidelity"] == pytest.approx(-0.25)
    assert float(approximate_norm_fraction.item()) == pytest.approx(1.25)


def test_whitened_oblique_map_recovers_truncated_qk_svd() -> None:
    generator = torch.Generator().manual_seed(29)
    key = torch.randn(19, 6, generator=generator, dtype=torch.float64)
    key = key - key.mean(dim=0, keepdim=True)
    queries = torch.randn(
        13,
        6,
        generator=generator,
        dtype=torch.float64,
    )
    rank = 3
    head_dim = key.shape[-1]

    key_gram = key.transpose(0, 1) @ key
    query_gram = queries.transpose(0, 1) @ queries
    eigenvalues, eigenvectors = torch.linalg.eigh(key_gram)
    key_sqrt = (
        eigenvectors
        @ torch.diag(eigenvalues.sqrt())
        @ eigenvectors.transpose(0, 1)
    )
    key_inverse_sqrt = (
        eigenvectors
        @ torch.diag(eigenvalues.rsqrt())
        @ eigenvectors.transpose(0, 1)
    )
    score_gram = key_sqrt @ query_gram @ key_sqrt / head_dim
    _, score_basis = torch.linalg.eigh(score_gram)
    score_basis = score_basis[:, -rank:]
    oblique_map = (
        key_sqrt
        @ score_basis
        @ score_basis.transpose(0, 1)
        @ key_inverse_sqrt
    )

    exact = queries @ key.transpose(0, 1) / (head_dim**0.5)
    left, singular_values, right_h = torch.linalg.svd(
        exact,
        full_matrices=False,
    )
    truncated = (
        left[:, :rank]
        @ torch.diag(singular_values[:rank])
        @ right_h[:rank]
    )
    reconstructed = (
        queries @ oblique_map @ key.transpose(0, 1) / (head_dim**0.5)
    )

    assert torch.allclose(
        oblique_map @ oblique_map,
        oblique_map,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert torch.allclose(
        reconstructed,
        truncated,
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    assert not torch.allclose(
        oblique_map,
        oblique_map.transpose(0, 1),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_spectral_metrics_match_materialized_qk_energy() -> None:
    generator = torch.Generator().manual_seed(17)
    key = torch.randn(31, 8, generator=generator)
    queries = torch.randn(12, 8, generator=generator)
    rank = 4
    metrics = spectral_metrics(
        key,
        queries,
        rank,
        basis_prefix_tokens=key.shape[0],
        basis_sample_stride=1,
    )

    _, _, right_h = torch.linalg.svd(key, full_matrices=False)
    basis = right_h[:rank].transpose(0, 1)
    exact = queries @ key.transpose(0, 1) / (key.shape[-1] ** 0.5)
    approximate = (
        (queries @ basis)
        @ (key @ basis).transpose(0, 1)
        / (key.shape[-1] ** 0.5)
    )
    centered_exact = exact - exact.mean(dim=1, keepdim=True)
    retained = 1.0 - (
        (exact - approximate).square().sum()
        / exact.square().sum()
    )
    row_mean_fraction = (
        exact.mean(dim=1, keepdim=True).square().sum() * exact.shape[1]
        / exact.square().sum()
    )
    _, _, right_h = torch.linalg.svd(exact, full_matrices=False)
    constant = torch.ones(exact.shape[1]) / (exact.shape[1] ** 0.5)
    top_constant_alignment = right_h[0].dot(constant).square()
    centered_singular_energy = torch.linalg.svdvals(centered_exact).square()
    centered_rank1_fraction = (
        centered_singular_energy[0] / centered_singular_energy.sum()
    )
    centered_key = key - key.mean(dim=0, keepdim=True)
    centered_key_covariance = centered_key.transpose(0, 1) @ centered_key
    centered_key_covariance /= key.shape[0]
    query_covariance = queries.transpose(0, 1) @ queries
    query_covariance /= queries.shape[0]
    centered_commutator = (
        centered_key_covariance @ query_covariance
        - query_covariance @ centered_key_covariance
    )
    centered_commutator_ratio = (
        centered_commutator.norm()
        / (centered_key_covariance.norm() * query_covariance.norm())
    )

    assert metrics["qk_energy_retained_key_pca48"] == pytest.approx(
        float(retained.item()),
        rel=1.0e-5,
        abs=1.0e-5,
    )
    assert metrics["qk_tail_bound_satisfied"] == 1.0
    assert (
        metrics["qk_energy_retained_optimal_rank48"]
        + 1.0e-6
        >= metrics["qk_energy_retained_key_pca48"]
    )
    assert metrics["production_prefix_pca_qk_fidelity"] == pytest.approx(
        metrics["full_key_pca_qk_fidelity"],
        rel=1.0e-5,
        abs=1.0e-5,
    )
    assert metrics["production_prefix_pca_subspace_overlap"] == pytest.approx(
        1.0,
        rel=1.0e-5,
        abs=1.0e-5,
    )
    assert metrics[
        "softmax_invariant_row_mean_energy_fraction"
    ] == pytest.approx(
        float(row_mean_fraction.item()),
        rel=1.0e-5,
        abs=1.0e-6,
    )
    assert metrics[
        "qk_top_right_vector_constant_alignment"
    ] == pytest.approx(float(top_constant_alignment.item()), rel=1.0e-5)
    assert metrics["centered_qk_rank1_energy_fraction"] == pytest.approx(
        float(centered_rank1_fraction.item()),
        rel=1.0e-5,
    )
    assert metrics[
        "centered_key_query_covariance_commutator_ratio"
    ] == pytest.approx(float(centered_commutator_ratio.item()), rel=1.0e-5)


def test_tail_aligned_queries_expose_key_pca_counterexample() -> None:
    diagonal = torch.tensor([9.0, 7.0, 5.0, 3.0])
    key = torch.diag(diagonal)
    queries = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    metrics = spectral_metrics(key, queries, rank=2)

    assert metrics["key_energy_retained_rank48"] > 0.75
    assert metrics["qk_energy_retained_key_pca48"] == pytest.approx(0.0)


def test_prefix_basis_metric_detects_late_subspace_shift() -> None:
    key = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
        ]
    )
    queries = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    metrics = spectral_metrics(
        key,
        queries,
        rank=2,
        basis_prefix_tokens=4,
        basis_sample_stride=1,
    )

    assert metrics["full_key_pca_qk_fidelity"] == pytest.approx(1.0)
    assert metrics["production_prefix_pca_qk_fidelity"] == pytest.approx(0.0)
    assert metrics["production_prefix_pca_subspace_overlap"] == pytest.approx(
        0.0
    )


def test_centered_qk_spectrum_ignores_softmax_invariant_key_offset() -> None:
    generator = torch.Generator().manual_seed(23)
    key = torch.randn(43, 8, generator=generator)
    queries = torch.randn(17, 8, generator=generator)
    offset = torch.tensor(
        [[100.0, -80.0, 60.0, 40.0, -20.0, 10.0, 5.0, -2.0]]
    )

    base = spectral_metrics(
        key,
        queries,
        rank=4,
        basis_prefix_tokens=key.shape[0],
        basis_sample_stride=1,
    )
    shifted = spectral_metrics(
        key + offset,
        queries,
        rank=4,
        basis_prefix_tokens=key.shape[0],
        basis_sample_stride=1,
    )

    assert shifted["qk_effective_rank"] < base["qk_effective_rank"]
    assert shifted["centered_qk_effective_rank"] == pytest.approx(
        base["centered_qk_effective_rank"],
        rel=1.0e-5,
        abs=1.0e-5,
    )
    assert shifted[
        "centered_qk_energy_retained_optimal_rank48"
    ] == pytest.approx(
        base["centered_qk_energy_retained_optimal_rank48"],
        rel=1.0e-5,
        abs=1.0e-5,
    )
    assert shifted["softmax_invariant_row_mean_energy_fraction"] > 0.99
    assert shifted["qk_top_right_vector_constant_alignment"] > 0.99
