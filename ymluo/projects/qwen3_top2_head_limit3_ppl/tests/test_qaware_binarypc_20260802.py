from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_qaware_binarypc_blockmean_layer0_20260802 import (  # noqa: E402
    assert_numeric_backend_sane,
    binary_proxy_scores,
    encode_binary_principal,
    encode_joint_kv_residual_codebook,
    encode_product_kv_residual_codebook,
    encode_residual_codebook,
    fit_binary_principal_projection,
    fit_joint_kv_residual_codebook,
    fit_product_kv_residual_codebook,
    fit_residual_codebook,
    oas_second_moment_shrinkage,
    quantize_blockwise_affine,
    quantize_log_error_norms,
    query_windows,
    query_metric_factors,
)
from analyze_qmetric_global_holdout_layer0_20260802 import (  # noqa: E402
    build_rabitq_index,
    fit_rvq_value_centroids,
    rabitq_proxy_scores,
    rvq_tail_output,
    value_jacobian_weights,
)


def test_binary_principal_shapes_and_finite_scores() -> None:
    generator = torch.Generator().manual_seed(17)
    coordinates = torch.randn(64, 8, generator=generator)
    projection = fit_binary_principal_projection(
        coordinates, bits=4, iterations=2, seed=19
    )
    codes, errors = encode_binary_principal(coordinates, projection)
    scores = binary_proxy_scores(
        codes, projection, torch.randn(8, generator=generator), 0.5
    )
    assert projection.shape == (4, 8)
    assert codes.shape == (64, 4)
    assert errors.shape == (64,)
    assert torch.isfinite(scores).all()


def test_more_bits_reduce_greedy_training_error() -> None:
    generator = torch.Generator().manual_seed(29)
    coordinates = torch.randn(128, 12, generator=generator)
    projection = fit_binary_principal_projection(
        coordinates, bits=8, iterations=3, seed=31
    )
    _, error4 = encode_binary_principal(coordinates, projection[:4])
    _, error8 = encode_binary_principal(coordinates, projection)
    assert error8.square().mean() <= error4.square().mean() + 1.0e-6


def test_query_metric_dual_factors_preserve_scores_and_metric_error() -> None:
    generator = torch.Generator().manual_seed(41)
    calibration = torch.randn(48, 12, generator=generator)
    keys = torch.randn(32, 12, generator=generator)
    reconstructed = keys + 0.1 * torch.randn(32, 12, generator=generator)
    queries = torch.randn(7, 12, generator=generator)
    query_factor, key_factor, resolved = query_metric_factors(calibration, 0.2)

    transformed_keys = keys @ key_factor
    transformed_queries = queries @ query_factor
    assert resolved == 0.2
    torch.testing.assert_close(key_factor, key_factor.T)
    torch.testing.assert_close(query_factor, query_factor.T)
    torch.testing.assert_close(
        transformed_queries @ transformed_keys.T,
        queries @ keys.T,
        rtol=2.0e-4,
        atol=2.0e-4,
    )

    second_moment = calibration.T @ calibration / calibration.shape[0]
    isotropic = torch.trace(second_moment) / second_moment.shape[0]
    metric = 0.8 * second_moment + 0.2 * isotropic * torch.eye(12)
    residual = keys - reconstructed
    expected_score_mse = torch.einsum("nd,df,nf->n", residual, metric, residual)
    transformed_residual_mse = ((keys - reconstructed) @ key_factor).square().sum(-1)
    torch.testing.assert_close(
        transformed_residual_mse,
        expected_score_mse,
        rtol=2.0e-4,
        atol=2.0e-4,
    )


def test_oas_second_moment_shrinkage_is_bounded() -> None:
    generator = torch.Generator().manual_seed(43)
    samples = torch.randn(24, 10, generator=generator)
    moment = samples.T @ samples / samples.shape[0]
    shrinkage = oas_second_moment_shrinkage(moment, samples.shape[0])
    assert 0.0 <= shrinkage <= 1.0
    _, _, resolved = query_metric_factors(samples, "oas")
    assert abs(resolved - shrinkage) < 1.0e-8


def test_spectral_initialization_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(47)
    coordinates = torch.randn(96, 10, generator=generator)
    first = fit_binary_principal_projection(
        coordinates, bits=8, iterations=3, seed=1, initialization="spectral"
    )
    second = fit_binary_principal_projection(
        coordinates, bits=8, iterations=3, seed=999, initialization="spectral"
    )
    torch.testing.assert_close(first, second)


def test_weighted_binary_projection_reduces_weighted_residual() -> None:
    generator = torch.Generator().manual_seed(91)
    samples = torch.randn(96, 16, generator=generator)
    weights = torch.linspace(0.25, 4.0, samples.shape[0])
    projection = fit_binary_principal_projection(
        samples,
        bits=8,
        iterations=4,
        seed=7,
        sample_weights=weights,
    )
    codes, _ = encode_binary_principal(samples, projection)
    reconstruction = codes @ projection
    weighted_error = (
        weights[:, None] * (samples - reconstruction).square()
    ).sum()
    weighted_zero_error = (weights[:, None] * samples.square()).sum()
    assert weighted_error < weighted_zero_error


def test_value_jacobian_weights_are_positive_and_normalized() -> None:
    generator = torch.Generator().manual_seed(93)
    keys = torch.randn(24, 8, generator=generator)
    values = torch.randn(24, 8, generator=generator)
    queries = torch.randn(7, 8, generator=generator)
    weights = value_jacobian_weights(keys, values, queries, 8**-0.5)
    assert weights.shape == (24,)
    assert bool((weights >= 1.0).all())
    torch.testing.assert_close(weights.mean(), torch.tensor(2.0))


def test_rabitq_reference_scores_are_finite_and_correlated() -> None:
    generator = torch.Generator().manual_seed(101)
    keys = torch.randn(256, 16, generator=generator)
    queries = torch.randn(32, 16, generator=generator)
    index = build_rabitq_index(keys, seed=103)
    query_centroid = queries.mean(dim=0)
    query = queries[0]
    proxy = rabitq_proxy_scores(index, keys, query, query_centroid, 0.25)
    exact = keys @ query * 0.25
    assert torch.isfinite(proxy).all()
    correlation = torch.corrcoef(torch.stack((proxy, exact)))[0, 1]
    assert float(correlation) > 0.5


def test_rvq_value_tail_is_exact_for_cluster_constant_values() -> None:
    assignments = torch.arange(32) % 4
    centroids = torch.randn(4, 6, generator=torch.Generator().manual_seed(107))
    values = centroids.index_select(0, assignments)
    fitted, _ = fit_rvq_value_centroids(values, assignments, clusters=4, bits=16)
    scores = torch.randn(32, generator=torch.Generator().manual_seed(109))
    selected = torch.tensor([0, 3, 9, 17])
    output = rvq_tail_output(
        scores,
        scores,
        values,
        selected,
        assignments,
        fitted,
    )
    reference = torch.softmax(scores, dim=0) @ values
    torch.testing.assert_close(output, reference, rtol=2.0e-5, atol=2.0e-5)


def test_selected_conditioned_tail_removes_selected_values_from_cluster_mean() -> None:
    assignments = torch.tensor([0, 0, 0, 1, 1, 1])
    values = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [0.0, 1.0], [0.0, 3.0], [0.0, 5.0]]
    )
    centroids, _ = fit_rvq_value_centroids(values, assignments, clusters=2, bits=16)
    scores = torch.zeros(6)
    selected = torch.tensor([2, 5])
    output = rvq_tail_output(
        scores,
        scores,
        values,
        selected,
        assignments,
        centroids,
        selected_conditioned=True,
    )
    torch.testing.assert_close(output, values.mean(dim=0), rtol=3.0e-5, atol=3.0e-5)


def test_projection_fit_stays_finite_on_high_dynamic_range_data() -> None:
    generator = torch.Generator().manual_seed(59)
    coordinates = torch.randn(256, 16, generator=generator)
    coordinates[:, 0] *= 1.0e4
    projection = fit_binary_principal_projection(
        coordinates, bits=64, iterations=4, seed=61, initialization="random"
    )
    _, errors = encode_binary_principal(coordinates, projection)
    assert torch.isfinite(projection).all()
    assert torch.isfinite(errors).all()
    assert errors.square().mean() <= coordinates.square().sum(-1).mean() + 1.0e-4


def test_numeric_backend_self_test_passes() -> None:
    assert_numeric_backend_sane()


def test_log_error_quantization_is_positive_and_bounded_cost() -> None:
    errors = torch.logspace(-4, 2, 513)
    reconstructed, bits_per_token = quantize_log_error_norms(errors, 4, 256)
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed > 0).all()
    assert 4.0 < bits_per_token < 4.2
    relative = (reconstructed / errors).maximum(errors / reconstructed)
    assert float(relative.max()) < 2.0


def test_blockwise_affine_quantization_tracks_signed_scalars() -> None:
    values = torch.linspace(-1.25, 2.75, 513)
    reconstructed, bits_per_token = quantize_blockwise_affine(values, 4, 256)
    assert torch.isfinite(reconstructed).all()
    assert 4.0 < bits_per_token < 4.2
    assert float((reconstructed - values).abs().max()) < 0.14


def test_residual_codebook_reduces_training_error() -> None:
    generator = torch.Generator().manual_seed(97)
    residuals = torch.randn(128, 12, generator=generator)
    codebook = fit_residual_codebook(residuals, clusters=8, iterations=5)
    assignments, errors = encode_residual_codebook(residuals, codebook)
    assert codebook.shape == (8, 12)
    assert assignments.shape == (128,)
    assert errors.square().mean() < residuals.square().sum(dim=-1).mean()


def test_joint_kv_residual_code_uses_one_id_for_both_spaces() -> None:
    generator = torch.Generator().manual_seed(113)
    key_centers = torch.tensor(((-2.0, 0.5), (2.0, -0.5)))
    value_centers = torch.tensor(((0.25, -3.0), (-0.25, 3.0)))
    labels = torch.arange(64) % 2
    keys = key_centers.index_select(0, labels) + 0.01 * torch.randn(
        64, 2, generator=generator
    )
    values = value_centers.index_select(0, labels) + 0.01 * torch.randn(
        64, 2, generator=generator
    )
    model = fit_joint_kv_residual_codebook(
        keys, values, clusters=2, iterations=4, value_weight=1.0
    )
    assignments, errors, key_codebook = encode_joint_kv_residual_codebook(
        keys, values, model
    )
    assert assignments.shape == (64,)
    assert key_codebook.shape == (2, 2)
    assert float(errors.mean()) < 0.05
    assert len(torch.unique(assignments[labels == 0])) == 1
    assert len(torch.unique(assignments[labels == 1])) == 1


def test_product_kv_code_packs_independent_subcodes() -> None:
    generator = torch.Generator().manual_seed(127)
    keys = torch.randn(96, 4, generator=generator)
    values = torch.randn(96, 4, generator=generator)
    model = fit_product_kv_residual_codebook(
        keys, values, total_bits=4, key_bits=2, iterations=3
    )
    assignments, errors, key_codebook = encode_product_kv_residual_codebook(
        keys, values, model
    )
    assert assignments.shape == (96,)
    assert int(assignments.min()) >= 0
    assert int(assignments.max()) < 16
    assert key_codebook.shape == (4, 4)
    assert errors.square().mean() < keys.square().sum(dim=-1).mean()


def test_history_tail_query_windows_do_not_use_heldout_queries() -> None:
    query = torch.arange(40).reshape(20, 2)
    calibration, heldout = query_windows(
        query,
        history_tokens=12,
        calibration_tokens=3,
        query_tokens=2,
        calibration_source="history_tail",
        heldout_gap=4,
    )
    torch.testing.assert_close(calibration, query[9:12])
    torch.testing.assert_close(heldout, query[16:18])
