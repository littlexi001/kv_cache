from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_head_top2_targeted_ppl_20260714 as head_top2  # noqa: E402
from analyze_qksieve_complementary_residual_sketch_20260803 import (  # noqa: E402
    ragged_union_mask,
    selection_metrics_from_mask,
)
from analyze_qksieve_output_risk_budget_20260803 import (  # noqa: E402
    affine_residual_bound_ladder_mask,
    approximate_output,
    balanced_head_output_scales,
    boundary_crossing_probability,
    conformal_score_uncertainty,
    crossfit_affine_score_rmse,
    crossfit_affine_softmax_kl,
    exact_prefix_tail_ratio_mass_ladder_mask,
    gaussian_mass_prefix_mask,
    global_floor_rss_mask,
    histogram_coverage_mask,
    key_allocation_distortion,
    key_quantization_candidates,
    oas_query_metric_parameters,
    qk_calibration_queries,
    interval_certified_mass_ladder_mask,
    joint_qk_value_rss_risk,
    minimum_refinement_mask,
    plain_quantize_band,
    progressive_error_balanced_masks,
    relative_tail_rss_mask,
    relative_tail_risk_mask,
    sampled_mass_prefix_mask,
    sampled_rank_mass_ladder_mask,
    sampled_score_output_error,
    sampled_tail_score_output_error,
    sampled_tail_partition_scale,
    scalar_residual_rss_mask,
)


def test_qk_calibration_queries_keeps_prefill_and_decode_semantics() -> None:
    prefill = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    decode = 100.0 + torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4)

    prefill_only = qk_calibration_queries(decode, prefill, "prefill")
    decode_only = qk_calibration_queries(decode, prefill, "decode")
    combined = qk_calibration_queries(decode, prefill, "prefill_decode")

    assert torch.equal(prefill_only, prefill.reshape(-1, 4))
    assert torch.equal(decode_only, decode.reshape(-1, 4))
    assert torch.equal(combined, torch.cat((prefill_only, decode_only), dim=0))


def test_key_allocator_scores_the_executed_quantizer() -> None:
    torch.manual_seed(7)
    keys = torch.randn(41, 128)
    queries = torch.randn(9, 128)
    for quantizer in ("plain", "metric"):
        candidates = key_quantization_candidates(keys, queries, quantizer)
        key_costs = key_allocation_distortion(
            keys, queries, candidates, "key_mse"
        )
        qk_costs = key_allocation_distortion(
            keys, queries, candidates, "qk_mse"
        )
        oas_costs = key_allocation_distortion(
            keys, queries, candidates, "oas_qk_mse"
        )
        weights = torch.linspace(0.5, 2.0, 128)
        balanced_costs = key_allocation_distortion(
            keys,
            queries,
            candidates,
            "balanced_qk_mse",
            weights,
        )
        robust_costs = key_allocation_distortion(
            keys,
            queries,
            candidates,
            "robust_qk_mse",
            weights,
        )
        for band_index in range(8):
            exact = keys[:, band_index * 16 : (band_index + 1) * 16]
            query = queries[:, band_index * 16 : (band_index + 1) * 16]
            for bits, reconstructed in candidates[band_index].items():
                residual = exact - reconstructed
                assert torch.allclose(
                    key_costs[band_index][bits], residual.square().mean()
                )
                assert torch.allclose(
                    qk_costs[band_index][bits],
                    (query @ residual.T).square().mean(),
                )
                alpha, isotropic_variance = oas_query_metric_parameters(
                    queries
                )
                expected_oas = (
                    (1.0 - alpha) * qk_costs[band_index][bits]
                    + alpha
                    * residual.square().sum(dim=-1).mean()
                    * isotropic_variance
                )
                assert torch.allclose(
                    oas_costs[band_index][bits], expected_oas
                )
                band_weights = weights[
                    band_index * 16 : (band_index + 1) * 16
                ]
                assert torch.allclose(
                    balanced_costs[band_index][bits],
                    (residual.square() * band_weights[None, :])
                    .sum(dim=-1)
                    .mean(),
                )
                assert torch.allclose(
                    robust_costs[band_index][bits],
                    torch.maximum(
                        qk_costs[band_index][bits],
                        balanced_costs[band_index][bits],
                    ),
                )


def test_plain_quantize_band_supports_smooth_intermediate_widths() -> None:
    torch.manual_seed(11)
    values = torch.randn(37, 16)
    error_by_bits = {
        bits: (values - plain_quantize_band(values, bits)).square().mean()
        for bits in (2, 3, 4, 6, 8)
    }

    assert all(torch.isfinite(error) for error in error_by_bits.values())
    assert all(
        error_by_bits[left] >= error_by_bits[right]
        for left, right in zip((2, 3, 4, 6), (3, 4, 6, 8))
    )


def test_sampled_score_output_error_is_exact_when_every_token_is_probed() -> None:
    torch.manual_seed(13)
    exact_scores = torch.randn(2, 11)
    proxy_scores = exact_scores + 0.1 * torch.randn(2, 11)
    value = torch.randn(11, 4)
    grams = torch.eye(4).repeat(2, 1, 1)

    diagnostic = sampled_score_output_error(
        exact_scores,
        proxy_scores,
        value,
        grams,
        query_groups=2,
        sample_count=11,
        top_count=3,
    )

    torch.testing.assert_close(
        diagnostic["estimate"], diagnostic["first_order"]
    )
    torch.testing.assert_close(
        diagnostic["standard_error"], torch.zeros(2)
    )


def test_sampled_tail_score_error_is_exact_when_tail_is_fully_probed() -> None:
    torch.manual_seed(19)
    exact_scores = torch.randn(2, 13)
    proxy_scores = exact_scores + 0.2 * torch.randn(2, 13)
    value = torch.randn(13, 4)
    reconstructed = value + 0.1 * torch.randn_like(value)
    selected = torch.zeros(2, 13, dtype=torch.bool)
    selected[:, :5] = True
    grams = torch.eye(4).repeat(2, 1, 1)

    diagnostic = sampled_tail_score_output_error(
        exact_scores,
        proxy_scores,
        value,
        reconstructed,
        selected,
        grams,
        query_groups=2,
        sample_count=13,
        top_count=3,
        score_uncertainty=torch.ones_like(exact_scores),
    )

    torch.testing.assert_close(
        diagnostic["estimate"], diagnostic["first_order"]
    )
    torch.testing.assert_close(
        diagnostic["standard_error"], torch.zeros(2)
    )


def test_conformal_score_uncertainty_covers_full_calibration_set() -> None:
    proxy = torch.zeros(2, 17)
    exact = torch.linspace(-2.0, 2.0, 34).reshape(2, 17)
    raw = exact.abs().clamp_min(0.1) / 2.0

    uncertainty, scale = conformal_score_uncertainty(
        exact, proxy, raw, sample_count=17, miscoverage=0.01
    )

    assert torch.all(scale >= 2.0 - 1.0e-6)
    assert torch.all((exact - proxy).abs() <= uncertainty + 1.0e-6)


def test_sampled_tail_partition_scale_is_exact_when_tail_is_fully_probed() -> None:
    exact = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    proxy = torch.tensor([[1.5, 0.5, 0.2, -0.5, -1.5]])
    selected = torch.tensor([[True, False, False, False, False]])
    scale = sampled_tail_partition_scale(
        exact, proxy, selected, sample_count=5, top_count=2
    )
    maximum = torch.maximum(exact.amax(dim=-1), proxy.amax(dim=-1))
    exact_weight = torch.exp(exact - maximum[:, None])
    proxy_weight = torch.exp(proxy - maximum[:, None])
    expected = (
        (exact_weight * (~selected)).sum(dim=-1)
        / (proxy_weight * (~selected)).sum(dim=-1)
    )

    torch.testing.assert_close(scale, expected)


def test_oas_query_metric_shrinks_underdetermined_queries() -> None:
    torch.manual_seed(17)
    queries = torch.randn(8, 128)
    alpha, isotropic_variance = oas_query_metric_parameters(queries)

    assert 0.0 <= float(alpha) <= 1.0
    assert float(alpha) > 0.5
    torch.testing.assert_close(isotropic_variance, queries.square().mean())


def test_balanced_head_output_scales_preserve_layer_rss_energy() -> None:
    scales = torch.tensor([[0.0, 1.0, 2.0, 7.0], [3.0, 3.0, 3.0, 3.0]])
    balanced = balanced_head_output_scales(scales)

    torch.testing.assert_close(
        balanced.square().sum(dim=-1), scales.square().sum(dim=-1)
    )
    assert balanced[0, 0] > 0.0
    torch.testing.assert_close(balanced[1], scales[1])


def test_joint_qk_value_rss_risk_combines_terms_in_quadrature() -> None:
    residual = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    score_rmse = torch.tensor([2.0, 3.0])
    deviation = torch.tensor([[2.0, 0.0], [4.0, 0.0]])

    risk = joint_qk_value_rss_risk(residual, score_rmse, deviation)

    torch.testing.assert_close(
        risk, torch.tensor([[5.0, 4.0], [12.0, 5.0]])
    )


def test_boundary_crossing_probability_is_largest_at_cutoff() -> None:
    priority = torch.tensor([[3.0, 2.0, 1.0, -2.0]])
    selected = torch.tensor([[True, True, False, False]])
    probability = boundary_crossing_probability(
        priority, selected, torch.tensor([0.5])
    )

    torch.testing.assert_close(probability[0, 1], torch.tensor(0.5))
    assert probability[0, 0] < probability[0, 1]
    assert probability[0, 2] < probability[0, 1]
    assert probability[0, 3] < probability[0, 2]


def test_minimum_refinement_mask_meets_squared_budget() -> None:
    contribution = torch.tensor([[9.0, 4.0, 1.0, 0.25]])
    mask = minimum_refinement_mask(contribution, torch.tensor([1.5]))

    assert torch.equal(mask, torch.tensor([[True, True, False, False]]))
    assert float(contribution.masked_fill(mask, 0.0).sum()) <= 1.5


def test_progressive_refinement_targets_uncertain_high_impact_tokens() -> None:
    base_scores = torch.tensor([[3.0, 2.9, 2.8, -4.0]])
    refined_scores = torch.tensor([[3.0, 2.5, 3.2, -4.0]])
    base_risk = torch.tensor([[1.0, 2.0, 2.0, 0.01]])
    refined_risk = base_risk.clone()

    selected, refined, unresolved = progressive_error_balanced_masks(
        base_scores,
        refined_scores,
        base_risk,
        refined_risk,
        torch.tensor([0.5]),
        torch.tensor([0.05]),
        torch.tensor([1.0]),
        tolerance=0.2,
        safety_factor=1.0,
        minimum_top_k=1,
        maximum_top_k=0,
        rounds=2,
    )

    assert refined[0, 1] or refined[0, 2]
    assert not refined[0, 3]
    assert selected.any()
    assert torch.isfinite(unresolved).all()


def test_block_residual_correction_is_exact_for_block_constant_weights() -> None:
    torch.manual_seed(29)
    tokens = 128
    dimensions = 4
    exact_scores = torch.zeros(1, tokens)
    proxy_scores = exact_scores.clone()
    value = torch.randn(tokens, dimensions)
    reconstructed = torch.zeros_like(value)
    mask = torch.zeros(1, tokens, dtype=torch.bool)
    mask[:, :8] = True

    outputs, full, _, _ = approximate_output(
        exact_scores,
        proxy_scores,
        value,
        reconstructed,
        mask,
    )

    torch.testing.assert_close(
        outputs["hybrid_sketch_blockresidual64"], full
    )
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _block_affine_pack_int4,
    _affine_value_residual_tail_correction,
    _choose_fulltopk_ragged_split_count,
    _choose_sparse_attention_split_count,
    _configure_packed_qmse_state,
    _crossing_bernstein_exact_rerank,
    _configured_unit_rope_pair_phase_tables,
    _exact_sparse_selection_metrics,
    _fit_affine_score_calibration,
    _global_floor_relative_rss_candidates,
    _head_local_dual_mass_candidates,
    _head_local_mass_floor_candidates,
    _head_local_relative_rss_candidates,
    _layer_equal_prefix_rss_candidates,
    _inverse_standard_rope,
    _post_overfetch_prerope_rerank_candidates,
    _proxy_value_output_scale,
    _projected_key_quantization_residual_norm,
    _small_matrix_cholesky,
    _small_matrix_eigh,
    _small_matrix_solve,
    _small_matrix_svd,
    _systematic_tail_sample_indices,
    _ensure_value_sketch_state,
    _value_residual_priority_log_risk,
    exact_head_adaptive_mass_attention,
    exact_head_top_fraction_attention,
    qabs_sampled_head_adaptive_attention,
)


def test_vectorized_block_int4_matches_per_block_reference() -> None:
    torch.manual_seed(31)
    coefficients = torch.randn(2, 3, 19, 6)
    packed, minimum, scale = _block_affine_pack_int4(coefficients, 8)
    unpacked = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
        2, 3, 19, 6
    )
    block_ids = torch.arange(19) // 8
    reconstructed = (
        unpacked.float() * scale.index_select(2, block_ids)
        + minimum.index_select(2, block_ids)
    )

    reference = torch.empty_like(reconstructed)
    for start in range(0, 19, 8):
        stop = min(19, start + 8)
        block = coefficients[..., start:stop, :]
        block_minimum = block.amin(dim=-2, keepdim=True)
        block_scale = (
            (block.amax(dim=-2, keepdim=True) - block_minimum) / 15.0
        ).clamp_min(1.0e-12)
        block_codes = torch.round(
            (block - block_minimum) / block_scale
        ).clamp(0, 15)
        reference[..., start:stop, :] = (
            block_codes * block_scale + block_minimum
        )

    torch.testing.assert_close(reconstructed, reference)


def test_value_sketch_initial_pack_streams_block_aligned_chunks(
    monkeypatch,
) -> None:
    torch.manual_seed(33)
    value = torch.randn(1, 2, 19, 8)
    state = {}
    monkeypatch.setenv("QKSIEVE_VALUE_SKETCH_INIT_CHUNK_TOKENS", "8")

    prefix, block_count = _ensure_value_sketch_state(
        value,
        state,
        rank=4,
        bits=4,
        block_size=4,
        sample_count=19,
    )

    mean = state[f"{prefix}_mean"].float()
    encoder = state[f"{prefix}_encoder"].float()
    coefficients = torch.einsum(
        "bhnd,bhdr->bhnr",
        value.float() - mean.unsqueeze(2),
        encoder,
    )
    expected_codes, expected_minimum, expected_scale = _block_affine_pack_int4(
        coefficients,
        4,
    )
    assert block_count == 5
    torch.testing.assert_close(
        state[f"{prefix}_packed_codes"][..., :19, :], expected_codes
    )
    torch.testing.assert_close(
        state[f"{prefix}_minimum"][..., :5, :].float(), expected_minimum
    )
    torch.testing.assert_close(
        state[f"{prefix}_scale"][..., :5, :].float(), expected_scale
    )


def test_small_matrix_solver_helpers_match_torch() -> None:
    torch.manual_seed(37)
    source = torch.randn(2, 5, 5)
    positive = source @ source.transpose(-1, -2) + 0.1 * torch.eye(5)

    values, vectors = _small_matrix_eigh(positive)
    reference_values, reference_vectors = torch.linalg.eigh(positive)
    torch.testing.assert_close(values, reference_values)
    torch.testing.assert_close(vectors.abs(), reference_vectors.abs())

    cholesky = _small_matrix_cholesky(positive)
    torch.testing.assert_close(cholesky, torch.linalg.cholesky(positive))

    rhs = torch.randn(2, 5, 3)
    torch.testing.assert_close(
        _small_matrix_solve(positive, rhs),
        torch.linalg.solve(positive, rhs),
    )

    left, singular, right_h = _small_matrix_svd(source)
    reconstructed = left @ torch.diag_embed(singular) @ right_h
    torch.testing.assert_close(reconstructed, source)


def test_small_matrix_eigh_stabilizes_rank_deficient_covariance() -> None:
    torch.manual_seed(38)
    factors = torch.randn(2, 32, 5)
    covariance = factors @ factors.transpose(-1, -2)

    values, vectors = _small_matrix_eigh(covariance)

    assert bool(torch.isfinite(values).all())
    assert bool(torch.isfinite(vectors).all())
    reconstructed = vectors @ torch.diag_embed(values) @ vectors.transpose(-1, -2)
    torch.testing.assert_close(reconstructed, covariance, rtol=1.0e-4, atol=1.0e-4)


def test_crossing_bernstein_mode_contract() -> None:
    mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "valuesketch16i4shared_wometric_keyrisk4_"
        "crossbernstein99_cal256_packed_fulltopk_oas"
    )
    state = {}
    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_bit_budget"] == 15
    assert state["packed_qmse_key_residual_risk_bits"] == 4
    assert math.isclose(
        state["packed_qmse_crossing_failure_probability"], 0.01
    )
    assert state["packed_qmse_crossing_calibration_samples"] == 256


def test_empirical_crossing_mode_contract() -> None:
    mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "valuesketch16i4shared_wometric_keyrisk4_"
        "crossempirical99_cal256_packed_fulltopk_oas"
    )
    state = {}
    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_crossing_probability_model"] == (
        "empirical_add_one"
    )
    assert math.isclose(
        state["packed_qmse_crossing_failure_probability"], 0.01
    )
    assert state["packed_qmse_crossing_calibration_samples"] == 256
    assert not state["packed_qmse_crossing_keep_union"]


def test_empirical_crossing_keep_union_mode_contract() -> None:
    mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "valuesketch16i4shared_wometric_keyrisk4_"
        "crossempirical99_cal256_keepunion_packed_fulltopk_oas"
    )
    state = {}
    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_crossing_probability_model"] == (
        "empirical_add_one"
    )
    assert state["packed_qmse_crossing_keep_union"]


def test_conditional_residual_crossing_mode_contract() -> None:
    mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "valuesketch16i4shared_wometric_condres8global_keyrisk4_"
        "crossempirical99_cal256_packed_fulltopk_oas"
    )
    state = {}
    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_value_sketch_rank"] == 16
    assert state["packed_qmse_conditional_value_residual_dim"] == 8
    assert state["packed_qmse_key_residual_risk_bits"] == 4
    assert state["packed_qmse_crossing_probability_model"] == (
        "empirical_add_one"
    )
    assert not state["packed_qmse_crossing_keep_union"]


def test_projected_key_residual_norm_has_exact_endpoints() -> None:
    torch.manual_seed(41)
    projected = torch.randn(1, 2, 7, 128)
    all_zero = torch.zeros(1, 2, 8, dtype=torch.int8)
    all_exact = torch.full((1, 2, 8), 16, dtype=torch.int8)

    torch.testing.assert_close(
        _projected_key_quantization_residual_norm(projected, all_zero),
        torch.linalg.vector_norm(projected.float(), dim=-1),
    )
    torch.testing.assert_close(
        _projected_key_quantization_residual_norm(projected, all_exact),
        torch.zeros(1, 2, 7),
    )


def test_projected_key_residual_norm_matches_scalar_mixed_bits() -> None:
    torch.manual_seed(43)
    projected = torch.randn(1, 2, 5, 128)
    allocation = torch.tensor(
        [[[0, 1, 2, 4, 8, 16, 2, 4], [16, 8, 4, 2, 1, 0, 4, 2]]],
        dtype=torch.int8,
    )
    expected_square = torch.zeros(1, 2, 5)
    for head in range(2):
        for band_index in range(8):
            bits = int(allocation[0, head, band_index])
            band = projected[
                0, head, :, 16 * band_index : 16 * (band_index + 1)
            ].float()
            if bits == 0:
                reconstructed = torch.zeros_like(band)
            elif bits == 16:
                reconstructed = band
            elif bits == 1:
                codes = torch.where(band >= 0.0, 1.0, -1.0)
                scale = band.abs().mean(dim=-1, keepdim=True)
                reconstructed = codes * scale
            else:
                maximum_code = (1 << (bits - 1)) - 1
                scale = band.abs().amax(dim=-1, keepdim=True) / maximum_code
                codes = torch.round(band / scale).clamp(
                    -maximum_code, maximum_code
                )
                reconstructed = codes * scale
            expected_square[0, head].add_(
                (band - reconstructed).square().sum(dim=-1)
            )

    torch.testing.assert_close(
        _projected_key_quantization_residual_norm(projected, allocation),
        expected_square.sqrt(),
    )


def test_crossing_bernstein_rescues_high_residual_hidden_needle() -> None:
    tokens = 32
    proxy = torch.linspace(4.0, -2.0, tokens).view(1, 1, tokens)
    exact = proxy.clone()
    needle = 27
    exact[..., needle] = 6.0
    query = torch.zeros(1, 1, 4)
    query[..., 0] = 1.0
    key = torch.zeros(1, 1, tokens, 4)
    key[..., 0] = exact / 1.0
    projected_query = query.view(1, 1, 1, 4)
    residual_norm = torch.full((1, 1, tokens), 0.1)
    residual_norm[..., needle] = 8.0
    base_indices = torch.topk(proxy, 4, dim=-1, sorted=False).indices
    base_counts = torch.full((1, 1), 4, dtype=torch.long)

    reranked, rescue_counts, _, _ = _crossing_bernstein_exact_rerank(
        query,
        key,
        proxy,
        projected_query,
        residual_norm,
        base_indices,
        base_counts,
        1.0,
        16,
        0.01,
        16,
    )

    assert int(rescue_counts.item()) > 0
    assert bool(torch.any(reranked == needle))

    kept_union, kept_counts, _, _ = _crossing_bernstein_exact_rerank(
        query,
        key,
        proxy,
        projected_query,
        residual_norm,
        base_indices,
        base_counts,
        1.0,
        16,
        0.01,
        16,
        keep_union=True,
    )
    assert torch.equal(kept_counts, rescue_counts)
    assert kept_union.shape[-1] == base_indices.shape[-1] + int(
        kept_counts.max().item()
    )


def test_ragged_union_keeps_base_and_per_row_rescue_budget() -> None:
    base = torch.tensor([[0, 1], [2, 3]])
    priority = torch.tensor(
        [
            [-torch.inf, -torch.inf, 4.0, 3.0],
            [2.0, 1.0, -torch.inf, -torch.inf],
        ]
    )
    mask = ragged_union_mask(base, priority, torch.tensor([1, 2]))
    assert torch.equal(mask[0], torch.tensor([True, True, True, False]))
    assert torch.equal(mask[1], torch.tensor([True, True, True, True]))


def test_selection_metrics_from_mask_reports_variable_budget() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    values = torch.eye(3)
    metrics = selection_metrics_from_mask(scores, mask, values)
    assert math.isclose(metrics["selected_tokens_mean"], 2.5)
    assert metrics["selected_tokens_maximum"] == 3
    assert metrics["attention_mass_mean"] > 0.9


from run_direct_countcap_denseprompt_ppl_20260725 import (  # noqa: E402
    PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER95_SCORE_MODE,
    PACKED_PREFILL_QUERY_SCORE_MODES,
    PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED4221_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED4421_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_PRERERANK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L18TO26_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FP16X2_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALCAL256_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALFLOORRSS25E4_S1_FULLTOPK_SCORE_MODE,
    PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QKMSE_DUALMASS975_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_DIAGONAL_NOAFFINE_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH12_WOMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SAMPLED_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH8_WOMETRIC_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE,
    PACKED_QKSIEVE_QMSE_OAS_RATE23_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_POST2X_BOUNDARY75_PRERERANK_FULLTOPK_SCORE_MODE,
    PACKED_QMSE_QKMETRIC_FIXED420_POST2X_PRERERANK_FULLTOPK_SCORE_MODE,
)


def test_global_calibration_mode_and_affine_fit() -> None:
    mode = (
        PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALCAL256_FULLTOPK_SCORE_MODE
    )
    state = {}
    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_global_risk_allocation"]
    assert state["packed_qmse_global_calibration_samples"] == 256
    proxy = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    exact = 2.0 * proxy + 7.0
    slope, intercept = _fit_affine_score_calibration(proxy, exact)
    torch.testing.assert_close(slope, torch.tensor([[2.0]]))
    torch.testing.assert_close(intercept, torch.tensor([[7.0]]))


def test_crossfit_affine_score_rmse_separates_affine_and_noisy_proxy() -> None:
    exact = torch.linspace(-3.0, 4.0, 512).repeat(2, 1)
    affine_proxy = (exact - 1.25) / 1.75
    clean_rmse, score_std = crossfit_affine_score_rmse(
        exact,
        affine_proxy,
        sample_count=64,
    )
    noisy_proxy = affine_proxy.clone()
    noisy_proxy[:, 1::2] += 0.4
    noisy_rmse, _ = crossfit_affine_score_rmse(
        exact,
        noisy_proxy,
        sample_count=64,
    )

    assert torch.all(clean_rmse < 1.0e-5)
    assert torch.all(score_std > 0.0)
    assert torch.all(noisy_rmse > 0.1)


def test_crossfit_affine_softmax_kl_separates_affine_and_noise() -> None:
    torch.manual_seed(41)
    exact = torch.randn(3, 1024)
    affine_proxy = (exact - 0.75) / 1.6
    noisy_proxy = affine_proxy + 0.8 * torch.randn_like(exact)

    clean_kl = crossfit_affine_softmax_kl(exact, affine_proxy, 256)
    noisy_kl = crossfit_affine_softmax_kl(exact, noisy_proxy, 256)

    assert clean_kl.max() < 1.0e-6
    assert noisy_kl.mean() > clean_kl.mean() + 0.05


def test_head_local_mass_floor_is_independent_per_head() -> None:
    logits = torch.tensor(
        [[[4.0, 3.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    indices, counts = _head_local_mass_floor_candidates(
        logits, minimum_top_k=1, target_mass=0.80
    )

    assert counts.tolist() == [[2, 4]]
    assert indices.shape == (1, 2, 4)
    assert set(indices[0, 0, :2].tolist()) == {0, 1}


def test_dual_mass_returns_attention_and_residual_risk_union() -> None:
    proxy = torch.tensor([[[4.0, 3.0, 0.0, 0.0]]])
    residual_log_risk = torch.tensor([[[0.0, 0.0, 5.0, -5.0]]])

    indices, counts = _head_local_dual_mass_candidates(
        proxy,
        residual_log_risk,
        minimum_top_k=1,
        target_mass=0.80,
    )

    assert counts.tolist() == [[3]]
    assert set(indices[0, 0, :3].tolist()) == {0, 1, 2}


def test_qksieve_qkmse_dual_mass_mode_contract() -> None:
    state = {}
    _configure_packed_qmse_state(
        state,
        PACKED_QKSIEVE_QKMSE_DUALMASS975_FULLTOPK_SCORE_MODE,
    )

    assert state["packed_qmse_allocation_objective"] == "qmse"
    assert state["packed_qmse_transform"] == "qk_metric"
    assert math.isclose(state["packed_qmse_dual_mass_target"], 0.975)
    assert state["packed_qmse_global_calibration_samples"] == 256
    assert state["packed_qmse_value_residual_risk_bits"] == 4
    assert state["packed_qmse_affine_value_residual"]
    assert (
        PACKED_QKSIEVE_QKMSE_DUALMASS975_FULLTOPK_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )


def test_qksieve_oas_modes_enable_covariance_shrinkage() -> None:
    for mode, expected_dual_mass, expected_prefix_rss in (
        (
            PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_NOAFFINE_FULLTOPK_SCORE_MODE,
            0.975,
            0.0,
        ),
        (
            PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE,
            0.0,
            0.0025,
        ),
        (
            PACKED_QKSIEVE_QMSE_OAS_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE,
            0.0,
            0.0025,
        ),
    ):
        state = {}
        _configure_packed_qmse_state(state, mode)

        assert state["packed_qmse_allocation_objective"] == "qmse"
        assert state["packed_qmse_covariance_shrinkage"] == "oas"
        assert math.isclose(
            state["packed_qmse_dual_mass_target"], expected_dual_mass
        )
        assert math.isclose(
            state["packed_qmse_prefix_rss_tolerance"], expected_prefix_rss
        )
        assert mode in PACKED_PREFILL_QUERY_SCORE_MODES


def test_qksieve_diagonal_residual_risk_mode_contract() -> None:
    state = {}
    mode = PACKED_QKSIEVE_QMSE_OAS_DUALMASS975_DIAGONAL_NOAFFINE_FULLTOPK_SCORE_MODE

    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_value_residual_risk_bits"] == 4
    assert state["packed_qmse_value_residual_risk_metric"] == "diagonal"
    assert math.isclose(state["packed_qmse_dual_mass_target"], 0.975)
    assert mode in head_top2._PACKED_QMSE_SCORE_MODES


def test_qksieve_rank8_value_sketch_mode_contract() -> None:
    state = {}
    mode = PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH8_WOMETRIC_FULLTOPK_SCORE_MODE

    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_value_sketch_rank"] == 8
    assert state["packed_qmse_value_sketch_bits"] == 4
    assert state["packed_qmse_value_wo_metric"]
    assert mode in head_top2._PACKED_QMSE_SCORE_MODES


def test_qksieve_rank12_value_sketch_mode_contract() -> None:
    state = {}
    mode = PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH12_WOMETRIC_FULLTOPK_SCORE_MODE

    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_value_sketch_rank"] == 12
    assert state["packed_qmse_value_sketch_bits"] == 4
    assert state["packed_qmse_value_wo_metric"]
    assert mode in head_top2._PACKED_QMSE_SCORE_MODES


def test_qksieve_rank16_sampled_value_sketch_mode_contract() -> None:
    state = {}
    mode = PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SAMPLED_SCORE_MODE

    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_covariance_shrinkage"] == "oas"
    assert state["packed_qmse_transform"] == "qk_metric"
    assert state["packed_qmse_allocation_objective"] == "qmse"
    assert state["packed_qmse_value_sketch_rank"] == 16
    assert state["packed_qmse_value_sketch_bits"] == 4
    assert state["packed_qmse_value_wo_metric"]
    assert not state["packed_qmse_full_topk"]
    assert state["packed_qmse_gqa4_scan"]
    assert mode in head_top2._QKSIEVE_FAST_RUNTIME_MODES
    assert mode in PACKED_PREFILL_QUERY_SCORE_MODES

    sorted_mode = (
        PACKED_QKSIEVE_QMSE_OAS_VALUESKETCH16_WOMETRIC_SORTED_SAMPLED_SCORE_MODE
    )
    sorted_state = {}
    _configure_packed_qmse_state(sorted_state, sorted_mode)
    assert sorted_state["packed_qmse_deterministic_compaction"]
    assert sorted_mode in head_top2._QKSIEVE_FAST_RUNTIME_MODES


def test_qksieve_rate23_mode_contract() -> None:
    state = {}
    mode = PACKED_QKSIEVE_QMSE_OAS_RATE23_PREFIXRSS25E4_S2_FULLTOPK_SCORE_MODE
    _configure_packed_qmse_state(state, mode)

    assert state["packed_qmse_allocation_objective"] == "qmse"
    assert state["packed_qmse_covariance_shrinkage"] == "oas"
    assert state["packed_qmse_bit_budget"] == 23
    assert math.isclose(state["packed_qmse_prefix_rss_tolerance"], 0.0025)
    assert math.isclose(state["packed_qmse_prefix_rss_safety_factor"], 2.0)
    assert mode in PACKED_PREFILL_QUERY_SCORE_MODES


def test_fulltopk_ragged_split_respects_shared_memory_limit() -> None:
    assert _choose_fulltopk_ragged_split_count(640) == 1
    assert _choose_fulltopk_ragged_split_count(1000) == 2
    assert _choose_fulltopk_ragged_split_count(1280) == 4
    assert _choose_fulltopk_ragged_split_count(50_000) == 8
    assert _choose_fulltopk_ragged_split_count(131_008) == 16


def test_head_local_relative_rss_uses_output_scale_and_minimum() -> None:
    proxy = torch.zeros((1, 1, 4))
    priority = torch.log(torch.tensor([[[4.0, 3.0, 2.0, 1.0]]]))
    indices, counts = _head_local_relative_rss_candidates(
        priority,
        proxy,
        output_scale=torch.ones((1, 1)),
        minimum_top_k=1,
        tolerance=0.30,
        safety_factor=1.0,
    )

    assert counts.tolist() == [[3]]
    assert set(indices[0, 0, :3].tolist()) == {0, 1, 2}


def test_proxy_value_output_scale_returns_one_scale_per_query_head() -> None:
    torch.manual_seed(0)
    value_history = torch.randn(1, 2, 8, 4)
    proxy_logits = torch.randn(1, 4, 8)
    state = {"qksieve_value_wo_head_gram": torch.eye(4).repeat(4, 1, 1)}

    scale = _proxy_value_output_scale(
        value_history,
        proxy_logits,
        state,
        rank=2,
        bits=8,
    )

    assert scale.shape == (1, 4)
    assert torch.isfinite(scale).all()
    assert torch.all(scale > 0)


def test_head_local_mass_floor_mode_contract() -> None:
    mode = (
        "pca_hierarchical_autokeytotal15z_qkmetric_"
        "valuesketch16i4shared_wometric_massfloor950_packed_fulltopk"
    )
    state = {}
    _configure_packed_qmse_state(state, mode)

    assert math.isclose(state["packed_qmse_head_local_mass_floor"], 0.95)
    assert not state["packed_qmse_global_risk_allocation"]


def test_mass_ladder_mode_contract_is_length_free() -> None:
    mode = (
        "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
        "kappend_valuesketch16i4shared_massladder900_unbiased_packed_direct"
    )
    state = {}
    _configure_packed_qmse_state(state, mode)

    assert math.isclose(state["packed_qmse_mass_ladder_target"], 0.90)
    assert state["packed_qmse_mass_ladder_floor_k"] == 1280
    assert math.isclose(state["packed_qmse_mass_ladder_growth"], 1.5)
    assert math.isclose(
        state["packed_qmse_mass_ladder_max_fraction"], 0.25
    )
    assert state["packed_qmse_value_sketch_rank"] == 16
    assert state["packed_qmse_gqa4_scan"]
    assert not state["packed_qmse_full_topk"]
    assert (
        PACKED_KEYMSE_QKMETRIC_QFUSED_GQA4_WMMA_KAPPEND_MASSLADDER95_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )


def test_exact_prefix_tail_ratio_expands_when_proxy_tail_is_biased() -> None:
    proxy = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])
    exact = torch.tensor([[5.0, 4.0, 5.0, 4.0, 1.0, 0.0, -1.0, -2.0]])

    mask = exact_prefix_tail_ratio_mass_ladder_mask(
        exact,
        proxy,
        target_mass=0.80,
        minimum_top_k=2,
        maximum_top_k=8,
        sample_count=4,
        growth=2.0,
    )

    assert mask.sum().item() == 4
    assert mask[0, :4].all()


def test_histogram_coverage_is_conservative_without_sorting() -> None:
    scores = torch.tensor(
        [[3.0, 2.9, 2.0, 1.0, -2.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    target = 0.80
    mask = histogram_coverage_mask(
        scores,
        target,
        minimum_top_k=1,
        maximum_top_k=0,
        bins=32,
        logit_range=8.0,
    )
    weights = torch.softmax(scores, dim=-1)
    retained_mass = (weights * mask).sum(dim=-1)

    assert torch.all(retained_mass >= target)
    assert mask[1].all()


def test_relative_tail_rss_uses_cancellation_aware_square_sum() -> None:
    proxy_scores = torch.zeros((1, 4))
    priority = torch.log(torch.tensor([[4.0, 3.0, 2.0, 1.0]]))
    output_scale = torch.ones(1)

    loose = relative_tail_rss_mask(
        priority,
        proxy_scores,
        output_scale,
        tolerance=1.00,
        safety_factor=1.0,
        minimum_top_k=1,
        maximum_top_k=0,
    )
    strict = relative_tail_rss_mask(
        priority,
        proxy_scores,
        output_scale,
        tolerance=0.30,
        safety_factor=1.0,
        minimum_top_k=1,
        maximum_top_k=0,
    )

    assert loose.sum().item() == 1
    assert strict.sum().item() == 3


def test_block_mean_tail_approximation_handles_partial_final_block() -> None:
    torch.manual_seed(1)
    exact_scores = torch.randn(2, 67)
    proxy_scores = exact_scores + 0.01 * torch.randn_like(exact_scores)
    value = torch.randn(67, 4)
    reconstructed_value = value + 0.01 * torch.randn_like(value)
    mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    mask[:, :8] = True

    outputs, full, _, _ = approximate_output(
        exact_scores,
        proxy_scores,
        value,
        reconstructed_value,
        mask,
        residual_sample_counts=(67,),
    )

    assert full.shape == (2, 4)
    assert outputs["hybrid_blockmean64"].shape == full.shape
    assert outputs["hybrid_blockmean256"].shape == full.shape
    torch.testing.assert_close(
        outputs["hybrid_sketch_proxyresidualsample67"],
        outputs["hybrid_full_value"],
    )
    assert all(torch.isfinite(output).all() for output in outputs.values())


def test_centered_residual_closure_is_exact_for_uniform_tail_weights() -> None:
    torch.manual_seed(2)
    scores = torch.zeros(2, 16)
    value = torch.randn(16, 4)
    reconstructed = value + 0.2 * torch.randn_like(value)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[:, :3] = True

    outputs, full, _, _ = approximate_output(
        scores,
        scores,
        value,
        reconstructed,
        mask,
    )

    torch.testing.assert_close(
        outputs["hybrid_sketch_centered_residual"], full
    )


def test_affine_residual_closure_is_exact_for_two_tail_score_levels() -> None:
    torch.manual_seed(3)
    scores = torch.tensor(
        [
            [3.0, 2.0, 1.0, 1.0, -0.5, -0.5, 1.0, -0.5],
            [2.5, 2.0, -0.2, 0.7, -0.2, 0.7, -0.2, 0.7],
        ]
    )
    value = torch.randn(8, 4)
    reconstructed = value + 0.2 * torch.randn_like(value)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[:, :2] = True

    outputs, full, _, _ = approximate_output(
        scores,
        scores,
        value,
        reconstructed,
        mask,
    )

    torch.testing.assert_close(
        outputs["hybrid_sketch_affine_residual"], full, atol=2.0e-6, rtol=2.0e-6
    )


def test_affine_residual_bound_ladder_is_monotone_in_tolerance() -> None:
    scores = torch.linspace(2.0, -2.0, 64).reshape(1, -1)
    residual_risk = torch.linspace(0.5, 1.5, 64).reshape(1, -1)
    output_scale = torch.ones(1)
    loose = affine_residual_bound_ladder_mask(
        scores,
        residual_risk,
        output_scale,
        0.50,
        4,
        64,
        sample_count=64,
        growth=2.0,
    )
    strict = affine_residual_bound_ladder_mask(
        scores,
        residual_risk,
        output_scale,
        0.05,
        4,
        64,
        sample_count=64,
        growth=2.0,
    )
    assert int(strict.sum()) >= int(loose.sum())
    assert torch.all(strict | ~loose)


def test_runtime_affine_residual_chunking_matches_direct_formula() -> None:
    torch.manual_seed(4)
    batch_count, kv_heads, groups, history, dimension, rank = 1, 2, 2, 17, 6, 3
    value_mean = torch.randn(batch_count, kv_heads, dimension)
    value_basis = torch.randn(batch_count, kv_heads, dimension, rank)
    coefficients = torch.randn(batch_count, kv_heads, history, rank)
    reconstructed = value_mean.unsqueeze(2) + torch.einsum(
        "bhnr,bhdr->bhnd", coefficients, value_basis
    )
    value = reconstructed + 0.2 * torch.randn_like(reconstructed)
    logits = torch.randn(batch_count, kv_heads * groups, history)
    selected = torch.zeros_like(logits, dtype=torch.bool)
    selected[..., :4] = True
    anchor = logits.amax(dim=-1, keepdim=True)
    tail_weights = torch.exp(logits - anchor).masked_fill(selected, 0.0)

    actual = _affine_value_residual_tail_correction(
        value,
        value_mean,
        value_basis,
        coefficients,
        logits,
        tail_weights,
        selected,
        chunk_size=5,
    )

    residual = (value - reconstructed).repeat_interleave(groups, dim=1)
    tail = (~selected).float()
    count = tail.sum(dim=-1).clamp_min(1.0)
    score_mean = (tail * logits).sum(dim=-1) / count
    centered = (logits - score_mean.unsqueeze(-1)) * tail
    slope = (centered * tail_weights).sum(dim=-1) / centered.square().sum(
        dim=-1
    ).clamp_min(1.0e-20)
    expected = (
        (tail_weights.sum(dim=-1) / count).unsqueeze(-1)
        * torch.einsum("bhn,bhnd->bhd", tail, residual)
        + slope.unsqueeze(-1)
        * torch.einsum("bhn,bhnd->bhd", centered, residual)
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_global_floor_rss_never_removes_per_head_floor() -> None:
    proxy = torch.tensor(
        [[[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]]
    )
    risk = proxy + torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]
    )
    loose = global_floor_rss_mask(
        risk,
        proxy,
        torch.ones(1),
        floor_k=1,
        tolerance=10.0,
        safety_factor=1.0,
    )
    strict = global_floor_rss_mask(
        risk,
        proxy,
        torch.ones(1),
        floor_k=1,
        tolerance=0.01,
        safety_factor=1.0,
    )

    assert torch.all(loose.sum(dim=-1) >= 1)
    assert torch.all(strict.sum(dim=-1) >= 1)
    assert strict.sum().item() > loose.sum().item()


def test_sampled_mass_prefix_matches_full_sample_prefix() -> None:
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    mask = sampled_mass_prefix_mask(
        scores,
        0.8,
        1,
        0,
        sample_count=4,
        aggregation="minimum",
    )
    exact = torch.softmax(scores, dim=-1)
    assert mask.tolist() == [[True, True, False, False]]
    assert float(exact.masked_fill(~mask, 0.0).sum()) >= 0.8


def test_gaussian_mass_prefix_matches_exponential_tilt_formula() -> None:
    normal = torch.distributions.Normal(0.0, 1.0)
    quantiles = torch.linspace(0.0001, 0.9999, 10000)
    scores = normal.icdf(quantiles).unsqueeze(0)
    mask = gaussian_mass_prefix_mask(
        scores,
        0.9,
        1,
        0,
        sample_count=scores.shape[-1],
    )
    retained_mass = torch.softmax(scores, dim=-1).masked_fill(
        ~mask, 0.0
    ).sum()
    assert abs(float(retained_mass) - 0.9) < 0.01


def test_sampled_rank_mass_ladder_measures_full_proxy_mass() -> None:
    scores = torch.linspace(-4.0, 4.0, 1024).unsqueeze(0)
    mask = sampled_rank_mass_ladder_mask(
        scores,
        0.9,
        16,
        0,
        sample_count=256,
        growth=2.0,
    )
    retained_mass = torch.softmax(scores, dim=-1).masked_fill(
        ~mask, 0.0
    ).sum()
    assert float(retained_mass) >= 0.9
    assert 16 <= int(mask.sum()) < scores.shape[-1]


def test_interval_mass_ladder_certifies_true_softmax_mass() -> None:
    scores = torch.linspace(-4.0, 4.0, 1024).unsqueeze(0)
    error_bound = 0.04 + 0.08 * torch.linspace(0.0, 1.0, 1024).unsqueeze(0)
    signs = torch.where(
        torch.arange(1024).remainder(3) == 0,
        -torch.ones(1024),
        torch.ones(1024),
    ).unsqueeze(0)
    true_scores = scores + signs * error_bound
    mask = interval_certified_mass_ladder_mask(
        scores,
        error_bound,
        0.9,
        16,
        0,
        sample_count=256,
        growth=1.5,
    )
    retained_mass = torch.softmax(true_scores, dim=-1).masked_fill(
        ~mask, 0.0
    ).sum()
    assert float(retained_mass) >= 0.9
    assert 16 <= int(mask.sum()) < scores.shape[-1]


def test_scalar_residual_rss_mask_obeys_probability_bound() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
    residual = torch.full_like(scores, 2.0)
    output_scale = torch.tensor([1.0])
    tolerance = 0.5
    mask = scalar_residual_rss_mask(
        scores,
        residual,
        output_scale,
        tolerance,
        1.0,
        1,
        0,
        "maximum",
    )
    probabilities = torch.softmax(scores, dim=-1)
    omitted_rss = torch.sqrt(
        torch.sum((probabilities.masked_fill(mask, 0.0) * residual) ** 2)
    )
    assert omitted_rss <= tolerance * output_scale[0] + 1.0e-6


def test_runtime_global_floor_rss_preserves_floor_and_is_monotone() -> None:
    proxy = torch.tensor(
        [[[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]]
    )
    risk = proxy + torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]
    )
    loose_indices, loose_counts = _global_floor_relative_rss_candidates(
        risk,
        proxy,
        torch.ones(1, 2),
        minimum_top_k=1,
        tolerance=10.0,
        safety_factor=1.0,
    )
    strict_indices, strict_counts = _global_floor_relative_rss_candidates(
        risk,
        proxy,
        torch.ones(1, 2),
        minimum_top_k=1,
        tolerance=0.01,
        safety_factor=1.0,
    )

    assert torch.all(loose_counts >= 1)
    assert torch.all(strict_counts >= 1)
    assert strict_counts.sum().item() > loose_counts.sum().item()
    for head, expected_floor_token in enumerate((0, 3)):
        assert expected_floor_token in loose_indices[0, head, : loose_counts[0, head]]
        assert expected_floor_token in strict_indices[0, head, : strict_counts[0, head]]


def test_layer_equal_prefix_rss_returns_proxy_prefixes() -> None:
    proxy = torch.tensor(
        [[[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]]
    )
    risk = proxy + torch.tensor(
        [[[0.0, 0.0, 2.0, 0.0], [2.0, 0.0, 0.0, 0.0]]]
    )
    loose_indices, loose_counts = _layer_equal_prefix_rss_candidates(
        risk,
        proxy,
        torch.ones(1, 2),
        minimum_top_k=1,
        tolerance=10.0,
        safety_factor=1.0,
    )
    strict_indices, strict_counts = _layer_equal_prefix_rss_candidates(
        risk,
        proxy,
        torch.ones(1, 2),
        minimum_top_k=1,
        tolerance=0.01,
        safety_factor=1.0,
    )

    expected_orders = torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]]])
    assert torch.all(strict_counts >= loose_counts)
    for head in range(2):
        loose_count = int(loose_counts[0, head])
        strict_count = int(strict_counts[0, head])
        assert torch.equal(
            loose_indices[0, head, :loose_count],
            expected_orders[0, head, :loose_count],
        )
        assert torch.equal(
            strict_indices[0, head, :strict_count],
            expected_orders[0, head, :strict_count],
        )


def test_value_residual_risk_uses_incremental_open_block() -> None:
    torch.manual_seed(7)
    state = {
        "qksieve_value_wo_group_gram": torch.eye(4).unsqueeze(0),
    }
    values = torch.randn(1, 1, 8, 4)

    risk6 = _value_residual_priority_log_risk(
        values[..., :6, :],
        state,
        rank=2,
        bits=4,
        risk_bits=4,
        block_size=4,
    )
    assert risk6.shape == (1, 1, 6)
    assert torch.isfinite(risk6).all()
    assert state["qksieve_value_residual_risk_open_buffer_tokens"] == 2

    risk7 = _value_residual_priority_log_risk(
        values[..., :7, :],
        state,
        rank=2,
        bits=4,
        risk_bits=4,
        block_size=4,
    )
    assert torch.equal(risk7[..., :6], risk6)
    assert state["qksieve_value_residual_risk_open_buffer_tokens"] == 3

    risk8 = _value_residual_priority_log_risk(
        values,
        state,
        rank=2,
        bits=4,
        risk_bits=4,
        block_size=4,
    )
    assert torch.isfinite(risk8).all()
    assert state["qksieve_value_residual_risk_open_buffer_tokens"] == 0


def test_global_floor_rss_mode_configuration(monkeypatch) -> None:
    monkeypatch.delenv("QKSIEVE_GLOBAL_FLOOR_RSS_LAYERS", raising=False)
    state = {}
    _configure_packed_qmse_state(
        state,
        PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALFLOORRSS25E4_S1_FULLTOPK_SCORE_MODE,
    )

    assert state["packed_qmse_global_floor_rss_tolerance"] == 0.0025
    assert state["packed_qmse_global_floor_rss_safety_factor"] == 1.0
    assert state["packed_qmse_head_local_rss_tolerance"] == 0.0

    monkeypatch.setenv("QKSIEVE_GLOBAL_FLOOR_RSS_LAYERS", "0,2")
    inactive_state = {"layer_index": 1}
    _configure_packed_qmse_state(
        inactive_state,
        PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_GLOBALFLOORRSS25E4_S1_FULLTOPK_SCORE_MODE,
    )
    assert inactive_state["packed_qmse_global_floor_rss_tolerance"] == 0.0
    assert inactive_state["packed_qmse_value_residual_risk_bits"] == 0

    prefix_state = {}
    _configure_packed_qmse_state(
        prefix_state,
        PACKED_QKBALANCED_AUTOKEY_VALUESKETCH16I4_WOMETRIC_RESIDUALRISK4_PREFIXRSS25E4_S1_FULLTOPK_SCORE_MODE,
    )
    assert prefix_state["packed_qmse_prefix_rss_tolerance"] == 0.0025
    assert prefix_state["packed_qmse_prefix_rss_safety_factor"] == 1.0


def test_relative_tail_risk_mask_uses_head_local_error_scale() -> None:
    priority = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    proxy_scores = torch.zeros_like(priority)
    output_scale = torch.ones(1)

    strict = relative_tail_risk_mask(
        priority,
        proxy_scores,
        output_scale,
        tolerance=0.05,
        minimum_top_k=1,
        maximum_top_k=0,
    )
    loose = relative_tail_risk_mask(
        priority,
        proxy_scores,
        output_scale,
        tolerance=0.20,
        minimum_top_k=1,
        maximum_top_k=0,
    )

    assert strict.tolist() == [[True, True, False]]
    assert loose.tolist() == [[True, False, False]]


def test_length_robust_fixed_allocations_are_registered() -> None:
    expected = {
        PACKED_QMSE_QKMETRIC_FIXED4221_FULLTOPK_SCORE_MODE: (
            4,
            2,
            2,
            1,
            0,
            0,
            0,
            0,
        ),
        PACKED_QMSE_QKMETRIC_FIXED4421_FULLTOPK_SCORE_MODE: (
            4,
            4,
            2,
            1,
            0,
            0,
            0,
            0,
        ),
    }
    for score_mode, allocation in expected.items():
        state = {}
        _configure_packed_qmse_state(state, score_mode)
        assert state["packed_qmse_transform"] == "qk_metric"
        assert state["packed_qmse_full_topk"]
        assert state["packed_qmse_fixed_allocation"] == allocation
        assert score_mode in PACKED_PREFILL_QUERY_SCORE_MODES


def test_fp16_first_two_band_mode_contract() -> None:
    state = {}
    _configure_packed_qmse_state(
        state,
        PACKED_QMSE_QKMETRIC_FP16X2_FULLTOPK_SCORE_MODE,
    )

    assert state["packed_qmse_transform"] == "qk_metric"
    assert state["packed_qmse_full_topk"]
    assert state["packed_qmse_fp16_first_two_bands"]
    assert state["packed_qmse_fixed_allocation"] == (
        16,
        16,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_exact_sparse_selection_metrics_match_manual_topk_and_mass() -> None:
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor(
        [[[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]]
    )
    candidates = torch.tensor([[[0, 2]]])

    metrics = _exact_sparse_selection_metrics(query, key, candidates, 1.0)
    probabilities = torch.softmax(torch.tensor([3.0, 2.0, 1.0, 0.0]), 0)

    torch.testing.assert_close(
        metrics["exact_topk_recall"], torch.tensor([[0.5]])
    )
    torch.testing.assert_close(
        metrics["selected_attention_mass"],
        (probabilities[0] + probabilities[2]).reshape(1, 1),
    )
    torch.testing.assert_close(
        metrics["oracle_topk_attention_mass"],
        (probabilities[0] + probabilities[1]).reshape(1, 1),
    )


def _apply_standard_rope(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    rope_theta: float,
) -> torch.Tensor:
    head_dim = hidden.shape[-1]
    half = head_dim // 2
    inverse_frequency = torch.pow(
        torch.tensor(rope_theta, dtype=torch.float32),
        -2.0 * torch.arange(half, dtype=torch.float32) / float(head_dim),
    )
    angle = positions.float().unsqueeze(-1) * inverse_frequency
    cosine = angle.cos()
    sine = angle.sin()
    first = hidden[..., :half]
    second = hidden[..., half:]
    return torch.cat(
        (
            first * cosine - second * sine,
            second * cosine + first * sine,
        ),
        dim=-1,
    )


def test_inverse_standard_rope_recovers_split_half_vectors() -> None:
    generator = torch.Generator().manual_seed(20260731)
    hidden = torch.randn((2, 3, 7, 8), generator=generator)
    positions = torch.arange(7).view(1, 1, 7).expand(2, 3, 7)
    post = _apply_standard_rope(hidden, positions, 5_000_000.0)
    recovered = _inverse_standard_rope(post, positions, 5_000_000.0)

    torch.testing.assert_close(recovered, hidden, rtol=2e-5, atol=2e-5)


def test_configured_unit_rope_pair_tables_are_normalized_and_cached() -> None:
    config = object()
    state = {"model_config": config}
    calls = 0
    original = head_top2._configured_rope_phase_tables
    head_top2._CONFIGURED_UNIT_ROPE_PAIR_PHASE_TABLE_CACHE.clear()

    def fake_tables(
        minimum_rows: int,
        head_dim: int,
        state: dict[str, object],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        del minimum_rows, state, device
        calls += 1
        cosine = torch.full((4096, head_dim), 1.5)
        sine = torch.full((4096, head_dim), 2.0)
        return cosine, sine

    head_top2._configured_rope_phase_tables = fake_tables
    try:
        first = _configured_unit_rope_pair_phase_tables(
            1024,
            8,
            state,
            torch.device("cpu"),
        )
        second = _configured_unit_rope_pair_phase_tables(
            2048,
            8,
            state,
            torch.device("cpu"),
        )
    finally:
        head_top2._configured_rope_phase_tables = original
        head_top2._CONFIGURED_UNIT_ROPE_PAIR_PHASE_TABLE_CACHE.clear()

    assert calls == 1
    assert first[0].shape == (4096, 4)
    torch.testing.assert_close(first[0], torch.full((4096, 4), 0.6))
    torch.testing.assert_close(first[1], torch.full((4096, 4), 0.8))
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()


def test_post_overfetch_reranks_pool_by_exact_prerope_score() -> None:
    generator = torch.Generator().manual_seed(314159)
    history_count = 10
    rope_theta = 5_000_000.0
    key_pre = torch.randn(
        (1, 1, history_count, 4),
        generator=generator,
    )
    query_pre = torch.randn((1, 2, 4), generator=generator)
    key_post = _apply_standard_rope(
        key_pre,
        torch.arange(history_count).view(1, 1, history_count),
        rope_theta,
    )
    query_post = _apply_standard_rope(
        query_pre,
        torch.full((1, 2), history_count),
        rope_theta,
    )
    proxy_scores = torch.full((1, 2, history_count), -100.0)
    proxy_scores[..., 1:9] = torch.tensor(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    )

    selected = _post_overfetch_prerope_rerank_candidates(
        proxy_scores,
        query_post,
        key_post,
        target_count=5,
        rope_theta=rope_theta,
        overfetch_factor=2,
        sink_tokens=1,
        recent_tokens=1,
    )

    assert selected.shape == (1, 2, 5)
    for head in range(2):
        direct_scores = torch.matmul(
            key_pre[0, 0, 1:7],
            query_pre[0, head],
        )
        expected_remote = (
            torch.topk(direct_scores, k=3, sorted=False).indices + 1
        )
        assert set(selected[0, head].tolist()) == {
            0,
            9,
            *expected_remote.tolist(),
        }


def test_dual_mass_rerank_maximizes_average_pre_post_candidate_mass() -> None:
    generator = torch.Generator().manual_seed(271828)
    history_count = 6
    rope_theta = 5_000_000.0
    key_pre = torch.randn((1, 1, history_count, 4), generator=generator)
    query_pre = torch.randn((1, 1, 4), generator=generator)
    key_post = _apply_standard_rope(
        key_pre,
        torch.arange(history_count).view(1, 1, history_count),
        rope_theta,
    )
    query_post = _apply_standard_rope(
        query_pre,
        torch.full((1, 1), history_count),
        rope_theta,
    )
    proxy_scores = torch.randn(
        (1, 1, history_count),
        generator=generator,
    )
    scaling = 0.5

    selected = _post_overfetch_prerope_rerank_candidates(
        proxy_scores,
        query_post,
        key_post,
        target_count=3,
        rope_theta=rope_theta,
        overfetch_factor=2,
        sink_tokens=0,
        recent_tokens=0,
        scaling=scaling,
        selection_mode="dual_mass",
    )
    pre_scores = torch.einsum(
        "bhnd,bhd->bhn",
        key_pre,
        query_pre,
    ) * scaling
    post_scores = torch.einsum(
        "bhnd,bhd->bhn",
        key_post,
        query_post,
    ) * scaling
    mixture_mass = (
        torch.softmax(pre_scores, dim=-1)
        + torch.softmax(post_scores, dim=-1)
    )
    expected = torch.topk(mixture_mass, k=3, dim=-1).indices

    assert torch.equal(
        torch.sort(selected, dim=-1).values,
        torch.sort(expected, dim=-1).values,
    )


def test_post2x_qksieve_mode_captures_prefill_queries() -> None:
    assert (
        PACKED_QMSE_QKMETRIC_FIXED420_POST2X_PRERERANK_FULLTOPK_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )
    assert (
        PACKED_QMSE_QKMETRIC_FIXED410_POST2X_PRERERANK_FULLTOPK_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )
    assert (
        PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )


def test_post2x_qksieve_layer_range_is_applied_per_layer() -> None:
    inside = {"layer_index": 22}
    outside = {"layer_index": 17}
    _configure_packed_qmse_state(
        inside,
        PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L18TO26_FULLTOPK_SCORE_MODE,
    )
    _configure_packed_qmse_state(
        outside,
        PACKED_QMSE_QKMETRIC_FIXED410_POST2X_L18TO26_FULLTOPK_SCORE_MODE,
    )
    assert inside["packed_qmse_post2x_prerope_rerank"]
    assert not outside["packed_qmse_post2x_prerope_rerank"]
    assert inside["packed_qmse_prerope_rerank_layer_start"] == 18
    assert inside["packed_qmse_prerope_rerank_layer_end"] == 26


def test_post2x_dual_mass_mode_is_limited_to_requested_layers() -> None:
    inside = {"layer_index": 8}
    outside = {"layer_index": 9}
    _configure_packed_qmse_state(
        inside,
        PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE,
    )
    _configure_packed_qmse_state(
        outside,
        PACKED_QMSE_QKMETRIC_FIXED410_POST2X_DUALMASS_L00TO08_FULLTOPK_SCORE_MODE,
    )

    assert inside["packed_qmse_post2x_prerope_rerank"]
    assert inside["packed_qmse_prerope_rerank_selection"] == "dual_mass"
    assert not outside["packed_qmse_post2x_prerope_rerank"]


def test_boundary_rerank_preserves_proxy_core_and_repairs_tail() -> None:
    generator = torch.Generator().manual_seed(271828)
    history_count = 10
    rope_theta = 5_000_000.0
    key_pre = torch.randn(
        (1, 1, history_count, 4),
        generator=generator,
    )
    query_pre = torch.randn((1, 2, 4), generator=generator)
    key_post = _apply_standard_rope(
        key_pre,
        torch.arange(history_count).view(1, 1, history_count),
        rope_theta,
    )
    query_post = _apply_standard_rope(
        query_pre,
        torch.full((1, 2), history_count),
        rope_theta,
    )
    proxy_scores = torch.full((1, 2, history_count), -100.0)
    proxy_scores[..., 1:9] = torch.tensor(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    )

    selected = _post_overfetch_prerope_rerank_candidates(
        proxy_scores,
        query_post,
        key_post,
        target_count=5,
        rope_theta=rope_theta,
        overfetch_factor=2,
        sink_tokens=1,
        recent_tokens=1,
        core_fraction=0.75,
    )

    for head in range(2):
        boundary_scores = torch.matmul(
            key_pre[0, 0, 3:7],
            query_pre[0, head],
        )
        expected_boundary = int(torch.argmax(boundary_scores).item()) + 3
        assert set(selected[0, head].tolist()) == {
            0,
            1,
            2,
            9,
            expected_boundary,
        }


def test_boundary75_qksieve_mode_captures_prefill_queries() -> None:
    assert (
        PACKED_QMSE_QKMETRIC_FIXED420_POST2X_BOUNDARY75_PRERERANK_FULLTOPK_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )


def test_full_prerope_qksieve_mode_captures_prefill_queries() -> None:
    assert (
        PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_FULLTOPK_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )
    assert (
        PACKED_QMSE_QKMETRIC_FIXED420_FULL_PREROPE_LOCALSINK_FULLTOPK_SCORE_MODE
        in PACKED_PREFILL_QUERY_SCORE_MODES
    )


def test_sparse_attention_autosplit_avoids_short_underfilled_work() -> None:
    assert _choose_sparse_attention_split_count(32, 1_281, 82) == 0
    assert _choose_sparse_attention_split_count(32, 2_561, 82) == 16


def test_sparse_attention_autosplit_adapts_to_sm_count() -> None:
    assert _choose_sparse_attention_split_count(32, 2_561, 40) == 8
    assert _choose_sparse_attention_split_count(64, 2_561, 40) == 4


def test_systematic_tail_samples_are_unique_per_head() -> None:
    indices = _systematic_tail_sample_indices(2, 4, 128, 13, torch.device("cpu"))
    assert indices.shape == (2, 4, 13)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 128
    for row in indices.reshape(-1, 13):
        assert torch.unique(row).numel() == row.numel()


def test_exact_top2_keeps_two_of_one_hundred_history_tokens_plus_self() -> None:
    query = torch.ones((1, 2, 1, 1), dtype=torch.float32)
    key = torch.arange(1, 102, dtype=torch.float32).view(1, 1, 101, 1)
    value = key.clone()

    output, indices = exact_head_top_fraction_attention(
        query, key, value, attention_mask=None, scaling=1.0, top_fraction=0.02
    )

    assert indices.shape == (1, 2, 1, 3)
    assert set(indices[0, 0, 0].tolist()) == {98, 99, 100}
    assert torch.isfinite(output).all()


def test_exact_top_k_keeps_an_absolute_history_budget_plus_self() -> None:
    query = torch.ones((1, 2, 1, 1), dtype=torch.float32)
    key = torch.arange(1, 102, dtype=torch.float32).view(1, 1, 101, 1)
    value = key.clone()
    diagnostics: dict[str, object] = {}

    output, indices = exact_head_top_fraction_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0,
        top_k=3,
        diagnostics=diagnostics,
    )

    assert indices.shape == (1, 2, 1, 4)
    assert set(indices[0, 0, 0].tolist()) == {97, 98, 99, 100}
    assert int(diagnostics["selected_count"]) == 4
    assert torch.isfinite(output).all()


def test_exact_top_k_rejects_ambiguous_or_invalid_budgets() -> None:
    query = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    key = torch.ones((1, 1, 2, 1), dtype=torch.float32)
    value = key.clone()

    for kwargs in (
        {},
        {"top_fraction": 0.5, "top_k": 1},
        {"top_k": 0},
    ):
        try:
            exact_head_top_fraction_attention(
                query,
                key,
                value,
                attention_mask=None,
                scaling=1.0,
                **kwargs,
            )
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_sparse_aggregation_matches_dense_masked_reference() -> None:
    torch.manual_seed(7)
    query = torch.randn((1, 4, 1, 3), dtype=torch.float32)
    key = torch.randn((1, 2, 51, 3), dtype=torch.float32)
    value = torch.randn((1, 2, 51, 3), dtype=torch.float32)
    scaling = 1.0 / math.sqrt(3.0)

    output, indices = exact_head_top_fraction_attention(
        query, key, value, attention_mask=None, scaling=scaling, top_fraction=0.10
    )

    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = torch.matmul(query, expanded_key.transpose(2, 3)) * scaling
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, indices, True)
    dense_weights = torch.softmax(scores.masked_fill(~keep, torch.finfo(scores.dtype).min), dim=-1)
    reference = torch.matmul(dense_weights, expanded_value).transpose(1, 2).contiguous()

    torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)


def test_sampled_quantile_budget_saturation_uses_all_history() -> None:
    generator = torch.Generator().manual_seed(20260726)
    query = torch.randn((1, 4, 1, 8), generator=generator)
    key = torch.randn((1, 2, 17, 8), generator=generator)
    value = torch.randn((1, 2, 17, 8), generator=generator)
    diagnostics: dict[str, object] = {}

    output, selected = qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=8**-0.5,
        mass_threshold=1.0e-6,
        budget_fractions=(1.0,),
        sample_fraction=1.0,
        qabs_dim_count=8,
        candidate_fraction=1.0,
        diagnostics=diagnostics,
        use_cuda_kernels=False,
        score_mode=(
            "pca_int4_chunked_logscale16_sampleq_direct_qkvfused_"
            "qprojscan_qkvsplitauto"
        ),
        projection_dim=8,
        pca_state={},
        gqa_candidate_mode="independent",
    )
    reference, reference_selected = exact_head_top_fraction_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=8**-0.5,
        top_fraction=1.0,
    )

    torch.testing.assert_close(output, reference, rtol=1e-6, atol=1e-6)
    assert torch.equal(selected, reference_selected)
    assert diagnostics["budget_saturated_full_history"] == 1.0


def test_current_token_is_kept_when_there_is_no_history() -> None:
    query = torch.tensor([[[[2.0]]]])
    key = torch.tensor([[[[3.0]]]])
    value = torch.tensor([[[[5.0]]]])

    output, indices = exact_head_top_fraction_attention(
        query, key, value, attention_mask=None, scaling=1.0, top_fraction=0.02
    )

    assert indices.tolist() == [[[[0]]]]
    torch.testing.assert_close(output, torch.tensor([[[[5.0]]]]))


def test_attention_mass_diagnostics_are_bounded() -> None:
    query = torch.ones((1, 2, 1, 1), dtype=torch.float32)
    key = torch.arange(1, 102, dtype=torch.float32).view(1, 1, 101, 1) / 20.0
    value = key.clone()
    diagnostics: dict[str, object] = {}

    exact_head_top_fraction_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0,
        top_fraction=0.02,
        diagnostics=diagnostics,
    )

    retained = diagnostics["retained_attention_mass"]
    top1 = diagnostics["top1_attention_mass"]
    assert isinstance(retained, torch.Tensor)
    assert isinstance(top1, torch.Tensor)
    assert torch.all((retained > 0.0) & (retained <= 1.0))
    assert torch.all((top1 > 0.0) & (top1 <= retained))


def test_adaptive_mass_selects_smallest_sufficient_budget() -> None:
    query = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    key = torch.tensor([[[[4.0], [3.0], [2.0], [1.0], [0.0]]]])
    value = key.clone()
    diagnostics: dict[str, object] = {}

    output, indices = exact_head_adaptive_mass_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0,
        mass_threshold=0.80,
        budget_fractions=(0.25, 0.50, 1.0),
        diagnostics=diagnostics,
    )

    assert indices.shape == (1, 1, 1, 5)
    assert float(diagnostics["selected_budget_fraction"].item()) == 0.5
    assert float(diagnostics["selected_history_fraction"].item()) == 0.5
    expected_scores = torch.tensor([4.0, 3.0, 0.0])
    expected_values = torch.tensor([4.0, 3.0, 0.0])
    expected = (torch.softmax(expected_scores, dim=0) * expected_values).sum()
    torch.testing.assert_close(output.squeeze(), expected)


def test_adaptive_mass_uses_maximum_rung_when_threshold_is_unreachable() -> None:
    query = torch.ones((1, 1, 1, 1), dtype=torch.float32)
    key = torch.zeros((1, 1, 101, 1), dtype=torch.float32)
    value = torch.arange(101, dtype=torch.float32).view(1, 1, 101, 1)
    diagnostics: dict[str, object] = {}

    output, _ = exact_head_adaptive_mass_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0,
        mass_threshold=0.95,
        budget_fractions=(0.01, 0.02, 0.04),
        diagnostics=diagnostics,
    )

    assert torch.isfinite(output).all()
    assert math.isclose(float(diagnostics["selected_budget_fraction"].item()), 0.04, abs_tol=1e-7)
    assert math.isclose(float(diagnostics["selected_history_fraction"].item()), 0.04, abs_tol=1e-7)


def test_sampled_tail_mass_uses_valid_budget_and_reports_actual_mass() -> None:
    torch.manual_seed(11)
    query = torch.randn((1, 2, 1, 3), dtype=torch.float32)
    key = torch.randn((1, 1, 101, 3), dtype=torch.float32)
    value = torch.randn((1, 1, 101, 3), dtype=torch.float32)
    diagnostics: dict[str, object] = {}

    output, _ = exact_head_adaptive_mass_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0 / math.sqrt(3.0),
        mass_threshold=0.8,
        budget_fractions=(0.01, 0.02, 0.04),
        sample_fraction=0.1,
        sample_offset=3,
        diagnostics=diagnostics,
    )

    assert torch.isfinite(output).all()
    retained = diagnostics["retained_attention_mass"]
    estimated = diagnostics["estimated_attention_mass"]
    assert torch.all((retained > 0.0) & (retained <= 1.0))
    assert torch.all((estimated > 0.0) & (estimated <= 1.0))


def test_qabs_sampled_mass_path_is_finite_without_full_qk() -> None:
    torch.manual_seed(13)
    query = torch.randn((1, 2, 1, 4), dtype=torch.float32)
    key = torch.randn((1, 1, 101, 4), dtype=torch.float32)
    value = torch.randn((1, 1, 101, 4), dtype=torch.float32)
    diagnostics: dict[str, object] = {}

    output, indices = qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=0.5,
        mass_threshold=0.8,
        budget_fractions=(0.01, 0.02, 0.04),
        sample_fraction=0.1,
        qabs_dim_count=2,
        candidate_fraction=0.2,
        diagnostics=diagnostics,
    )

    assert torch.isfinite(output).all()
    assert indices.shape[-1] == 5
    assert float(diagnostics["qabs_dim_fraction"]) == 0.5
    assert 0.0 < float(diagnostics["candidate_fraction"]) <= 0.2


def test_pca_int4_index_path_is_finite_and_incremental() -> None:
    torch.manual_seed(17)
    query = torch.randn((1, 2, 1, 4), dtype=torch.float32)
    key = torch.randn((1, 1, 101, 4), dtype=torch.float32)
    value = torch.randn((1, 1, 101, 4), dtype=torch.float32)
    state: dict[str, object] = {}

    output, _ = qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=0.5,
        mass_threshold=0.8,
        budget_fractions=(0.01, 0.02, 0.04),
        sample_fraction=0.1,
        qabs_dim_count=2,
        candidate_fraction=0.2,
        score_mode="pca_int4",
        projection_dim=2,
        pca_state=state,
    )

    assert torch.isfinite(output).all()
    assert int(state["indexed_count"]) == 100
    assert state["packed"].dtype == torch.uint8
