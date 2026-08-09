from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_head_top2_targeted_ppl_20260714 as targeted  # noqa: E402
import qksieve_query_cuda_20260728 as qksieve_query  # noqa: E402
import validate_qksieve_qfused_cuda_20260728 as qfused_validate  # noqa: E402
import validate_qksieve_qfused_matrix_20260728 as qfused_matrix  # noqa: E402
import variablebit_spectral_cuda_20260727 as variablebit  # noqa: E402
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _PACKED_QMSE_FIXED_ALLOCATIONS,
    _deterministic_random_orthogonal_basis,
    _expand_packed_qmse_fixed_allocation,
    _hierarchical_qmse_rate_allocation,
    _hierarchical_qmse_rate_allocation_reference,
    _oas_shrink_query_band_second_moments,
    _packed_qmse_mode_contract,
    _packed_qmse_should_fuse_query_prepare,
    _packed_qmse_uses_fused_query_prepare,
    _resolve_packed_qmse_sample_count,
    exact_head_top_fraction_attention,
    qabs_sampled_head_adaptive_attention,
)
from run_hierarchical_physical_cache_ppl_20260715 import (  # noqa: E402
    parse_fixed_bit_allocation,
)


def test_sample_count_is_clamped_to_short_history() -> None:
    assert _resolve_packed_qmse_sample_count(171, 256, 0.06) == 171


def test_sample_count_preserves_configured_and_statistical_floors() -> None:
    assert _resolve_packed_qmse_sample_count(4096, 512, 0.06) == 512
    assert _resolve_packed_qmse_sample_count(4096, 256, 0.01) == 1600


def test_sample_count_obeys_kernel_limit() -> None:
    assert _resolve_packed_qmse_sample_count(8192, 4096, 0.01) == 2048


def test_sample_count_accepts_explicit_larger_kernel_limit() -> None:
    assert (
        _resolve_packed_qmse_sample_count(
            8192,
            4096,
            0.01,
            maximum_sample_count=8192,
        )
        == 4096
    )


def test_sample_count_obeys_runtime_statistical_cap(monkeypatch) -> None:
    monkeypatch.setenv("QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT", "512")
    assert _resolve_packed_qmse_sample_count(131072, 256, 0.01) == 512
    assert _resolve_packed_qmse_sample_count(8192, 256, 0.06) == 267


def test_sample_count_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="non-empty history"):
        _resolve_packed_qmse_sample_count(0, 256, 0.06)


def test_fixed4421_expands_per_kv_head_at_exact_240_bit_rate() -> None:
    schedule = _PACKED_QMSE_FIXED_ALLOCATIONS[
        "pca_hierarchical_fixed4421_qkmetric_packed_direct"
    ]
    allocation = _expand_packed_qmse_fixed_allocation(
        schedule,
        batch_count=2,
        kv_head_count=3,
        device=torch.device("cpu"),
    )
    assert allocation.shape == (2, 3, 8)
    assert allocation.dtype == torch.int8
    assert allocation[1, 2].tolist() == [4, 4, 2, 1, 0, 0, 0, 0]
    code_bits = sum(schedule) * 16
    scale_bits = sum(bits > 0 for bits in schedule) * 16
    assert code_bits + scale_bits == 240


@pytest.mark.parametrize(
    "score_mode",
    (
        "pca_hierarchical_fixed11111111_packed_fulltopk",
        "pca_hierarchical_fixed11111111_qkmetric_packed_fulltopk",
        "pca_hierarchical_fixed11111111_random_packed_fulltopk",
    ),
)
def test_uniform1_ablation_has_the_same_256_bit_rate_as_fier(
    score_mode,
) -> None:
    schedule = _PACKED_QMSE_FIXED_ALLOCATIONS[score_mode]
    allocation = _expand_packed_qmse_fixed_allocation(
        schedule,
        batch_count=1,
        kv_head_count=2,
        device=torch.device("cpu"),
    )

    assert allocation[0, 1].tolist() == [1] * 8
    code_bits = sum(schedule) * 16
    scale_bits = sum(bits > 0 for bits in schedule) * 16
    assert code_bits + scale_bits == 256


def test_random_basis_is_reproducible_orthogonal_and_layer_specific() -> None:
    first = _deterministic_random_orthogonal_basis(
        1, 2, 16, 3, torch.device("cpu"), torch.float32
    )
    repeated = _deterministic_random_orthogonal_basis(
        1, 2, 16, 3, torch.device("cpu"), torch.float32
    )
    next_layer = _deterministic_random_orthogonal_basis(
        1, 2, 16, 4, torch.device("cpu"), torch.float32
    )
    identity = torch.eye(16).expand(1, 2, 16, 16)

    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(
        first.transpose(-1, -2) @ first,
        identity,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert not torch.equal(first, next_layer)


@pytest.mark.parametrize(
    ("score_mode", "expected"),
    (
        (
            "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk",
            ("qk_metric", "qmse"),
        ),
        (
            "pca_hierarchical_autoqmsetotal15z_qkmetric_"
            "qfused_packed_fulltopk",
            ("qk_metric", "qmse"),
        ),
        (
            "pca_hierarchical_fixed11111111_random_packed_fulltopk",
            ("random_orthogonal", "qmse"),
        ),
        (
            "pca_hierarchical_autokeytotal15z_packed_fulltopk",
            ("key_pca", "key_mse"),
        ),
        (
            "pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk",
            ("qk_metric", "key_mse"),
        ),
    ),
)
def test_packed_qmse_ablation_mode_contract(score_mode, expected) -> None:
    assert _packed_qmse_mode_contract(score_mode) == expected


@pytest.mark.parametrize(
    "score_mode",
    (
        "pca_hierarchical_autokeytotal15z_packed_fulltopk",
        "pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk",
    ),
)
def test_keymse_fulltopk_modes_enter_runtime_context(score_mode) -> None:
    with targeted.head_qabs_sampled_mass_mode(
        mass_threshold=0.75,
        score_mode=score_mode,
        projection_dim=48,
    ):
        pass


def test_qfused_mode_is_opt_in_and_does_not_change_frozen_mode() -> None:
    frozen = "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
    experimental = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_"
        "qfused_packed_fulltopk"
    )

    assert not _packed_qmse_uses_fused_query_prepare(frozen)
    assert _packed_qmse_uses_fused_query_prepare(experimental)


def test_qfused_query_prepare_waits_for_frozen_allocation() -> None:
    state = {
        "packed_qmse_fused_query_prepare": True,
        "packed_qmse_allocation_frozen": False,
    }
    assert not _packed_qmse_should_fuse_query_prepare(state)

    state["packed_qmse_allocation_frozen"] = True
    assert _packed_qmse_should_fuse_query_prepare(state)


def test_qfused_reference_matches_unfused_cpu_contract() -> None:
    generator = torch.Generator().manual_seed(20260728)
    query = torch.randn(2, 3, 4, 128, generator=generator)
    basis = torch.randn(2, 3, 128, 128, generator=generator)
    expected = variablebit.quantize_projected_query(
        torch.einsum("bhgd,bhdm->bhgm", query, basis)
    )
    actual = qksieve_query.project_quantize_reference(query, basis)

    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_qfused_reference_rejects_invalid_shapes_and_dtypes() -> None:
    query = torch.randn(1, 2, 4, 128)
    basis = torch.randn(1, 2, 128, 128)

    with pytest.raises(ValueError, match="grouped_query"):
        qksieve_query.project_quantize_reference(query[..., :64], basis)
    with pytest.raises(ValueError, match="basis"):
        qksieve_query.project_quantize_reference(query, basis[:, :1])
    with pytest.raises(ValueError, match="dtypes"):
        qksieve_query.project_quantize_reference(query, basis.half())
    with pytest.raises(ValueError, match="1-16"):
        qksieve_query.project_quantize_reference(
            torch.randn(1, 2, 17, 128),
            basis,
        )


def test_qfused_validation_matrix_axes_are_explicit() -> None:
    assert qfused_validate.parse_dtype("fp16") == torch.float16
    assert qfused_validate.parse_dtype("bfloat16") == torch.bfloat16
    with pytest.raises(ValueError, match="dtype"):
        qfused_validate.parse_dtype("float32")

    assert qfused_matrix.parse_csv("4,8,4") == ["4", "8"]
    with pytest.raises(ValueError, match="empty"):
        qfused_matrix.parse_csv(" , ")


def test_qfused_v2_reuses_basis_with_coalesced_output_dimension() -> None:
    source = qksieve_query.CUDA_SOURCE
    assert "int kv_row = blockIdx.x;" in source
    assert "float accumulators[kMaxGroups];" in source
    assert (
        "basis_row[input_dimension * 128 + output_dimension]"
        in source
    )
    assert (
        "static_cast<scalar_t>(accumulators[group])"
        in source
    )
    assert "int query_row = kv_row * group_count + group;" in source
    assert "<<<kv_row_count, 128, 0, stream>>>" in source


def test_qk_trace_records_only_registered_decode_steps(monkeypatch) -> None:
    class FakeAttention:
        layer_idx = 2

    def original(*args, **kwargs):
        del args, kwargs
        return torch.zeros(1), None

    monkeypatch.setattr(
        targeted,
        "_ORIGINAL_LLAMA_EAGER_ATTENTION_FORWARD",
        original,
    )
    query = torch.randn(1, 4, 1, 128)
    key = torch.randn(1, 1, 16, 128)
    value = torch.randn_like(key)
    records = []

    with targeted.capture_qk_trace(
        records,
        layers=(2,),
        max_records_per_layer=2,
        state_on_first_record_only=True,
        record_steps=(1, 3),
    ):
        for _ in range(5):
            targeted._patched_llama_eager_attention_forward(
                FakeAttention(),
                query,
                key,
                value,
                None,
                scaling=128**-0.5,
            )

    assert [record["step"] for record in records] == [1, 3]
    assert records[0]["key"] is not None
    assert records[1]["key"] is None
    assert targeted._ACTIVE_QK_TRACE is None


def test_qk_trace_rejects_invalid_registered_steps() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        with targeted.capture_qk_trace(
            [],
            layers=(0,),
            max_records_per_layer=1,
            record_steps=(-1,),
        ):
            pass
    with pytest.raises(ValueError, match="smaller"):
        with targeted.capture_qk_trace(
            [],
            layers=(0,),
            max_records_per_layer=1,
            record_steps=(0, 1),
        ):
            pass


def test_fixed_allocation_parser_and_validation() -> None:
    assert parse_fixed_bit_allocation("8,4,4,1,1,1,1,1") == (
        8,
        4,
        4,
        1,
        1,
        1,
        1,
        1,
    )
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="eight comma-separated",
    ):
        parse_fixed_bit_allocation("4,4,2,1")
    with pytest.raises(ValueError, match="selected from"):
        _expand_packed_qmse_fixed_allocation(
            (3, 0, 0, 0, 0, 0, 0, 0),
            1,
            1,
            torch.device("cpu"),
        )


def test_oas_query_metric_is_psd_and_preserves_total_trace() -> None:
    generator = torch.Generator().manual_seed(20260727)
    queries = torch.randn(
        2,
        3,
        11,
        128,
        generator=generator,
    )
    shrunk, alpha = _oas_shrink_query_band_second_moments(queries)
    raw_bands = queries.reshape(2, 3, 11, 8, 16)
    raw = torch.einsum(
        "bhqgd,bhqge->bhgde",
        raw_bands,
        raw_bands,
    ) / 11

    assert shrunk.shape == (2, 3, 8, 16, 16)
    assert alpha.shape == (2, 3)
    assert bool(torch.all((alpha >= 0.0) & (alpha <= 1.0)))
    assert float(torch.linalg.eigvalsh(shrunk).min()) >= -1.0e-5
    torch.testing.assert_close(
        shrunk.diagonal(dim1=-2, dim2=-1).sum(dim=(-1, -2)),
        raw.diagonal(dim1=-2, dim2=-1).sum(dim=(-1, -2)),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_oas_query_metric_shrinks_sample_poor_anisotropic_estimates_more() -> None:
    generator = torch.Generator().manual_seed(20260728)
    coordinate_scale = torch.linspace(0.5, 2.0, 128)
    small = (
        torch.randn(1, 1, 16, 128, generator=generator)
        * coordinate_scale
    )
    large = (
        torch.randn(1, 1, 4096, 128, generator=generator)
        * coordinate_scale
    )
    _, small_alpha = _oas_shrink_query_band_second_moments(small)
    _, large_alpha = _oas_shrink_query_band_second_moments(large)

    assert float(small_alpha.item()) > float(large_alpha.item())


def test_vectorized_oas_rate_cost_matches_reference_without_metric_scale() -> None:
    generator = torch.Generator().manual_seed(20260729)
    keys = torch.randn(1, 2, 64, 128, generator=generator)
    queries = torch.randn(1, 2, 12, 128, generator=generator)
    expected = _hierarchical_qmse_rate_allocation_reference(
        keys,
        queries,
        bit_budget_per_coordinate=15,
        allow_zero_bits=True,
        include_scale_metadata=True,
        query_covariance_shrinkage="oas",
    )
    actual = _hierarchical_qmse_rate_allocation(
        keys,
        queries,
        bit_budget_per_coordinate=15,
        allow_zero_bits=True,
        include_scale_metadata=True,
        query_covariance_shrinkage="oas",
    )
    assert torch.equal(actual, expected)


def test_oas_allocator_and_encoder_simulation_share_one_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(20260730)
    keys = torch.randn(1, 1, 48, 128, generator=generator)
    queries = torch.randn(1, 1, 9, 128, generator=generator)
    expected_metric, _ = _oas_shrink_query_band_second_moments(queries)
    captured_metrics = []
    original = targeted._hierarchical_metric_scale_quantize_bands

    def capture_metric(values, bits, query_metrics):
        captured_metrics.append(query_metrics.detach().clone())
        return original(values, bits, query_metrics)

    monkeypatch.setattr(
        targeted,
        "_hierarchical_metric_scale_quantize_bands",
        capture_metric,
    )
    allocation = targeted._hierarchical_qmse_rate_allocation(
        keys,
        queries,
        bit_budget_per_coordinate=15,
        allow_zero_bits=True,
        include_scale_metadata=True,
        query_covariance_shrinkage="oas",
        metric_scale_quantization=True,
    )

    assert allocation.shape == (1, 1, 8)
    assert len(captured_metrics) == 5
    for metric in captured_metrics:
        torch.testing.assert_close(metric, expected_metric)


@pytest.mark.parametrize(
    "score_mode",
    (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_direct",
        "pca_hierarchical_autokeytotal15z_packed_fulltopk",
        "pca_hierarchical_autokeytotal15z_qkmetric_packed_fulltopk",
    ),
)
def test_packed_budget_saturation_is_exact_and_skips_index(score_mode) -> None:
    generator = torch.Generator().manual_seed(20260727)
    query = torch.randn(1, 2, 1, 128, generator=generator)
    key = torch.randn(1, 1, 172, 128, generator=generator)
    value = torch.randn(1, 1, 172, 128, generator=generator)
    diagnostics: dict[str, object] = {}

    expected, expected_indices = exact_head_top_fraction_attention(
        query,
        key,
        value,
        None,
        scaling=128**-0.5,
        top_fraction=1.0,
    )
    actual, actual_indices = qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        None,
        scaling=128**-0.5,
        mass_threshold=1.0,
        budget_fractions=(1.0,),
        sample_fraction=0.01,
        qabs_dim_count=8,
        candidate_fraction=1.0,
        use_cuda_kernels=False,
        diagnostics=diagnostics,
        score_mode=score_mode,
        projection_dim=128,
        pca_state=None,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_indices, expected_indices)
    assert diagnostics["budget_saturated_full_history"] == 1.0
