from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


SRC = Path(__file__).resolve().parents[1] / "src"
REPO = Path(__file__).resolve().parents[4]
PAPER = REPO / "ymluo" / "papers" / "countcap_iclr2027"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_head_top2_targeted_ppl_20260714 as targeted  # noqa: E402
import run_sample_calibrated_longbench_20260717 as longbench  # noqa: E402
import analyze_qksieve_public_selectors_longbench_20260728 as analysis  # noqa: E402
import run_qksieve_coldskip_longcontext_quality_20260730 as long_quality  # noqa: E402


def test_binarypc_exact_rerank_factor_parser() -> None:
    assert long_quality.binarypc_overfetch_factor(
        "binarypc_exactrerank8x_k1280"
    ) == 8.0
    assert long_quality.binarypc_overfetch_factor(
        "binarypc_exactrerank4x_k1280"
    ) == 4.0


def test_quest_incremental_page_index_matches_fresh_rebuild() -> None:
    generator = torch.Generator().manual_seed(20260728)
    keys = torch.randn(1, 2, 35, 8, generator=generator)
    incremental_state: dict[str, object] = {}

    targeted._quest_update_page_index(
        keys[:, :, :17],
        incremental_state,
        page_size=16,
    )
    incremental = targeted._quest_update_page_index(
        keys,
        incremental_state,
        page_size=16,
    )
    fresh = targeted._quest_update_page_index(
        keys,
        {},
        page_size=16,
    )

    torch.testing.assert_close(incremental[0], fresh[0])
    torch.testing.assert_close(incremental[1], fresh[1])
    assert incremental_state["quest_indexed_count"] == 35


def test_quest_selects_pages_by_sign_aware_min_max_bound() -> None:
    keys = torch.zeros(1, 1, 32, 4)
    keys[:, :, 16:, 0] = 8.0
    query = torch.zeros(1, 2, 4)
    query[..., 0] = 1.0

    indices, counts = targeted._quest_page_candidates(
        query,
        keys,
        target_count=8,
        state={},
        page_size=16,
    )

    assert counts.tolist() == [[16, 16]]
    assert indices[0, 0].tolist() == list(range(16, 32))
    assert indices[0, 1].tolist() == list(range(16, 32))

    negative_keys = keys.clone()
    negative_keys[:, :, :16, 0] = -9.0
    negative_query = -query
    negative_indices, _ = targeted._quest_page_candidates(
        negative_query,
        negative_keys,
        target_count=8,
        state={},
        page_size=16,
    )
    assert negative_indices[0, 0].tolist() == list(range(0, 16))


def test_unique_incremental_page_statistics_match_fresh_rebuild() -> None:
    generator = torch.Generator().manual_seed(20260801)
    keys = torch.randn(1, 2, 21, 8, generator=generator)
    state: dict[str, object] = {}

    targeted._unique_update_page_index(keys[:, :, :10], state, page_size=8)
    incremental = targeted._unique_update_page_index(keys, state, page_size=8)
    fresh = targeted._unique_update_page_index(keys, {}, page_size=8)

    torch.testing.assert_close(incremental[0], fresh[0])
    torch.testing.assert_close(incremental[1], fresh[1])
    assert state["unique_indexed_count"] == 21
    assert state["unique_page_counts"].tolist() == [8.0, 8.0, 5.0]


def test_unique_candidates_match_published_gqa_mean_std_formula() -> None:
    generator = torch.Generator().manual_seed(41)
    keys = torch.randn(1, 1, 24, 4, generator=generator)
    query = torch.randn(1, 2, 4, generator=generator)

    indices, counts = targeted._unique_page_candidates(
        query,
        keys,
        target_count=8,
        state={},
        page_size=8,
        offset_scale=0.5,
    )

    pages = keys.reshape(1, 1, 3, 8, 4)
    means = pages.mean(dim=3)
    stds = pages.std(dim=3, correction=0).norm(dim=-1)
    grouped = query.reshape(1, 1, 2, 4)
    scores = (
        torch.einsum("bhgd,bhpd->bhgp", grouped, means)
        + 0.5
        * grouped.norm(dim=-1).unsqueeze(-1)
        * stds.unsqueeze(2)
    ).amax(dim=2)
    page = int(scores.argmax(dim=-1).item())
    expected = list(range(page * 8, page * 8 + 8))

    assert counts.tolist() == [[8, 8]]
    assert indices[0, 0].tolist() == expected
    assert indices[0, 1].tolist() == expected


def test_sparq_selector_supports_gqa_and_uses_large_query_dimensions() -> None:
    keys = torch.zeros(1, 2, 6, 4)
    keys[0, 0, 3, 0] = 7.0
    keys[0, 0, 4, 1] = 100.0
    keys[0, 1, 2, 2] = 9.0
    query = torch.zeros(1, 4, 4)
    query[0, 0:2, 0] = 5.0
    query[0, 2:4, 2] = 6.0

    indices, counts = targeted._sparq_dimension_candidates(
        query,
        keys,
        target_count=1,
        dimension_count=1,
    )

    assert indices.squeeze(-1).tolist() == [[3, 3, 2, 2]]
    assert counts.tolist() == [[1, 1, 1, 1]]


def test_rabitq_reference_selector_is_deterministic_and_incremental() -> None:
    generator = torch.Generator().manual_seed(20260801)
    query = torch.randn(1, 4, 8, generator=generator)
    keys = torch.randn(1, 2, 7, 8, generator=generator)
    query_sum = torch.randn(4, 8, generator=generator)
    state = {
        "layer_index": 3,
        "rabitq_prefill_query_sum": query_sum,
        "rabitq_prefill_query_count": 5,
    }

    first, first_counts = targeted._rabitq_reference_candidates(
        query,
        keys[:, :, :5],
        target_count=2,
        state=state,
    )
    second, second_counts = targeted._rabitq_reference_candidates(
        query,
        keys,
        target_count=2,
        state=state,
    )

    assert first.shape == (1, 4, 2)
    assert second.shape == (1, 4, 2)
    assert first_counts.tolist() == [[2, 2, 2, 2]]
    assert second_counts.tolist() == [[2, 2, 2, 2]]
    assert state["rabitq_indexed_count"] == 7
    assert state["rabitq_codes"].shape == (1, 4, 7, 8)
    assert torch.equal(
        state["rabitq_rotation"],
        targeted._rabitq_rotation_cpu(3, 8),
    )


def test_rabitq_reference_selector_matches_centered_int4_formula() -> None:
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(1, 4, 8, generator=generator)
    keys = torch.randn(1, 2, 11, 8, generator=generator)
    query_sum = torch.randn(4, 8, generator=generator)
    query_count = 7
    state = {
        "layer_index": 2,
        "rabitq_prefill_query_sum": query_sum,
        "rabitq_prefill_query_count": query_count,
    }

    indices, _ = targeted._rabitq_reference_candidates(
        query,
        keys,
        target_count=3,
        state=state,
    )

    query_centroid = (query_sum / query_count).unsqueeze(0)
    centered_query = query - query_centroid
    query_norm = centered_query.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    rotated_query = torch.matmul(
        centered_query / query_norm,
        state["rabitq_rotation"].transpose(0, 1),
    )
    query_min = rotated_query.amin(dim=-1, keepdim=True)
    query_step = (
        (rotated_query.amax(dim=-1, keepdim=True) - query_min) / 15.0
    ).clamp_min(1.0e-8)
    quantized_query = (
        torch.round((rotated_query - query_min) / query_step)
        .clamp(0.0, 15.0)
        .mul(query_step)
        .add(query_min)
    )
    signed_codes = (
        state["rabitq_codes"].float().mul(2.0).sub(1.0) / (8.0**0.5)
    )
    centered_scores = torch.einsum(
        "bhd,bhnd->bhn",
        quantized_query,
        signed_codes,
    ) * query_norm
    centered_scores = (
        centered_scores
        * state["rabitq_key_norms"]
        / state["rabitq_alignment"]
    )
    key_centroid = state["rabitq_key_centroid"]
    scores = (
        centered_scores
        + (query * key_centroid).sum(dim=-1, keepdim=True)
        + state["rabitq_cq_dot_key"]
        - (query_centroid * key_centroid).sum(dim=-1, keepdim=True)
    )
    expected = torch.topk(scores, k=3, dim=-1, sorted=False).indices

    assert torch.equal(indices.sort(dim=-1).values, expected.sort(dim=-1).values)
    assert not torch.equal(quantized_query, rotated_query)


def test_binarypc_reference_matches_official_hash_formula_and_gqa() -> None:
    generator = torch.Generator().manual_seed(20260801)
    query = torch.randn(1, 4, 8, generator=generator)
    keys = torch.randn(1, 2, 13, 8, generator=generator)
    projection = torch.randn(2, 64, 8, generator=generator) * 0.05
    state: dict[str, object] = {"binarypc_projection": projection}

    indices, counts = targeted._binarypc_reference_candidates(
        query,
        keys,
        target_count=4,
        state=state,
        error_ratio=0.25,
    )

    grouped_query = query.reshape(1, 2, 2, 8)
    probe = torch.einsum("bhgd,hrd->bhgr", grouped_query, projection)
    probe_max = probe.abs().reshape(1, 2, -1).amax(
        dim=-1,
        keepdim=True,
    ).unsqueeze(-1).clamp_min(1.0e-6)
    quantized_probe = torch.round(probe * (127.0 / probe_max))
    unpacked = targeted._binarypc_unpack_codes(
        state["binarypc_hashcodes"],
        quantized_probe.dtype,
    )
    scores = torch.einsum(
        "bhgr,bhnr->bhgn",
        quantized_probe,
        unpacked,
    ).amax(dim=2)
    rescue = state["binarypc_error_order"][..., :1]
    scores.scatter_(
        -1,
        rescue,
        (scores.amax(dim=-1, keepdim=True) + 1.0).expand_as(rescue),
    )
    expected = torch.topk(scores, k=4, dim=-1, sorted=False).indices

    assert counts.tolist() == [[4, 4, 4, 4]]
    assert state["binarypc_indexed_count"] == 13
    assert state["binarypc_hashcodes"].shape == (1, 2, 13)
    assert torch.equal(
        indices[:, 0].sort().values,
        expected[:, 0].sort().values,
    )
    assert torch.equal(indices[:, 0], indices[:, 1])
    assert torch.equal(indices[:, 2], indices[:, 3])


def test_binarypc_reference_incrementally_hashes_new_keys() -> None:
    generator = torch.Generator().manual_seed(9)
    query = torch.randn(1, 4, 8, generator=generator)
    keys = torch.randn(1, 2, 9, 8, generator=generator)
    state: dict[str, object] = {
        "binarypc_projection": torch.randn(
            2,
            64,
            8,
            generator=generator,
        )
        * 0.03
    }

    targeted._binarypc_reference_candidates(query, keys[:, :, :7], 3, state)
    first_codes = state["binarypc_hashcodes"].clone()
    targeted._binarypc_reference_candidates(query, keys, 3, state)

    assert state["binarypc_indexed_count"] == 9
    assert state["binarypc_hashcodes"].shape == (1, 2, 9)
    assert torch.equal(state["binarypc_hashcodes"][:, :, :7], first_codes)


def test_binarypc_exact_rerank_selects_exact_best_from_coarse_pool() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(1, 4, 8, generator=generator)
    keys = torch.randn(1, 2, 20, 8, generator=generator)
    state: dict[str, object] = {
        "binarypc_projection": torch.randn(
            2, 64, 8, generator=generator
        ) * 0.04
    }

    indices, counts = targeted._binarypc_exact_rerank_candidates(
        query,
        keys,
        target_count=3,
        state=state,
        cuda_kernels=None,
        overfetch_factor=4.0,
        error_ratio=0.1,
    )
    coarse_count = int(state["binarypc_exact_rerank_coarse_count"])
    coarse_indices, _ = targeted._binarypc_reference_candidates(
        query,
        keys,
        target_count=coarse_count,
        state=state,
        error_ratio=0.1 * 3 / coarse_count,
    )
    grouped_keys = keys.unsqueeze(2).expand(1, 2, 2, 20, 8)
    grouped_indices = coarse_indices.reshape(1, 2, 2, coarse_count)
    selected_keys = torch.gather(
        grouped_keys,
        3,
        grouped_indices.unsqueeze(-1).expand(1, 2, 2, coarse_count, 8),
    )
    scores = (
        selected_keys * query.reshape(1, 2, 2, 8).unsqueeze(3)
    ).sum(-1).reshape(1, 4, coarse_count)
    expected_local = torch.topk(scores, k=3, dim=-1).indices
    expected = torch.gather(coarse_indices, -1, expected_local)

    assert counts.tolist() == [[3, 3, 3, 3]]
    assert state["binarypc_error_rescue_count"] == 1
    assert torch.equal(
        indices.sort(dim=-1).values,
        expected.sort(dim=-1).values,
    )


def test_sparq_formula_candidates_match_temperature_mass_and_budget() -> None:
    keys = torch.tensor(
        [
            [
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                    [1.5, -0.5, 0.0, 0.0],
                    [0.1, 0.2, 0.0, 0.0],
                ]
            ]
        ]
    )
    query = torch.tensor(
        [[[4.0, 2.0, 0.1, 0.0], [-3.0, 1.0, 0.0, 0.2]]]
    )

    indices, counts, mass, local_window = (
        targeted._sparq_formula_candidates(
            query,
            keys,
            target_history_count=3,
            dimension_count=2,
            local_fraction=0.25,
        )
    )

    assert indices.shape == (1, 2, 3)
    assert counts.tolist() == [[3, 3]]
    assert bool((indices < 7).all())
    assert local_window == 1

    manual_mass = []
    for head in range(2):
        selected_dims = torch.topk(
            query[0, head].abs(),
            k=2,
        ).indices
        selected_query = query[0, head, selected_dims]
        selected_keys = keys[0, 0, :, selected_dims]
        temperature = torch.sqrt(
            4.0
            * (
                selected_query.abs().sum()
                / query[0, head].abs().sum()
            )
        )
        probabilities = F.softmax(
            (selected_keys * selected_query).sum(dim=-1) / temperature,
            dim=-1,
        )
        selected = indices[0, head].tolist() + [7]
        manual_mass.append(probabilities[selected].sum())
    torch.testing.assert_close(mass[0], torch.stack(manual_mass))


def test_sparq_running_value_mean_incremental_matches_full_rebuild() -> None:
    generator = torch.Generator().manual_seed(20260728)
    value = torch.randn(1, 2, 7, 4, generator=generator)
    state: dict[str, object] = {}

    targeted._sparq_running_value_mean(value[:, :, :3], state)
    incremental = targeted._sparq_running_value_mean(value, state)
    fresh = targeted._sparq_running_value_mean(value, {})

    torch.testing.assert_close(incremental, value.float().mean(dim=2))
    torch.testing.assert_close(incremental, fresh)
    assert state["sparq_value_indexed_count"] == 7


def test_sparq_mean_value_correction_has_correct_endpoints() -> None:
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(1, 4, 1, 4, generator=generator)
    key = torch.randn(1, 2, 3, 4, generator=generator)
    value = torch.randn(1, 2, 3, 4, generator=generator)
    indices = torch.tensor([[[0], [1], [0], [1]]])
    counts = torch.ones(1, 4, dtype=torch.long)
    scaling = 0.5
    sparse = targeted._public_selector_exact_attention(
        query,
        key,
        value,
        indices,
        counts,
        scaling,
        cuda_kernels=None,
    )

    all_sparse = targeted._sparq_mean_corrected_attention(
        query,
        key,
        value,
        indices,
        counts,
        torch.ones(1, 4),
        scaling,
        {},
        cuda_kernels=None,
    )
    all_mean = targeted._sparq_mean_corrected_attention(
        query,
        key,
        value,
        indices,
        counts,
        torch.zeros(1, 4),
        scaling,
        {},
        cuda_kernels=None,
    )
    repeated_mean = value.float().mean(dim=2).repeat_interleave(2, dim=1)

    torch.testing.assert_close(all_sparse, sparse)
    torch.testing.assert_close(
        all_mean,
        repeated_mean.to(all_mean.dtype).unsqueeze(1),
    )


def test_sparq_formula_score_mode_executes_through_public_entrypoint() -> None:
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(1, 2, 1, 4, generator=generator)
    key = torch.randn(1, 1, 9, 4, generator=generator)
    value = torch.randn(1, 1, 9, 4, generator=generator)
    diagnostics: dict[str, object] = {}
    state: dict[str, object] = {}

    output, indices = targeted.qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=0.5,
        mass_threshold=1.0,
        budget_fractions=(0.25,),
        sample_fraction=0.25,
        qabs_dim_count=4,
        candidate_fraction=0.25,
        use_cuda_kernels=False,
        diagnostics=diagnostics,
        score_mode="sparq_r32_meanvalue_fulltopk",
        projection_dim=4,
        pca_state=state,
    )

    assert output.shape == (1, 1, 2, 4)
    assert indices.shape == (1, 2, 2)
    assert torch.isfinite(output).all()
    assert diagnostics["public_selector_mean_value_correction"] == 1.0
    assert diagnostics["public_selector_dimension_count"] == 32.0
    assert diagnostics["public_selector_local_window"] == 1.0
    assert state["sparq_value_indexed_count"] == 9


def test_public_selector_cpu_consumer_matches_manual_gqa_attention() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 4, 1, 4, generator=generator)
    key = torch.randn(1, 2, 3, 4, generator=generator)
    value = torch.randn(1, 2, 3, 4, generator=generator)
    indices = torch.tensor([[[0, 1], [1, 0], [0, 1], [1, 0]]])
    counts = torch.tensor([[1, 2, 1, 2]])
    scaling = 0.5

    actual = targeted._public_selector_exact_attention(
        query,
        key,
        value,
        indices,
        counts,
        scaling,
        cuda_kernels=None,
    )

    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    expected_heads = []
    for head in range(4):
        count = int(counts[0, head])
        selected = indices[0, head, :count].tolist() + [2]
        selected_key = repeated_key[0, head, selected]
        selected_value = repeated_value[0, head, selected]
        scores = query[0, head, 0] @ selected_key.T * scaling
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        expected_heads.append(weights @ selected_value)
    expected = torch.stack(expected_heads).reshape(1, 1, 4, 4)

    torch.testing.assert_close(actual, expected)


def test_public_selectors_use_frozen_qksieve_budget_contract() -> None:
    args = SimpleNamespace(countcap_direct_fraction_override=0.0)
    history = 32_000
    expected_tokens, expected_fraction = longbench.countcap_direct_budget(
        history
    )

    for method, score_mode in (
        (
            longbench.QUEST_P16_FULLTOPK_METHOD,
            longbench.QUEST_P16_FULLTOPK_SCORE_MODE,
        ),
        (
            longbench.UNIQUE_P8_FULLTOPK_METHOD,
            longbench.UNIQUE_P8_FULLTOPK_SCORE_MODE,
        ),
        (
            longbench.RABITQCACHE_RTN1_FULLTOPK_METHOD,
            longbench.RABITQCACHE_RTN1_FULLTOPK_SCORE_MODE,
        ),
        (
            longbench.BINARYPC_OFFLINE64_FULLTOPK_METHOD,
            longbench.BINARYPC_OFFLINE64_FULLTOPK_SCORE_MODE,
        ),
        (
            longbench.SPARQ_R32_SELECTOR_FULLTOPK_METHOD,
            longbench.SPARQ_R32_SELECTOR_FULLTOPK_SCORE_MODE,
        ),
        (
            longbench.SPARQ_R32_FORMULA_FULLTOPK_METHOD,
            longbench.SPARQ_R32_FORMULA_FULLTOPK_SCORE_MODE,
        ),
    ):
        assert longbench.parse_methods(method) == [method]
        assert longbench.uses_dense_prompt_suffix(method)
        config = longbench.sparse_method_config(
            method,
            history,
            (0.01, 0.02),
            args,
        )
        assert config["score_mode"] == score_mode
        assert config["attention_tokens"] == expected_tokens
        assert config["budget_fractions"] == (expected_fraction,)
        assert config["candidate_fraction"] == expected_fraction

    assert (
        longbench.configured_index_bits_per_token(
            longbench.QKSIEVE_FULLTOPK_SCORE_MODE
        )
        == 240.0
    )
    assert (
        longbench.configured_index_bits_per_token(
            longbench.QUEST_P16_FULLTOPK_SCORE_MODE
        )
        == 256.0
    )
    assert (
        longbench.configured_index_bits_per_token(
            longbench.UNIQUE_P8_FULLTOPK_SCORE_MODE
        )
        == 258.0
    )
    assert (
        longbench.configured_index_bits_per_token(
            longbench.RABITQCACHE_RTN1_FULLTOPK_SCORE_MODE
        )
        == 224.0
    )
    assert (
        longbench.configured_index_bits_per_token(
            longbench.BINARYPC_OFFLINE64_FULLTOPK_SCORE_MODE
        )
        == 64.0
    )
    assert (
        longbench.configured_index_bits_per_token(
            longbench.SPARQ_R32_SELECTOR_FULLTOPK_SCORE_MODE
        )
        == 0.0
    )
    assert (
        longbench.configured_index_bits_per_token(
            longbench.SPARQ_R32_FORMULA_FULLTOPK_SCORE_MODE
        )
        == 0.0
    )


def test_public_selector_report_requires_strict_same_budget_pairs() -> None:
    rows = []
    score_modes = {
        analysis.FULL: "full_kv",
        analysis.QKSIEVE: longbench.QKSIEVE_FULLTOPK_SCORE_MODE,
        analysis.QUEST: longbench.QUEST_P16_FULLTOPK_SCORE_MODE,
        analysis.RABITQ: longbench.RABITQCACHE_RTN1_FULLTOPK_SCORE_MODE,
        analysis.SPARQ_SELECTOR: (
            longbench.SPARQ_R32_SELECTOR_FULLTOPK_SCORE_MODE
        ),
        analysis.SPARQ_FORMULA: (
            longbench.SPARQ_R32_FORMULA_FULLTOPK_SCORE_MODE
        ),
    }
    index_bits = {
        analysis.FULL: 0,
        analysis.QKSIEVE: 240,
        analysis.QUEST: 256,
        analysis.RABITQ: 224,
        analysis.SPARQ_SELECTOR: 0,
        analysis.SPARQ_FORMULA: 0,
    }
    for task_index in range(16):
        for method in analysis.EXPECTED_METHODS:
            rows.append(
                {
                    "task": f"task{task_index}",
                    "sample_id": "0",
                    "method": method,
                    "executed_path": method,
                    "configured_score_mode": score_modes[method],
                    "configured_index_bits_per_token": str(
                        index_bits[method]
                    ),
                    "configured_attention_tokens": (
                        "1000" if method != analysis.FULL else "16000"
                    ),
                    "prefix_tokens": "16000",
                    "score": "1.0",
                }
            )

    report = analysis.analyze(rows, expected_pairs=16)

    assert report["strict_pairs"] == 16
    assert report["methods"][analysis.QKSIEVE]["quality_retention"] == 1.0
    assert (
        report["methods"][analysis.QUEST][
            "configured_mean_loaded_token_ratio"
        ]
        == 1008 / 16000
    )
    assert not report["latency_claim"]["valid"]

    broken = [dict(row) for row in rows]
    next(row for row in broken if row["method"] == analysis.SPARQ_FORMULA)[
        "configured_attention_tokens"
    ] = "999"
    try:
        analysis.analyze(broken, expected_pairs=16)
    except ValueError as error:
        assert "token budget" in str(error)
    else:
        raise AssertionError("mismatched active-token budget was accepted")


def test_paper_keeps_selector_controls_separate_from_official_systems() -> None:
    experiments = (
        PAPER / "sections" / "05_experiments.tex"
    ).read_text(encoding="utf-8")
    appendix = (
        PAPER / "sections" / "appendix.tex"
    ).read_text(encoding="utf-8")
    system_diagnostics = (
        PAPER / "sections" / "appendix_system_diagnostics.tex"
    ).read_text(encoding="utf-8")
    experiment_surface = experiments + system_diagnostics
    normalized_surface = " ".join(experiment_surface.split())

    assert "Quest-P16, page-rounded" in experiment_surface
    assert "SparQ-R32 selector only" in experiment_surface
    assert "SparQ-R32 complete formula" in experiment_surface
    assert "formula/reference controls, not official optimized-system timings" in (
        normalized_surface
    )
    assert "we do not report a fabricated native RaBitQ speed" in (
        normalized_surface
    )
    assert "provides RetroInfer, not a runnable artifact" in normalized_surface
    assert "matched-budget formula reference includes SparQ's temperature" in appendix
    assert "RetrievalAttention numbers remain explicitly paper reported" in appendix
    assert "heterogeneous system is described as index-byte matched" in appendix
