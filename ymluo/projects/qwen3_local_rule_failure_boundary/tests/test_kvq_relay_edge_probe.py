from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT = SRC.parent
sys.path.insert(0, str(SRC))

import run_kvq_relay_edge_probe_8b as target  # noqa: E402


def test_runner_is_independent_and_launcher_is_physical_gpu67_only() -> None:
    source = (SRC / "run_kvq_relay_edge_probe_8b.py").read_text(encoding="utf-8")
    for forbidden in (
        "run_phase_coherent_rope_probe_8b",
        "run_value_mediated_rope_probe_8b",
        "run_suppression_certificate_safety_probe_8b",
        "run_queryspan_prerope_retrieval_probe_8b",
    ):
        assert forbidden not in source
    launcher = (
        PROJECT / "scripts" / "run_kvq_relay_edge_probe_gpu67_20260801.sh"
    ).read_text(encoding="utf-8")
    assert launcher.count("CUDA_VISIBLE_DEVICES=6") == 1
    assert launcher.count("CUDA_VISIBLE_DEVICES=7") == 1
    for device in range(6):
        assert f"CUDA_VISIBLE_DEVICES={device}" not in launcher
    protocol = (
        PROJECT
        / "analysis"
        / "rope_method_search_20260801"
        / "kvq_relay_edge_probe_protocol.md"
    ).read_text(encoding="utf-8")
    assert "arxiv.org/abs/2502.13913" in protocol
    assert "不主张首次发现 sequential-query mechanism" in protocol


def test_default_relay_layers_are_four_fixed_depth_fractions() -> None:
    assert target.resolve_relay_layers(36) == (9, 18, 27, 34)
    assert target.resolve_relay_layers(32) == (8, 16, 24, 30)
    assert target.resolve_relay_layers(12, (1, 4, 7, 10)) == (1, 4, 7, 10)


def test_label_free_segmentation_uses_only_boundary_and_length() -> None:
    tokens = [10, 11, 99, 12, 13, 14, 15, 99, 16]
    blocks = target.segment_label_free_blocks(tokens, 0, len(tokens), 3, {99})
    assert [(block.start, block.end) for block in blocks] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 9),
    ]
    assert [block.index for block in blocks] == list(range(4))


def test_gqa_pre_scores_match_explicit_key_repetition() -> None:
    torch.manual_seed(5)
    query = torch.randn(4, 6)
    key = torch.randn(2, 9, 6)
    actual = target.gqa_query_key_scores(query, key)
    repeated = key.repeat_interleave(2, dim=0)
    expected = torch.einsum("hd,htd->ht", query, repeated) / (6**0.5)
    torch.testing.assert_close(actual, expected)


def test_block_logmeanexp_is_length_corrected() -> None:
    scores = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    blocks = [target.CandidateBlock(0, 0, 1), target.CandidateBlock(1, 1, 4)]
    actual = target.block_logmeanexp_scores(scores, blocks, temperature=0.7)
    torch.testing.assert_close(actual, torch.tensor([[2.0, 2.0]]), atol=1e-6, rtol=0)


def test_candidate_selection_is_capped_and_accepts_no_labels() -> None:
    blocks = [target.CandidateBlock(index, index, index + 1) for index in range(6)]
    layer_scores = {
        1: torch.tensor([[0.0, 3.0, 2.0, 1.0, -1.0, -2.0]]),
        2: torch.tensor([[0.0, 2.0, 3.0, 1.0, -1.0, -2.0]]),
    }
    selected, aggregate = target.select_candidate_blocks(layer_scores, blocks, 3)
    assert len(selected) == 3
    assert {block.index for block in selected} == {1, 2, 3}
    assert aggregate.shape == (6,)
    parameters = set(inspect.signature(target.select_candidate_blocks).parameters)
    assert not {"gold", "labels", "events", "answer"} & parameters
    score_parameters = set(inspect.signature(target.score_case_label_free).parameters)
    assert not {"case", "gold", "labels", "events", "answer"} & score_parameters


def test_gqa_value_aggregation_matches_explicit_repeat() -> None:
    token_scores = torch.tensor(
        [
            [3.0, 1.0, -1.0],
            [1.0, 3.0, -1.0],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
        ]
    )
    value = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    blocks = [target.CandidateBlock(0, 0, 2), target.CandidateBlock(1, 2, 3)]
    actual = target.aggregate_block_head_values(token_scores, value, blocks, 1.0)
    repeated = value.repeat_interleave(2, dim=0)
    expected = []
    for block in blocks:
        weights = torch.softmax(token_scores[:, block.start : block.end], dim=-1)
        expected.append(
            torch.einsum("ht,htd->hd", weights, repeated[:, block.start : block.end])
        )
    torch.testing.assert_close(actual, torch.stack(expected))


def test_shuffle_and_random_controls_preserve_required_norms() -> None:
    values = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5) + 1
    shuffled, permutation = target.shuffled_block_values(values, seed=17)
    assert sorted(permutation.tolist()) == [0, 1, 2, 3]
    assert not bool((permutation == torch.arange(4)).any())
    torch.testing.assert_close(shuffled, values.index_select(0, permutation))
    random_values = target.norm_matched_random_values(values, seed=19)
    torch.testing.assert_close(
        random_values.norm(dim=-1), values.norm(dim=-1), atol=1e-5, rtol=1e-5
    )
    assert not torch.allclose(random_values, values)


def test_w_o_projection_cancels_bias_exactly() -> None:
    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = torch.nn.Linear(6, 4, bias=True)

    torch.manual_seed(23)
    attention = Attention()
    values = torch.randn(5, 2, 3)
    actual = target.project_head_values(attention, values)
    expected = torch.nn.functional.linear(values.reshape(5, 6), attention.o_proj.weight)
    torch.testing.assert_close(actual, expected)


def test_random_control_matches_norm_after_anisotropic_w_o() -> None:
    candidate = torch.tensor([[10.0, 1.0], [1.0, 0.1]])
    reference = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    matched = target.match_projected_write_norms(candidate, reference)
    torch.testing.assert_close(
        matched.norm(dim=-1), reference.norm(dim=-1), atol=1e-6, rtol=1e-6
    )


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.q_norm = torch.nn.Identity()
        self.o_proj = torch.nn.Linear(4, 4, bias=False)


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = _FakeAttention()


class _FakeModelBody(torch.nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(layer_count)])


class _FakeModel(torch.nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.model = _FakeModelBody(layer_count)


def test_next_query_finite_difference_matches_linear_map_and_audits() -> None:
    torch.manual_seed(29)
    layer = _FakeLayer()
    baseline = torch.randn(1, 4)
    delta = torch.randn(3, 4)
    actual, recomputed = target.finite_difference_next_queries(
        layer, baseline, delta, epsilon=0.05, batch_size=2
    )
    expected = torch.nn.functional.linear(delta, layer.self_attn.q_proj.weight).reshape(3, 2, 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    captured = target.next_layer_query_projection(layer, baseline)
    torch.testing.assert_close(recomputed, captured)
    audit = target.finite_difference_audit(
        layer,
        baseline,
        delta,
        captured,
        epsilon=0.05,
        batch_size=2,
    )
    assert audit["audit_block_count"] == 3
    assert audit["baseline_q_reconstruction_max_abs"] == 0.0
    assert audit["fd_halving_relative_error_max"] < 1e-4
    assert audit["fd_halving_cosine_min"] > 0.9999


def test_directed_relay_prefers_matching_destination_direction() -> None:
    delta_query = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    blocks = [target.CandidateBlock(0, 0, 1), target.CandidateBlock(1, 1, 2)]
    scores = target.directed_relay_scores(delta_query, key, blocks, temperature=1.0)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_label_free_case_scorer_runs_end_to_end_on_tiny_frozen_model() -> None:
    torch.manual_seed(31)
    model = _FakeModel(5)

    class Cache:
        def __init__(self) -> None:
            self.value_cache = [torch.randn(1, 1, 8, 2) for _ in range(5)]

    state = target.CaptureState((0, 1, 2, 3), (0, 1, 2, 3, 4), "cpu")
    for layer_index, layer in enumerate(model.model.layers):
        hidden = torch.randn(1, 4)
        state.layer_inputs[layer_index] = hidden
        state.query_pre[layer_index] = target.next_layer_query_projection(layer, hidden)
        state.pre_keys[layer_index] = torch.randn(1, 1, 8, 2)
    blocks = [target.CandidateBlock(index, 2 * index, 2 * index + 2) for index in range(4)]
    scored = target.score_case_label_free(
        model=model,
        cache=Cache(),
        state=state,
        case_seed=0,
        target_context_tokens=8,
        all_blocks=blocks,
        maximum_blocks=4,
        block_temperature=1.0,
        value_temperature=1.0,
        fd_epsilon=0.05,
        fd_batch_size=2,
    )
    assert len(scored["candidate_blocks"]) == 4
    assert set(scored["aggregate"]) == set(target.SCORE_NAMES)
    assert all(matrix.shape == (4, 4) for matrix in scored["aggregate"].values())
    assert len(scored["finite_difference_audits"]) == 4
    assert all(audit["audit_block_count"] == 4 for audit in scored["finite_difference_audits"])
    assert set(scored["relay_tensors"]) == {0, 1, 2, 3}


def test_key_key_control_is_cosine_and_layer_aggregate_is_robust() -> None:
    keys = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    similarity = target.key_key_similarity(keys, keys)
    torch.testing.assert_close(similarity, torch.eye(2))
    matrices = {
        2: {name: similarity.clone() for name in target.SCORE_NAMES},
        5: {name: (2.0 * similarity).clone() for name in target.SCORE_NAMES},
    }
    aggregate = target.aggregate_layer_matrices(matrices)
    assert set(aggregate) == set(target.SCORE_NAMES)
    assert aggregate["kvq_relay"][0, 0] > aggregate["kvq_relay"][0, 1]


def _event(kind: str, label: str, step: int, start: int, end: int) -> dict[str, object]:
    return {
        "kind": kind,
        "label": label,
        "step": step,
        "start_token": start,
        "end_token": end,
    }


def test_labels_enter_only_after_scores_and_find_structured_record_control() -> None:
    blocks = [
        target.CandidateBlock(0, 0, 5),
        target.CandidateBlock(1, 5, 10),
        target.CandidateBlock(2, 10, 15),
        target.CandidateBlock(3, 15, 20),
        target.CandidateBlock(4, 20, 25),
    ]
    events = [
        _event("relevant", "T0", 0, 1, 3),
        _event("relevant", "T1", 1, 6, 8),
        _event("competitor", "C0_0", 0, 11, 13),
        _event("competitor", "C0_1", 1, 16, 18),
    ]
    positives, negatives, metadata = target.choose_evaluation_pairs(
        blocks, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]), events, 3
    )
    assert positives == {(0, 1)}
    assert negatives[(2, 3)] == "structured_record"
    assert metadata["positive_pair_count"] == 1
    assert metadata["matched_negative_count"] == 3
    assert metadata["gold_both_candidate_covered"] == 1
    assert metadata["gold_edge_resolved"] == 1


def test_binary_auroc_handles_wins_and_ties() -> None:
    assert target.binary_auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert target.binary_auroc([1.0], [1.0]) == 0.5
    assert target.binary_auroc([], [1.0]) is None


def test_summary_excludes_failed_finite_difference_cases() -> None:
    def row(seed: int, label: int, value: float, passed: int) -> dict[str, object]:
        item: dict[str, object] = {
            "target_context_tokens": 8,
            "condition": "mixed",
            "seed": seed,
            "layer": -1,
            "label": label,
            "finite_difference_audit_pass": passed,
        }
        item.update({name: value for name in target.SCORE_NAMES})
        return item

    rows = [
        row(0, 1, 2.0, 1),
        row(0, 0, 1.0, 1),
        row(1, 1, -10.0, 0),
        row(1, 0, 10.0, 0),
    ]
    summary = target.summarize_edge_rows(rows)
    local = next(item for item in summary if item["condition"] == "mixed")
    assert local["kvq_relay_auroc"] == 1.0
    assert local["positive_edge_count"] == 1


def test_summary_macro_averages_each_case_instead_of_pooling_scores() -> None:
    def row(seed: int, label: int, value: float) -> dict[str, object]:
        item: dict[str, object] = {
            "target_context_tokens": 8,
            "condition": "mixed",
            "seed": seed,
            "layer": -1,
            "label": label,
            "finite_difference_audit_pass": 1,
        }
        item.update({name: value for name in target.SCORE_NAMES})
        return item

    rows = [
        row(0, 1, 100.0),
        row(0, 0, 99.0),
        row(1, 1, 1.0),
        row(1, 0, 0.0),
        # Unresolved case: its extreme negative must not pollute edge AUROC.
        row(2, 0, 1000.0),
    ]
    local = next(
        item for item in target.summarize_edge_rows(rows) if item["condition"] == "mixed"
    )
    assert local["kvq_relay_auroc"] == 1.0
    assert local["valid_case_auroc_count"] == 2
    assert local["invalid_case_auroc_count"] == 1
    assert local["kvq_relay_auroc_seed_bootstrap_ci_low"] == 1.0
    assert local["kvq_relay_auroc_seed_bootstrap_ci_high"] == 1.0


def test_case_summary_reports_candidate_coverage_and_audit_rate() -> None:
    common = {
        "target_context_tokens": 8,
        "condition": "mixed",
        "candidate_block_count": 4,
        "gold_edge_resolved": 1,
        "prefill_seconds": 2.0,
        "edge_score_seconds": 1.0,
    }
    rows = [
        {
            **common,
            "gold_both_candidate_covered": 1,
            "finite_difference_audit_pass": 1,
        },
        {
            **common,
            "gold_both_candidate_covered": 0,
            "finite_difference_audit_pass": 0,
        },
    ]
    summary = target.summarize_case_rows(rows)
    local = next(item for item in summary if item["condition"] == "mixed")
    assert local["gold_both_candidate_coverage"] == 0.5
    assert local["gold_both_candidate_missing_rate"] == 0.5
    assert local["finite_difference_audit_pass_rate"] == 0.5


def test_prefix_fingerprint_is_immutable_sensitive() -> None:
    class Cache:
        def __init__(self) -> None:
            self.key_cache = [torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)]
            self.value_cache = [torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2)]

    cache = Cache()
    before = target.cache_prefix_fingerprint(cache, [0], 4)
    same = target.cache_prefix_fingerprint(cache, [0], 4)
    assert before == same
    cache.value_cache[0][0, 0, 1, 0] += 1
    after = target.cache_prefix_fingerprint(cache, [0], 4)
    assert before != after


def test_prefix_fingerprint_detects_content_swap_with_identical_moments() -> None:
    class Cache:
        def __init__(self) -> None:
            self.key_cache = [torch.arange(256, dtype=torch.float32).reshape(1, 2, 64, 2)]
            self.value_cache = [torch.arange(256, dtype=torch.float32).reshape(1, 2, 64, 2)]

    cache = Cache()
    before = target.cache_prefix_fingerprint(cache, [0], 64)
    flat = cache.value_cache[0].reshape(-1)
    left = flat[17].clone()
    flat[17] = flat[219]
    flat[219] = left
    after = target.cache_prefix_fingerprint(cache, [0], 64)
    assert before["layers"]["0"]["value"]["sum"] == after["layers"]["0"]["value"]["sum"]
    assert before["layers"]["0"]["value"]["square_sum"] == after["layers"]["0"]["value"]["square_sum"]
    assert before != after


def test_case_transaction_ignores_partial_case_and_validates_hash(tmp_path: Path) -> None:
    row = {
        "target_context_tokens": 8,
        "condition": "mixed",
        "seed": 3,
    }
    edge = {
        **row,
        "layer": -1,
        "source_candidate_index": 0,
        "destination_candidate_index": 1,
        "label": 1,
    }
    target.commit_case_transaction(
        tmp_path,
        "hash-a",
        row,
        [edge],
        {"pass": True},
        {"tensor": torch.tensor([1.0])},
    )
    partial = tmp_path / "cases" / "length_9_condition_mixed_seed_4"
    partial.mkdir(parents=True)
    target.write_json(
        partial / "case_row.json",
        {"target_context_tokens": 9, "condition": "mixed", "seed": 4},
    )
    cases, edges = target.collect_committed_cases(tmp_path, "hash-a")
    assert [target.case_key(item) for item in cases] == [(8, "mixed", 3)]
    assert len(edges) == 1
    with pytest.raises(RuntimeError, match="incompatible committed case config"):
        target.collect_committed_cases(tmp_path, "hash-b")


def test_resume_rejects_changed_method_config(tmp_path: Path) -> None:
    def config(tag: str) -> dict[str, object]:
        method = {"protocol_version": target.PROTOCOL_VERSION, "tag": tag}
        method_hash = target.stable_json_hash(method)
        full = {
            "method_config_hash": method_hash,
            "seed_start": 0,
            "num_seeds": 1,
        }
        return {
            "method_config": method,
            **full,
            "full_config_hash": target.stable_json_hash(full),
        }

    target.initialize_or_validate_run_config(tmp_path, config("a"))
    target.initialize_or_validate_run_config(tmp_path, config("a"))
    with pytest.raises(RuntimeError, match="different or unversioned run config"):
        target.initialize_or_validate_run_config(tmp_path, config("b"))


def test_json_writer_rejects_nonstandard_nan_and_serializes_none(tmp_path: Path) -> None:
    path = tmp_path / "valid.json"
    target.write_json(path, {"auroc": None})
    assert json.loads(path.read_text(encoding="utf-8")) == {"auroc": None}
    with pytest.raises(ValueError):
        target.write_json(tmp_path / "invalid.json", {"auroc": float("nan")})


def test_merge_requires_complete_unique_compatible_shards(tmp_path: Path) -> None:
    method_config = {"protocol_version": target.PROTOCOL_VERSION, "probe": "same"}
    method_hash = target.stable_json_hash(method_config)

    def build_shard(name: str, seed: int) -> Path:
        shard = tmp_path / name
        full_hash = target.stable_json_hash(
            {
                "method_config_hash": method_hash,
                "seed_start": seed,
                "num_seeds": 1,
            }
        )
        config = {
            "method_config": method_config,
            "method_config_hash": method_hash,
            "full_config_hash": full_hash,
            "resolved_lengths": [8],
            "resolved_conditions": ["mixed"],
            "seed_start": seed,
            "num_seeds": 1,
        }
        target.write_json(shard / "config.json", config)
        row = {
            "target_context_tokens": 8,
            "condition": "mixed",
            "seed": seed,
            "gold_both_candidate_covered": 1,
            "gold_edge_resolved": 1,
            "finite_difference_audit_pass": 1,
            "candidate_block_count": 4,
            "prefill_seconds": 1.0,
            "edge_score_seconds": 1.0,
        }
        edges = []
        for label, value, destination in ((1, 2.0, 1), (0, 1.0, 2)):
            edge = {
                "target_context_tokens": 8,
                "condition": "mixed",
                "seed": seed,
                "layer": -1,
                "source_candidate_index": 0,
                "destination_candidate_index": destination,
                "label": label,
                "finite_difference_audit_pass": 1,
            }
            edge.update({score: value for score in target.SCORE_NAMES})
            edges.append(edge)
        target.commit_case_transaction(
            shard, full_hash, row, edges, {"pass": True}, {"seed": seed}
        )
        target.write_aggregate_outputs(shard, [row], edges)
        target.write_done_marker(shard / "done.txt")
        return shard

    shard0 = build_shard("shard0", 0)
    shard1 = build_shard("shard1", 1)
    merged = tmp_path / "merged"
    target.merge_shards(merged, [str(shard0), str(shard1)])
    manifest = target.read_json(merged / "manifest.json")
    assert manifest["case_count"] == 2
    assert manifest["unique_case_key_count"] == 2
    assert (merged / "done.txt").exists()

    duplicate = build_shard("duplicate", 1)
    with pytest.raises(RuntimeError, match="duplicate case keys"):
        target.merge_shards(tmp_path / "bad_merge", [str(shard1), str(duplicate)])


def test_query_append_then_reset_restores_exact_prefix_fingerprint() -> None:
    class Cache:
        def __init__(self) -> None:
            self.key_cache = [torch.randn(1, 2, 4, 3)]
            self.value_cache = [torch.randn(1, 2, 4, 3)]
            self._seen_tokens = 4

    cache = Cache()
    before = target.cache_prefix_fingerprint(cache, [0], 4)
    cache.key_cache[0] = torch.cat((cache.key_cache[0], torch.randn(1, 2, 1, 3)), dim=2)
    cache.value_cache[0] = torch.cat((cache.value_cache[0], torch.randn(1, 2, 1, 3)), dim=2)
    cache._seen_tokens = 5
    target.reset_dynamic_cache(cache, 4)
    after = target.cache_prefix_fingerprint(cache, [0], 4)
    assert after == before
    assert cache._seen_tokens == 4


def test_capture_hooks_store_prefix_key_query_and_layer_input(monkeypatch) -> None:
    state = target.CaptureState((0, 1, 2, 3), (0, 1, 2, 3), "cpu")
    key_output = torch.randn(1, 4, 2, 3)
    state.mode = "prefix"
    target._make_key_capture_hook(state, 0)(None, (), key_output)
    assert state.pre_key_chunks[0][0].shape == (1, 2, 4, 3)
    state.mode = "query"
    query_output = torch.randn(1, 1, 4, 3)
    target._make_query_capture_hook(state, 0)(None, (), query_output)
    hidden = torch.randn(1, 1, 12)
    target._make_layer_input_hook(state, 1)(None, (hidden,))
    torch.testing.assert_close(state.query_pre[0], query_output[:, -1])
    torch.testing.assert_close(state.layer_inputs[1], hidden[:, -1])
