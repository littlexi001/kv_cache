from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_value_mediated_rope_probe_8b as target  # noqa: E402


def test_value_mediated_formula_matches_exact_autograd_score_gradient() -> None:
    torch.manual_seed(20260801)
    heads, tokens, dimension = 3, 7, 5
    scores = torch.randn(heads, tokens, dtype=torch.float64, requires_grad=True)
    values = torch.randn(heads, tokens, dimension, dtype=torch.float64)
    downstream = torch.randn(heads, dimension, dtype=torch.float64)
    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum("hn,hnd->hd", attention, values)
    margin = (output * downstream).sum()

    output_gradient = torch.autograd.grad(margin, output, retain_graph=True)[0]
    exact_score_gradient = torch.autograd.grad(margin, scores)[0]
    reconstructed = target.value_mediated_derivative(
        attention, values, output, output_gradient
    )

    torch.testing.assert_close(
        reconstructed.double(), exact_score_gradient, atol=1e-6, rtol=1e-6
    )


def test_grouped_query_math_matches_explicit_kv_repetition() -> None:
    torch.manual_seed(9)
    batch, kv_heads, groups, query_tokens, keys, dimension = 1, 2, 3, 1, 7, 5
    query = torch.randn(batch, kv_heads * groups, query_tokens, dimension)
    key = torch.randn(batch, kv_heads, keys, dimension)
    value = torch.randn(batch, kv_heads, keys, dimension)
    repeated_key = key.repeat_interleave(groups, dim=1)
    repeated_value = value.repeat_interleave(groups, dim=1)

    scores = target.grouped_query_scores(query, key, groups)
    expected_scores = torch.matmul(query, repeated_key.transpose(2, 3))
    weights = torch.softmax(scores, dim=-1)
    output = target.grouped_attention_output(weights, value, groups)
    expected_output = torch.matmul(weights, repeated_value)

    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(output, expected_output)


def test_direct_ov_controls_match_explicit_isolated_head_projection() -> None:
    torch.manual_seed(17)
    heads, tokens, dimension, vocab = 2, 3, 4, 11
    hidden = heads * dimension
    projection = torch.nn.Linear(hidden, hidden, bias=False, dtype=torch.float64)
    unembedding = torch.nn.Linear(hidden, vocab, bias=False, dtype=torch.float64)
    values = torch.randn(heads, tokens, dimension, dtype=torch.float64)
    attention = torch.softmax(
        torch.randn(heads, tokens, dtype=torch.float64), dim=-1
    )
    output = torch.einsum("hn,hnd->hd", attention, values)
    controls = target.direct_ov_proxies(
        projection,
        unembedding,
        values,
        output,
        attention,
        gold_token_id=9,
        conflict_token_id=4,
    )

    expected_gold = torch.zeros_like(attention)
    expected_margin = torch.zeros_like(attention)
    expected_centered = torch.zeros_like(attention)
    u_gold = unembedding.weight[9]
    u_conflict = unembedding.weight[4]
    for head in range(heads):
        for token in range(tokens):
            isolated = torch.zeros(hidden, dtype=torch.float64)
            isolated[head * dimension : (head + 1) * dimension] = values[head, token]
            centered = torch.zeros(hidden, dtype=torch.float64)
            centered[head * dimension : (head + 1) * dimension] = (
                values[head, token] - output[head]
            )
            projected = projection(isolated)
            projected_centered = projection(centered)
            expected_gold[head, token] = attention[head, token] * torch.dot(
                u_gold, projected
            )
            expected_margin[head, token] = attention[head, token] * torch.dot(
                u_gold - u_conflict, projected
            )
            expected_centered[head, token] = attention[head, token] * torch.dot(
                u_gold - u_conflict, projected_centered
            )

    torch.testing.assert_close(
        controls["locos_direct_ov_gold"].double(), expected_gold
    )
    torch.testing.assert_close(
        controls["locos_direct_ov_margin"].double(), expected_margin
    )
    torch.testing.assert_close(
        controls["direct_ov_centered_margin_derivative"].double(),
        expected_centered,
    )


def test_target_and_random_selection_is_exact_and_deterministic() -> None:
    positions = torch.tensor([11, 17, 23, 29])
    gap = torch.tensor(
        [[0.0, 3.0, 2.0, 1.0], [8.0, 1.0, 9.0, 2.0]], dtype=torch.float32
    )
    first = target.select_target_and_random_positions(
        positions, gap, seed=4, layer_index=7, class_index=2
    )
    second = target.select_target_and_random_positions(
        positions, gap, seed=4, layer_index=7, class_index=2
    )

    assert first["target_positions"].tolist() == [17, 23]
    assert first["random_positions"].tolist() == second["random_positions"].tolist()
    assert all(
        target_position != random_position
        for target_position, random_position in zip(
            first["target_positions"].tolist(),
            first["random_positions"].tolist(),
        )
    )
    assert first["target_suppression_gap"].tolist() == [3.0, 9.0]


def test_selection_requires_an_alternative_random_token() -> None:
    with pytest.raises(ValueError, match="at least two"):
        target.select_target_and_random_positions(
            torch.tensor([3]),
            torch.tensor([[1.0], [2.0]]),
            seed=0,
            layer_index=0,
            class_index=0,
        )


def test_uniform_score_lift_changes_one_frozen_position_per_head() -> None:
    scores = torch.zeros(1, 3, 1, 8)
    positions = torch.tensor([0, 4, 7])
    modified, summary = target.apply_uniform_score_lift(
        scores, positions, score_lift=0.25
    )

    changed = (modified != scores)[0, :, 0, :]
    assert changed[0].nonzero().reshape(-1).tolist() == [0]
    assert changed[1].nonzero().reshape(-1).tolist() == [4]
    assert changed[2].nonzero().reshape(-1).tolist() == [7]
    assert modified[0, 0, 0, 0].item() == pytest.approx(0.25)
    assert modified[0, 1, 0, 4].item() == pytest.approx(0.25)
    assert modified[0, 2, 0, 7].item() == pytest.approx(0.25)
    assert summary["applied_count"] == 3
    assert summary["score_delta_sum"] == pytest.approx(0.75)


def test_uniform_score_lift_enforces_small_cap() -> None:
    scores = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match="score_lift"):
        target.apply_uniform_score_lift(scores, torch.tensor([0]), 0.251)
    with pytest.raises(ValueError, match="score_lift"):
        target.apply_uniform_score_lift(scores, torch.tensor([0]), 0.0)


def test_single_score_lift_changes_exactly_one_coordinate() -> None:
    scores = torch.zeros(1, 3, 1, 8)
    modified, summary = target.apply_single_score_lift(
        scores, head=1, position=6, score_lift=0.25
    )

    changed = (modified != scores).nonzero(as_tuple=False).tolist()
    assert changed == [[0, 1, 0, 6]]
    assert modified[0, 1, 0, 6].item() == pytest.approx(0.25)
    assert summary["applied_count"] == 1
    with pytest.raises(ValueError, match="score_lift"):
        target.apply_single_score_lift(
            scores, head=1, position=6, score_lift=0.251
        )


def _sample_row(
    category: str,
    layer: int,
    head: int,
    sample_index: int,
    ranking_value: float,
) -> dict:
    position = 100 * layer + 10 * head + sample_index
    return {
        "layer": layer,
        "head": head,
        "class": category,
        "sample_index": sample_index,
        "token_position": position,
        "relative_distance": 1000 - position,
        "is_decisive_token": int(sample_index == 0),
        "post_score": 1.0,
        "grid_envelope_score": 2.0,
        "suppression_gap": 2.0,
        "attention_probability": 0.01,
        "best_anchor_distance": 16,
        "dm_dscore": ranking_value / 2.0,
        "suppression_x_dm_dscore": ranking_value,
        "positive_suppression_x_dm_dscore": ranking_value,
        "locos_direct_ov_gold": 0.2,
        "locos_direct_ov_margin": 0.1,
        "direct_ov_centered_margin_derivative": 0.05,
        "suppression_x_direct_ov_centered_margin": 0.1,
    }


def test_singleton_candidates_are_global_frozen_and_matched() -> None:
    rows = []
    for class_index, category in enumerate(target.CLASS_ORDER):
        for layer in range(2):
            for sample_index in range(3):
                rows.append(
                    _sample_row(
                        category,
                        layer,
                        head=class_index,
                        sample_index=sample_index,
                        ranking_value=float(
                            10 * class_index + 3 * layer + sample_index + 1
                        ),
                    )
                )
    first = target.freeze_singleton_candidates(
        rows,
        top_n=2,
        ranking_metric="abs_positive_suppression_x_dm_dscore",
        seed=7,
    )
    second = target.freeze_singleton_candidates(
        rows,
        top_n=2,
        ranking_metric="abs_positive_suppression_x_dm_dscore",
        seed=7,
    )

    assert first == second
    assert len(first) == 2 * len(target.CLASS_ORDER)
    for candidate in first:
        selected = candidate["target"]
        control = candidate["random"]
        assert selected["class"] == control["class"] == candidate["class"]
        assert selected["layer"] == control["layer"]
        assert selected["head"] == control["head"]
        assert selected["token_position"] != control["token_position"]
        assert candidate["candidate_frozen_before_intervention"] == 1
    # The two largest values in every class are globally selected, not one per
    # layer/head cell.
    for category in target.CLASS_ORDER:
        selected_values = [
            candidate["ranking_value"]
            for candidate in first
            if candidate["class"] == category
        ]
        expected = sorted(
            (
                abs(row["positive_suppression_x_dm_dscore"])
                for row in rows
                if row["class"] == category
            ),
            reverse=True,
        )[:2]
        assert selected_values == expected

    # Intervention outcomes are absent from the ranking contract and cannot
    # retroactively change an already-frozen plan.
    for row in rows:
        row["actual_delta_gold_conflict_margin"] = -999.0
    assert first == target.freeze_singleton_candidates(
        rows,
        top_n=2,
        ranking_metric="abs_positive_suppression_x_dm_dscore",
        seed=7,
    )


def test_singleton_controller_only_touches_its_layer_head_and_token() -> None:
    controller = target.ValueMediatedController(
        mode="intervene",
        case={},
        anchor_distances=(1,),
        fixed_anchor_distance=1,
        score_lift=0.25,
        target_class=target.CLASS_ORDER[0],
        plan_kind="target",
        intervention_scope="singleton",
        singleton_layer=4,
        singleton_head=2,
        singleton_position=5,
    )
    scores = torch.zeros(1, 4, 1, 9)
    assert controller.intervene_layer(3, scores) is scores
    modified = controller.intervene_layer(4, scores)
    assert (modified != scores).nonzero(as_tuple=False).tolist() == [
        [0, 2, 0, 5]
    ]
    assert controller.applied_count == 1


def test_noop_controller_is_an_exact_zero_score_replay() -> None:
    controller = target.ValueMediatedController(
        mode="intervene",
        case={},
        anchor_distances=(1,),
        fixed_anchor_distance=1,
        score_lift=0.0,
        target_class=target.CLASS_ORDER[0],
        plan_kind="target",
        intervention_scope="singleton",
        singleton_layer=4,
        singleton_head=2,
        singleton_position=5,
    )
    scores = torch.randn(1, 4, 1, 9)

    replayed = controller.intervene_layer(4, scores)
    assert replayed is not scores
    torch.testing.assert_close(replayed, scores, atol=0.0, rtol=0.0)
    assert controller.applied_count == 0


def _noop_audit_fixture() -> tuple[list[dict], list[dict], dict]:
    noop_answer = {"gold_conflict_margin": 1.0, "gold_nll": 2.0}
    noop_row = {
        "intervention_class": target.CUSTOM_NOOP_BASELINE,
        "intervention_scope": "noop",
        "uniform_score_lift": 0.0,
        "applied_count": 0,
        "replayed_coordinate_count": 1,
        "epsilon_zero_noop_control": 1,
    }
    rows = [noop_row]
    for plan_kind, position in (("target", 5), ("random", 6)):
        rows.append(
            {
                "intervention_class": "gold_evidence",
                "intervention_scope": "singleton",
                "comparison_baseline": target.CUSTOM_NOOP_BASELINE,
                "causal_delta_reference": target.CUSTOM_NOOP_BASELINE,
                "pair_id": "gold_evidence_001",
                "plan_kind": plan_kind,
                "selected_baseline_layer": 3,
                "selected_baseline_head": 7,
                "selected_baseline_token_position": position,
                "applied_count": 1,
                "gold_conflict_margin": 1.2,
                "gold_nll": 1.9,
                "delta_gold_conflict_margin": 0.2,
                "delta_gold_nll": -0.1,
            }
        )
    candidates = [{"pair_id": "gold_evidence_001"}]
    return rows, candidates, noop_answer


def test_case_audit_requires_noop_reference_and_matched_random() -> None:
    rows, candidates, noop_answer = _noop_audit_fixture()
    audit = target.audit_noop_referenced_case(rows, candidates, noop_answer)

    assert audit["passed"] is True
    assert audit["epsilon_zero_noop_count"] == 1
    assert audit["singleton_replay_count"] == 2
    assert audit["matched_target_random_pair_count"] == 1


def test_case_audit_fails_if_actual_delta_uses_old_baseline() -> None:
    rows, candidates, noop_answer = _noop_audit_fixture()
    rows[1]["comparison_baseline"] = "instrumented_baseline"

    with pytest.raises(RuntimeError, match="does not reference"):
        target.audit_noop_referenced_case(rows, candidates, noop_answer)


def test_active_attention_uses_read_only_final_query_cache_helper() -> None:
    import inspect

    source = inspect.getsource(target.value_mediated_attention_forward)
    assert "safety.read_only_final_query_kv" in source
    assert "past_key_value.update" not in source


def test_closure_metrics_distinguish_sign_and_magnitude() -> None:
    exact = target.closure_metrics(0.4, 0.4)
    assert exact["first_order_sign_match"] == 1
    assert exact["first_order_absolute_closure_error"] == 0.0
    assert exact["first_order_symmetric_closure_error"] == 0.0

    wrong = target.closure_metrics(0.2, -0.1)
    assert wrong["predicted_margin_change_sign"] == 1
    assert wrong["actual_margin_change_sign"] == -1
    assert wrong["first_order_sign_match"] == 0
    assert wrong["first_order_symmetric_closure_error"] == pytest.approx(1.0)


def test_correlations_are_tie_aware_and_report_degenerate_as_nan() -> None:
    x = [1.0, 2.0, 2.0, 4.0]
    y = [10.0, 20.0, 20.0, 40.0]
    assert target.correlation(x, y) == pytest.approx(1.0)
    assert target.spearman_correlation(x, y) == pytest.approx(1.0)
    assert math.isnan(target.correlation([1.0, 1.0], [2.0, 3.0]))


def test_fp32_pair_margin_overrides_coarse_model_logit_margin() -> None:
    lm_head = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.copy_(
            torch.tensor(
                [[0.0, 0.0], [1.0078125, 0.0], [1.0, 0.0]],
                dtype=torch.bfloat16,
            )
        )
    model = SimpleNamespace(lm_head=lm_head)
    output = SimpleNamespace(
        # Deliberately tied coarse logits; the FP32 pair readout below is not.
        logits=torch.tensor([[[0.0, 1.0, 1.0]]], dtype=torch.bfloat16),
        hidden_states=(
            torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16),
        ),
    )

    class Tokenizer:
        @staticmethod
        def decode(ids, **kwargs):
            return str(ids[0])

    metrics = target.answer_metrics_with_fp32_pair(
        model,
        Tokenizer(),
        output,
        {"nine": 1, "four": 2},
        "four",
    )
    assert metrics["gold_conflict_margin_model_logits"] == 0.0
    assert metrics["gold_conflict_margin_fp32_pair"] == pytest.approx(
        0.0078125
    )
    assert metrics["gold_conflict_margin"] == pytest.approx(0.0078125)
    assert metrics["pair_margin_compute_dtype"] == "float32"
    assert metrics["final_hidden_source_dtype"] == "bfloat16"
    assert metrics["pair_margin_hidden_precision_limited"] == 1


def _intervention_row(
    length: int,
    seed: int,
    category: str,
    plan_kind: str,
    predicted: float,
    actual: float,
    scope: str = "singleton",
) -> dict:
    row = {
        "target_context_tokens": length,
        "seed": seed,
        "intervention_class": category,
        "plan_kind": plan_kind,
        "intervention_scope": scope,
        "gold_nll": 1.0 - actual,
        "next_token_correct": int(actual >= 0),
        "delta_gold_nll": -actual,
    }
    row.update(target.closure_metrics(predicted, actual))
    return row


def test_prediction_summary_reports_sign_correlation_and_ppl() -> None:
    rows = [
        _intervention_row(8192, 0, "gold_evidence", "target", 0.1, 0.2),
        _intervention_row(8192, 1, "gold_evidence", "target", 0.2, 0.4),
        _intervention_row(8192, 0, "gold_evidence", "random", -0.1, 0.1),
    ]
    summary = target.prediction_summary(rows)
    assert not any(
        row["target_context_tokens"] == "all" for row in summary
    )
    selected = next(
        row
        for row in summary
        if row["target_context_tokens"] == "8192"
        and row["plan_kind"] == "target"
        and row["intervention_class"] == "gold_evidence"
    )
    assert selected["n"] == 2
    assert selected["intervention_scope"] == "singleton"
    assert selected["pearson_predicted_vs_actual"] == pytest.approx(1.0)
    assert selected["spearman_predicted_vs_actual"] == pytest.approx(1.0)
    assert selected["sign_accuracy"] == 1.0
    assert selected["gold_ppl_exp_mean_nll"] == pytest.approx(math.exp(0.7))


def test_prediction_summary_never_mixes_singleton_and_joint_scopes() -> None:
    rows = [
        _intervention_row(
            8192, 0, "gold_evidence", "target", 0.1, 0.2, "singleton"
        ),
        _intervention_row(
            8192, 0, "gold_evidence", "target", 9.0, -8.0, "joint"
        ),
    ]
    summary = target.prediction_summary(rows)
    selected = [
        row
        for row in summary
        if row["target_context_tokens"] == "8192"
        and row["plan_kind"] == "target"
        and row["intervention_class"] == "gold_evidence"
    ]
    assert {row["intervention_scope"] for row in selected} == {
        "singleton",
        "joint",
    }
    assert all(row["n"] == 1 for row in selected)


def test_prediction_summary_adds_all_only_for_multiple_lengths() -> None:
    rows = [
        _intervention_row(
            8192, 0, "gold_evidence", "target", 0.1, 0.2
        ),
        _intervention_row(
            32768, 1, "gold_evidence", "target", 0.3, 0.4
        ),
    ]
    summary = target.prediction_summary(rows)
    combined = next(
        row
        for row in summary
        if row["target_context_tokens"] == "all"
        and row["intervention_scope"] == "singleton"
        and row["plan_kind"] == "target"
        and row["intervention_class"] == "gold_evidence"
    )
    assert combined["n"] == 2


def test_shard_merge_preserves_oracle_warning_and_outputs(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    raw = shard / "raw"
    raw.mkdir(parents=True)
    case_rows = [
        _intervention_row(
            8192, 0, category, plan_kind, predicted=0.1, actual=0.2
        )
        for category in target.CLASS_ORDER
        for plan_kind in target.PLAN_KINDS
    ]
    target.write_json(raw / "length_8192_seed_0_result.json", {"case_rows": case_rows})
    samples = []
    for category in target.CLASS_ORDER:
        row = {
                "target_context_tokens": 8192,
                "seed": 0,
                "class": category,
                "suppression_gap": 1.0,
                "attention_probability": 0.01,
                "dm_dscore": 0.5,
                "suppression_x_dm_dscore": 0.5,
                "positive_suppression_x_dm_dscore": 0.5,
            }
        for field_name in (
            "locos_direct_ov_gold",
            "locos_direct_ov_margin",
            "direct_ov_centered_margin_derivative",
            "suppression_x_direct_ov_centered_margin",
        ):
            row[field_name] = 0.25
        samples.append(row)
    target.append_jsonl(raw / "length_8192_seed_0_value_samples.jsonl", samples)

    merged = tmp_path / "merged"
    merged.mkdir()
    target.write_aggregate_outputs(merged, [shard])
    summary = json.loads((merged / "summary.json").read_text(encoding="utf-8"))

    assert summary["oracle_diagnostic_only"] is True
    assert summary["oracle_gradient_target"] == target.ORACLE_GRADIENT_TARGET
    assert summary["case_row_count"] == 8
    assert summary["value_sample_count"] == 4
    assert (merged / "value_sample_summary.csv").exists()
    assert (merged / "first_order_prediction_summary.csv").exists()
    assert (merged / "singleton_prediction_summary.csv").exists()


def _write_merge_test_config(
    shard: Path,
    *,
    seed_start: int,
    cuda_visible_devices: str,
    dtype: str = "bfloat16",
) -> dict[str, object]:
    config: dict[str, object] = {
        "model_name_or_path": "/models/actual-qwen3-8b",
        "output_dir": str(shard),
        "lengths": "8192",
        "resolved_lengths": [8192],
        "seed_start": seed_start,
        "num_seeds": 4,
        "score_lift": 0.125,
        "singleton_top_n": 3,
        "dtype": dtype,
        "load_in_4bit": True,
        "cuda_visible_devices": cuda_visible_devices,
        "merge_shards": "",
        "oracle_gradient_target": target.ORACLE_GRADIENT_TARGET,
    }
    shard.mkdir()
    target.write_json(shard / "config.json", config)
    return config


def test_merge_config_uses_shard_configs_not_merge_cli_defaults(
    tmp_path: Path,
) -> None:
    shard6 = tmp_path / "shard_gpu6"
    shard7 = tmp_path / "shard_gpu7"
    config6 = _write_merge_test_config(
        shard6, seed_start=0, cuda_visible_devices="6"
    )
    config7 = _write_merge_test_config(
        shard7, seed_start=4, cuda_visible_devices="7"
    )
    merged = tmp_path / "merged"
    merge_argument = f"{shard6},{shard7}"

    provenance = target.build_merge_config(
        merged, [shard6, shard7], merge_argument
    )

    assert provenance["merge_schema_version"] == 2
    assert provenance["shared_config"]["model_name_or_path"] == (
        "/models/actual-qwen3-8b"
    )
    assert provenance["shared_config"]["score_lift"] == 0.125
    assert provenance["shared_config"]["singleton_top_n"] == 3
    assert provenance["shared_config"]["load_in_4bit"] is True
    assert "seed_start" not in provenance["shared_config"]
    assert provenance["merge_invocation"]["merge_shards"] == merge_argument
    assert provenance["shards"][0]["config"] == config6
    assert provenance["shards"][1]["config"] == config7
    assert [entry["config"]["seed_start"] for entry in provenance["shards"]] == [
        0,
        4,
    ]
    assert [
        entry["config"]["cuda_visible_devices"]
        for entry in provenance["shards"]
    ] == ["6", "7"]


def test_merge_main_writes_real_shard_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard6 = tmp_path / "shard_gpu6"
    shard7 = tmp_path / "shard_gpu7"
    _write_merge_test_config(shard6, seed_start=0, cuda_visible_devices="6")
    _write_merge_test_config(shard7, seed_start=4, cuda_visible_devices="7")
    merged = tmp_path / "merged"
    merge_argument = f"{shard6},{shard7}"
    monkeypatch.setattr(
        target,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=str(merged), merge_shards=merge_argument
        ),
    )
    monkeypatch.setattr(target, "write_aggregate_outputs", lambda *_: None)

    target.main()

    payload = json.loads(
        (merged / "merge_config.json").read_text(encoding="utf-8")
    )
    assert payload["shared_config"]["model_name_or_path"] == (
        "/models/actual-qwen3-8b"
    )
    assert payload["shared_config"]["load_in_4bit"] is True
    assert [entry["config"]["seed_start"] for entry in payload["shards"]] == [
        0,
        4,
    ]
    assert "model_name_or_path" not in payload["merge_invocation"]


def test_merge_config_rejects_inconsistent_experiment_fields(
    tmp_path: Path,
) -> None:
    shard6 = tmp_path / "shard_gpu6"
    shard7 = tmp_path / "shard_gpu7"
    _write_merge_test_config(shard6, seed_start=0, cuda_visible_devices="6")
    _write_merge_test_config(
        shard7,
        seed_start=4,
        cuda_visible_devices="7",
        dtype="float16",
    )

    with pytest.raises(ValueError, match="inconsistent shard config field 'dtype'"):
        target.build_merge_config(
            tmp_path / "merged", [shard6, shard7], f"{shard6},{shard7}"
        )


def test_validate_args_requires_eager_and_two_samples() -> None:
    args = argparse.Namespace(
        lengths="8192,32768",
        anchor_distances="1,2,4,8,16,32,64,128",
        fixed_anchor_distance=128,
        class_sample_count=8,
        num_seeds=4,
        prefill_chunk_size=64,
        score_lift=0.25,
        singleton_top_n=16,
        singleton_ranking_metric=(
            "abs_positive_suppression_x_dm_dscore"
        ),
        run_joint_interventions=False,
        attn_implementation="eager",
    )
    assert target.validate_args(args) == (
        [8192, 32768],
        (1, 2, 4, 8, 16, 32, 64, 128),
    )
    args.attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        target.validate_args(args)
    args.attn_implementation = "eager"
    args.class_sample_count = 1
    with pytest.raises(ValueError, match="two class samples"):
        target.validate_args(args)
    args.class_sample_count = 8
    args.singleton_top_n = 0
    with pytest.raises(ValueError, match="enable singleton"):
        target.validate_args(args)


def test_reuses_safety_case_and_digit_target_without_truth_labels() -> None:
    assert target.CLASS_ORDER is target.safety.CLASS_ORDER
    assert target.safety.ANSWER_DIGIT_BY_WORD["nine"] == "9"
    assert target.ORACLE_GRADIENT_TARGET == "gold_digit_vs_conflict_digit_margin"
    assert "verified" in target.safety.TRUTH_STATUS_MARKERS


def test_gpu_launcher_is_hard_limited_to_physical_six_and_seven() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_value_mediated_probe_gpu67_20260801.sh"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=6" in launcher
    assert "CUDA_VISIBLE_DEVICES=7" in launcher
    for forbidden in range(6):
        assert f"CUDA_VISIBLE_DEVICES={forbidden}" not in launcher
    assert "--lengths 8192,32768" in launcher
    assert "--score-lift 0.25" in launcher
    assert "--singleton-top-n 16" in launcher
    assert (
        "--singleton-ranking-metric "
        "abs_positive_suppression_x_dm_dscore"
    ) in launcher
    assert "--run-joint-interventions" not in launcher
    assert "--attn-implementation eager" in launcher


def test_v2_bf16_smoke_launcher_is_small_unquantized_and_gpu67_only() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_value_mediated_probe_v2_bf16_smoke_gpu67_20260801.sh"
    ).read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=6" in launcher
    assert "CUDA_VISIBLE_DEVICES=7" in launcher
    for forbidden in range(6):
        assert f"CUDA_VISIBLE_DEVICES={forbidden}" not in launcher
    assert "--lengths 8192" in launcher
    assert "--class-sample-count 2" in launcher
    assert "--singleton-top-n 1" in launcher
    assert "--score-lift 0.25" in launcher
    assert "--dtype bfloat16" in launcher
    assert "--load-in-4bit" not in launcher
