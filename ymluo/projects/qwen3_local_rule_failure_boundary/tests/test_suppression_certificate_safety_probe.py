from __future__ import annotations

import math
import csv
import sys
from pathlib import Path

import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import run_suppression_certificate_safety_probe_8b as target  # noqa: E402


def _rotate(
    values: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    rope_scale: float = 1.0,
) -> torch.Tensor:
    cos, sin = target._phase_values(
        positions, inv_freq, int(values.shape[-1]), values.dtype
    )
    while cos.dim() < values.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return rope_scale * (values * cos + target.rotate_half(values) * sin)


@pytest.mark.parametrize("groups", (1, 4))
def test_grouped_gqa_matches_repeated_kv_values_and_gradients(groups: int) -> None:
    torch.manual_seed(101 + groups)
    batch, kv_heads, query_tokens, key_tokens, head_dim = 2, 2, 3, 7, 6
    query_heads = kv_heads * groups
    scale = 1.0 / math.sqrt(head_dim)
    initial_query = torch.randn(
        batch, query_heads, query_tokens, head_dim, dtype=torch.float64
    )
    initial_key = torch.randn(
        batch, kv_heads, key_tokens, head_dim, dtype=torch.float64
    )
    initial_value = torch.randn(
        batch, kv_heads, key_tokens, head_dim, dtype=torch.float64
    )
    mask = torch.randn(batch, 1, query_tokens, key_tokens, dtype=torch.float64)
    output_probe = torch.randn(
        batch, query_heads, query_tokens, head_dim, dtype=torch.float64
    )
    score_probe = torch.randn(
        batch, query_heads, query_tokens, key_tokens, dtype=torch.float64
    )

    reference_tensors = [
        tensor.clone().requires_grad_()
        for tensor in (initial_query, initial_key, initial_value)
    ]
    query_ref, key_ref, value_ref = reference_tensors
    repeated_key = key_ref.repeat_interleave(groups, dim=1)
    repeated_value = value_ref.repeat_interleave(groups, dim=1)
    scores_ref = torch.matmul(query_ref, repeated_key.transpose(2, 3)) * scale + mask
    weights_ref = torch.softmax(scores_ref, dim=-1)
    output_ref = torch.matmul(weights_ref, repeated_value)
    loss_ref = (scores_ref * score_probe).sum() + (output_ref * output_probe).sum()
    loss_ref.backward()

    grouped_tensors = [
        tensor.clone().requires_grad_()
        for tensor in (initial_query, initial_key, initial_value)
    ]
    query_gqa, key_gqa, value_gqa = grouped_tensors
    scores_gqa = target.gqa_query_key_scores(query_gqa, key_gqa, scale) + mask
    weights_gqa = torch.softmax(scores_gqa, dim=-1)
    output_gqa = target.gqa_attention_output(weights_gqa, value_gqa)
    loss_gqa = (scores_gqa * score_probe).sum() + (output_gqa * output_probe).sum()
    loss_gqa.backward()

    torch.testing.assert_close(scores_gqa, scores_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(weights_gqa, weights_ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(output_gqa, output_ref, atol=1e-12, rtol=1e-12)
    for grouped, reference in zip(grouped_tensors, reference_tensors):
        torch.testing.assert_close(
            grouped.grad, reference.grad, atol=1e-11, rtol=1e-11
        )


@pytest.mark.parametrize("groups", (1, 4))
def test_gqa_sampled_gathers_match_full_repeat_reference(groups: int) -> None:
    torch.manual_seed(211 + groups)
    batch, kv_heads, keys, head_dim = 2, 2, 9, 5
    values = torch.randn(batch, kv_heads, keys, head_dim)
    query_heads = kv_heads * groups
    repeated = values.repeat_interleave(groups, dim=1)
    shared_positions = torch.tensor([0, 4, 8])
    per_head_positions = torch.stack(
        [torch.tensor([(head + 1) % keys, (2 * head + 3) % keys]) for head in range(query_heads)]
    )

    expected_shared = target.gather_shared_positions(repeated, shared_positions)
    actual_shared = target.gather_shared_gqa_positions(
        values, shared_positions, groups
    )
    expected_per_head = target.gather_per_head_positions(
        repeated, per_head_positions
    )
    actual_per_head = target.gather_per_query_head_gqa_positions(
        values, per_head_positions, groups
    )

    torch.testing.assert_close(actual_shared, expected_shared)
    torch.testing.assert_close(actual_per_head, expected_per_head)


def test_read_only_final_query_kv_never_mutates_or_grows_prefix_cache() -> None:
    class FakeCache:
        def __init__(self) -> None:
            self.key_cache = [torch.randn(1, 2, 5, 4) for _ in range(3)]
            self.value_cache = [torch.randn(1, 2, 5, 4) for _ in range(3)]

        def update(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("read-only final-query path must not call update")

    cache = FakeCache()
    before_keys = [tensor.clone() for tensor in cache.key_cache]
    before_values = [tensor.clone() for tensor in cache.value_cache]
    key_ptrs = [tensor.data_ptr() for tensor in cache.key_cache]
    value_ptrs = [tensor.data_ptr() for tensor in cache.value_cache]

    for layer in range(3):
        current_key = torch.randn(1, 2, 1, 4)
        current_value = torch.randn(1, 2, 1, 4)
        combined_key, combined_value = target.read_only_final_query_kv(
            cache, layer, current_key, current_value
        )
        assert combined_key.shape[-2] == 6
        assert combined_value.shape[-2] == 6
        torch.testing.assert_close(combined_key[..., :-1, :], before_keys[layer])
        torch.testing.assert_close(combined_value[..., :-1, :], before_values[layer])
        torch.testing.assert_close(combined_key[..., -1:, :], current_key)
        torch.testing.assert_close(combined_value[..., -1:, :], current_value)

    assert [tensor.data_ptr() for tensor in cache.key_cache] == key_ptrs
    assert [tensor.data_ptr() for tensor in cache.value_cache] == value_ptrs
    assert all(tensor.shape[-2] == 5 for tensor in cache.key_cache)
    assert all(tensor.shape[-2] == 5 for tensor in cache.value_cache)
    for actual, expected in zip(cache.key_cache, before_keys):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(cache.value_cache, before_values):
        torch.testing.assert_close(actual, expected)


def test_inverse_and_relative_reconstruction_match_standard_rope() -> None:
    torch.manual_seed(20260801)
    heads, keys, head_dim = 3, 7, 8
    query_position = 41
    key_positions = torch.tensor([0, 2, 9, 17, 28, 39, 41])
    inv_freq = torch.tensor([1.0, 0.27, 0.051, 0.006], dtype=torch.float64)
    rope_scale = 1.13
    score_scale = 1.0 / math.sqrt(head_dim)
    query_pre = torch.randn(1, heads, 1, head_dim, dtype=torch.float64)
    key_pre = torch.randn(1, heads, keys, head_dim, dtype=torch.float64)
    query_post = _rotate(
        query_pre, torch.tensor([query_position]), inv_freq, rope_scale
    )
    key_post = _rotate(key_pre, key_positions, inv_freq, rope_scale)

    recovered = target.invert_selected_rope(
        key_post, key_positions, inv_freq, rope_scale
    )
    reconstructed = target.relative_score_from_pre(
        query_pre.expand(-1, -1, keys, -1),
        recovered,
        query_position - key_positions,
        inv_freq,
        rope_scale,
        score_scale,
    )
    expected = (
        torch.matmul(query_post, key_post.transpose(2, 3)) * score_scale
    ).squeeze(2)

    # Qwen stores inv_freq and forms phases in float32 even when the surrounding
    # test vectors are float64, so one float32 phase ULP is expected here.
    torch.testing.assert_close(recovered, key_pre, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(reconstructed, expected, atol=2e-6, rtol=2e-6)


def test_move_post_keys_matches_direct_virtual_position() -> None:
    torch.manual_seed(2)
    heads, selected, head_dim = 2, 3, 8
    inv_freq = torch.tensor([1.0, 0.4, 0.1, 0.01], dtype=torch.float64)
    pre = torch.randn(1, heads, selected, head_dim, dtype=torch.float64)
    old = torch.tensor([[5, 9, 17], [5, 9, 17]])
    new = torch.tensor([[30, 31, 32], [28, 29, 30]])
    post = _rotate(pre, old, inv_freq)

    moved = target.move_post_keys(post, old, new, inv_freq)
    expected = _rotate(pre, new, inv_freq)

    torch.testing.assert_close(moved, expected, atol=5e-6, rtol=5e-6)


def test_phase_upper_is_not_below_any_single_relative_phase() -> None:
    torch.manual_seed(3)
    query = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    key = torch.randn(1, 2, 5, 8, dtype=torch.float64)
    upper = target.phase_upper_scores(
        query.expand(-1, -1, 5, -1), key, rope_scale=1.0, score_scale=1.0
    )
    inv_freq = torch.tensor([1.0, 0.3, 0.07, 0.01], dtype=torch.float64)
    for distance in (0, 1, 7, 31, 127):
        actual = target.relative_score_from_pre(
            query.expand(-1, -1, 5, -1),
            key,
            torch.full((5,), distance),
            inv_freq,
            rope_scale=1.0,
            score_scale=1.0,
        )
        assert bool(torch.all(upper + 1e-10 >= actual))


def test_matched_plan_selects_exactly_k_per_head() -> None:
    positions = torch.tensor([11, 17, 23, 29])
    certificate = torch.tensor(
        [[0.1, 1.0, 0.3, 0.5], [2.0, -1.0, 4.0, 3.0]]
    )
    anchors = torch.tensor(
        [[1, 2, 4, 8], [16, 32, 64, 128]], dtype=torch.long
    )
    plan = target.select_matched_plan(
        positions, certificate, anchors, matched_tokens=2
    )

    assert tuple(plan["positions"].shape) == (2, 2)
    assert plan["positions"].tolist() == [[17, 29], [23, 29]]
    assert plan["anchor_distances"].tolist() == [[2, 8], [64, 128]]


def test_apply_frozen_plan_changes_only_planned_scores() -> None:
    heads, keys, head_dim = 2, 6, 4
    query_position = keys - 1
    inv_freq = torch.tensor([1.0, 0.1])
    query_pre = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]])
    key_pre = torch.zeros(1, heads, keys, head_dim)
    key_pre[..., 0] = 1.0
    all_positions = torch.arange(keys)
    query_post = _rotate(
        query_pre, torch.tensor([query_position]), inv_freq
    )
    key_post = _rotate(key_pre, all_positions, inv_freq)
    native = torch.matmul(query_post, key_post.transpose(2, 3))
    plan = {
        "positions": torch.tensor([[0], [2]]),
        "anchor_distances": torch.tensor([[1], [1]]),
        "baseline_certificates": torch.ones(2, 1),
    }

    modified, summary = target.apply_frozen_plan_to_scores(
        native,
        query_post,
        key_post,
        plan,
        groups=1,
        query_position=query_position,
        inv_freq=inv_freq,
        score_scale=1.0,
    )

    changed = (modified != native)[0, :, 0, :]
    assert changed[0].nonzero().reshape(-1).tolist() == [0]
    assert changed[1].nonzero().reshape(-1).tolist() == [2]
    assert summary["applied_count"] == 2


def test_assemble_case_is_exact_and_balanced_without_prompt_labels() -> None:
    records = [
        {
            "category": "gold_evidence",
            "text": "gold",
            "ids": [10, 11, 12],
            "decisive_local": [1],
        },
        {
            "category": "conflict_evidence",
            "text": "conflict",
            "ids": [20, 21, 22],
            "decisive_local": [1],
        },
        {
            "category": "lexical_format_distractor",
            "text": "lexical",
            "ids": [30, 31, 32, 33],
            "decisive_local": [2],
        },
    ]
    case = target.assemble_case_from_encoded_records(
        records,
        suffix_ids=[90, 91],
        filler_id=1,
        total_tokens=40,
        seed=0,
        packet_gap_tokens=3,
        class_sample_count=2,
    )

    assert len(case["prompt_ids"]) == 40
    assert case["query_span"] == [38, 40]
    assert all(len(case["sample_positions"][name]) == 2 for name in target.CLASS_ORDER)
    for category in target.CLASS_ORDER[:-1]:
        decisive = case["decisive_positions"][category][0]
        assert decisive in case["sample_positions"][category]


def test_truth_status_leakage_guard() -> None:
    target.assert_no_truth_status_leakage(
        "The school register lists Xiaoming's age as nine years."
    )
    with pytest.raises(ValueError):
        target.assert_no_truth_status_leakage("VERIFIED: Xiaoming is nine.")
    with pytest.raises(ValueError):
        target.assert_no_truth_status_leakage("This line is unverified.")


def test_answer_token_ids_score_digits_after_prompt_owned_space() -> None:
    seen: list[str] = []

    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            assert not add_special_tokens
            seen.append(text)
            return {"input_ids": [1000 + target.NUMBER_DIGITS.index(text)]}

    ids = target.answer_token_ids(FakeTokenizer())
    assert seen == list(target.NUMBER_DIGITS)
    assert ids["nine"] == 1000 + target.NUMBER_DIGITS.index("9")
    assert all(not digit.startswith(" ") for digit in seen)


def test_binary_auroc_is_tie_aware() -> None:
    assert target.binary_auroc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert target.binary_auroc([0.0, 1.0], [2.0, 3.0]) == 0.0
    assert target.binary_auroc([1.0, 1.0], [1.0, 1.0]) == 0.5


def test_native_cutoff_and_aggregate_keep_missing_64k_calibration_explicit(
    tmp_path: Path,
) -> None:
    assert target.native_baseline_enabled(8192, 32768)
    assert target.native_baseline_enabled(32768, 32768)
    assert not target.native_baseline_enabled(65536, 32768)
    assert target.native_baseline_enabled(65536, 0)

    zero = {
        "delta_gold_nll": 0.0,
        "gold_ppl_ratio": 1.0,
        "delta_gold_ppl": 0.0,
        "delta_gold_full_vocab_margin": 0.0,
        "delta_gold_conflict_margin": 0.0,
    }
    answer = {
        "gold_nll": 1.0,
        "gold_ppl": math.e,
        "gold_probability": math.exp(-1.0),
        "gold_full_vocab_margin": 0.5,
        "gold_conflict_margin": 0.75,
        "next_token_correct": 1,
    }
    case_rows = []
    for length in (8192, 32768, 65536):
        available = int(length <= 32768)
        common = {
            "target_context_tokens": length,
            "seed": 0,
            "primary_baseline": "instrumented_none",
            "native_baseline_available": available,
            "native_baseline_status": (
                "measured_untouched_native"
                if available
                else "skipped_context_exceeds_native_max"
            ),
        }
        if available:
            case_rows.append(
                {
                    **common,
                    "intervention_class": "native_baseline",
                    "comparison_baseline": "native_baseline",
                    **answer,
                    **zero,
                }
            )
        none_row = {
            **common,
            "intervention_class": "none",
            "comparison_baseline": "instrumented_none",
            **answer,
            **zero,
        }
        if available:
            none_row["instrumentation_delta_gold_nll"] = 0.01
        case_rows.append(none_row)
        intervention_row = {
            **common,
            "intervention_class": "gold_evidence",
            "comparison_baseline": "instrumented_none",
            **answer,
            **{**zero, "delta_gold_nll": -0.2},
        }
        if available:
            intervention_row["native_delta_gold_nll"] = -0.19
        case_rows.append(intervention_row)

    coverage = target.native_baseline_coverage(case_rows)
    at_64k = next(
        row for row in coverage if row["target_context_tokens"] == 65536
    )
    assert at_64k["instrumented_baseline_case_count"] == 1
    assert at_64k["native_baseline_measured_case_count"] == 0
    assert at_64k["native_baseline_skipped_case_count"] == 1
    assert at_64k["native_baseline_coverage_fraction"] == 0.0

    summaries = target.intervention_summary(
        case_rows,
        bootstrap_replicates=10,
        bootstrap_seed=5,
        minimum_bootstrap_seeds=1,
    )
    assert not any(
        row["target_context_tokens"] == 65536
        and row["intervention_class"] == "native_baseline"
        for row in summaries
    )
    none_64k = next(
        row
        for row in summaries
        if row["target_context_tokens"] == 65536
        and row["intervention_class"] == "none"
    )
    assert none_64k["comparison_baseline"] == "instrumented_none"
    assert none_64k["mean_delta_gold_nll"] == 0.0
    assert "mean_instrumentation_delta_gold_nll" not in none_64k
    native_all = next(
        row
        for row in summaries
        if row["target_context_tokens"] == "all"
        and row["intervention_class"] == "native_baseline"
    )
    assert native_all["observed_context_lengths"] == "8192,32768"

    csv_path = tmp_path / "case_rows.csv"
    target.write_csv(csv_path, case_rows)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    written_64k = [
        row for row in written if row["target_context_tokens"] == "65536"
    ]
    assert all(row["instrumentation_delta_gold_nll"] == "" for row in written_64k)
    assert all(row["native_delta_gold_nll"] == "" for row in written_64k)


def test_shard_aggregation_writes_distributions_and_aurocs(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    raw = shard / "raw"
    raw.mkdir(parents=True)
    case_rows = []
    for intervention in ("none", *target.CLASS_ORDER):
        case_rows.append(
            {
                "target_context_tokens": 128,
                "seed": 0,
                "intervention_class": intervention,
                "gold_nll": 1.0,
                "gold_ppl": math.e,
                "gold_probability": math.exp(-1.0),
                "gold_full_vocab_margin": 0.1,
                "gold_conflict_margin": 0.2,
                "next_token_correct": 1,
                "delta_gold_nll": 0.0,
                "gold_ppl_ratio": 1.0,
                "delta_gold_ppl": 0.0,
                "delta_gold_full_vocab_margin": 0.0,
                "delta_gold_conflict_margin": 0.0,
            }
        )
    target.write_json(raw / "length_128_seed_0_result.json", {"case_rows": case_rows})
    samples = []
    for class_index, category in enumerate(target.CLASS_ORDER):
        row = {
            "target_context_tokens": 128,
            "seed": 0,
            "class": category,
            "is_decisive_token": int(category != "filler"),
        }
        for field in target.RAW_SCORE_FIELDS:
            row[field] = float(class_index)
        for field in target.CERTIFICATE_FIELDS:
            row[field] = float(3 - class_index)
        samples.append(row)
    target.append_jsonl(
        raw / "length_128_seed_0_certificate_samples.jsonl", samples
    )

    merged = tmp_path / "merged"
    merged.mkdir()
    target.write_aggregate_outputs(
        merged,
        [shard],
        trigger_threshold=0.0,
        bootstrap_replicates=100,
        bootstrap_seed=7,
        minimum_bootstrap_seeds=4,
    )
    summary = target.json.loads((merged / "summary.json").read_text(encoding="utf-8"))

    assert summary["case_row_count"] == 5
    assert summary["certificate_sample_count"] == 4
    assert (merged / "certificate_aurocs.csv").exists()
    assert (merged / "intervention_summary.csv").exists()
    all_sampled_aurocs = [
        row
        for row in summary["certificate_aurocs"]
        if row["scope"] == "all_sampled"
    ]
    assert all(
        row["auroc_ci95_low"] == "NA"
        and row["bootstrap_status"].startswith("NA:insufficient_seeds")
        for row in all_sampled_aurocs
    )
    assert all(
        row["mean_delta_gold_nll_ci95_low"] == "NA"
        and row["mean_delta_gold_conflict_margin_ci95_high"] == "NA"
        and row["delta_gold_nll_bootstrap_status"].startswith(
            "NA:insufficient_seeds"
        )
        for row in summary["intervention_summary"]
    )


def _certificate_sample(length: int, seed: int, category: str, score: float) -> dict:
    row = {
        "target_context_tokens": length,
        "seed": seed,
        "class": category,
        "is_decisive_token": int(category != "filler"),
    }
    for field in target.RAW_SCORE_FIELDS:
        row[field] = score
    for field in target.CERTIFICATE_FIELDS:
        row[field] = score
    return row


def test_seed_stratified_auroc_bootstrap_has_exact_ci_when_separable() -> None:
    samples = []
    for seed in range(4):
        samples.extend(
            (
                _certificate_sample(128, seed, "gold_evidence", 4.0 + seed),
                _certificate_sample(128, seed, "conflict_evidence", 3.0 + seed),
                _certificate_sample(128, seed, "lexical_format_distractor", -2.0),
                _certificate_sample(128, seed, "filler", -3.0),
            )
        )
    rows = target.certificate_auroc_summary(
        samples,
        bootstrap_replicates=200,
        bootstrap_seed=11,
        minimum_bootstrap_seeds=4,
    )
    selected = next(
        row
        for row in rows
        if row["target_context_tokens"] == 128
        and row["scope"] == "all_sampled"
        and row["metric"] == "pre_suppression"
        and row["task"] == "gold_vs_lexical_format"
    )

    assert selected["auroc"] == 1.0
    assert selected["auroc_ci95_low"] == 1.0
    assert selected["auroc_ci95_high"] == 1.0
    assert selected["bootstrap_valid_replicates"] == 200
    assert selected["bootstrap_status"] == "ok"
    assert selected["bootstrap_seed_counts"] == "128:4"


def test_seed_stratified_bootstrap_is_na_below_minimum_seed_count() -> None:
    samples = []
    for seed in range(3):
        samples.extend(
            (
                _certificate_sample(128, seed, "gold_evidence", 2.0),
                _certificate_sample(128, seed, "conflict_evidence", 1.0),
                _certificate_sample(128, seed, "lexical_format_distractor", 0.0),
                _certificate_sample(128, seed, "filler", -1.0),
            )
        )
    rows = target.certificate_auroc_summary(
        samples,
        bootstrap_replicates=100,
        bootstrap_seed=13,
        minimum_bootstrap_seeds=4,
    )
    assert rows
    all_sampled = [row for row in rows if row["scope"] == "all_sampled"]
    assert all(row["auroc_ci95_low"] == "NA" for row in all_sampled)
    assert all(row["auroc_ci95_high"] == "NA" for row in all_sampled)
    assert all(
        row["bootstrap_status"].startswith("NA:insufficient_seeds[128:3]")
        for row in all_sampled
    )


def test_auroc_reports_raw_scores_and_decisive_scope() -> None:
    samples = []
    for seed in range(4):
        samples.extend(
            (
                _certificate_sample(128, seed, "gold_evidence", 4.0),
                _certificate_sample(128, seed, "conflict_evidence", 3.0),
                _certificate_sample(
                    128, seed, "lexical_format_distractor", 1.0
                ),
                _certificate_sample(128, seed, "filler", 0.0),
            )
        )
    rows = target.certificate_auroc_summary(
        samples,
        bootstrap_replicates=100,
        bootstrap_seed=23,
        minimum_bootstrap_seeds=4,
    )
    raw = next(
        row
        for row in rows
        if row["target_context_tokens"] == 128
        and row["scope"] == "decisive_only"
        and row["metric"] == "grid_envelope_score"
        and row["task"] == "gold_vs_conflict"
    )
    assert raw["auroc"] == 1.0
    assert raw["pooled_auroc"] == 1.0
    filler = next(
        row
        for row in rows
        if row["target_context_tokens"] == 128
        and row["scope"] == "decisive_only"
        and row["metric"] == "grid_envelope_score"
        and row["task"] == "gold_vs_filler"
    )
    assert filler["negative_count"] == 0
    assert filler["auroc"] == "NA"
    assert filler["bootstrap_status"] == "NA:no_rows"


def test_all_length_bootstrap_rejects_unbalanced_seed_grid() -> None:
    rows = [
        {"target_context_tokens": 128, "seed": seed, "value": float(seed)}
        for seed in (0, 1, 2, 3)
    ]
    rows.extend(
        {"target_context_tokens": 256, "seed": seed, "value": float(seed)}
        for seed in (1, 2, 3, 4)
    )
    result = target.seed_stratified_bootstrap(
        rows,
        lambda sampled: sum(float(row["value"]) for row in sampled) / len(sampled),
        replicates=100,
        random_seed=19,
        minimum_seeds_per_stratum=4,
    )

    assert result["ci95_low"] == "NA"
    assert result["ci95_high"] == "NA"
    assert result["bootstrap_status"] == "NA:unbalanced_seed_length_grid"


def test_intervention_seed_bootstrap_reports_required_effect_cis() -> None:
    rows = []
    for seed in range(4):
        rows.append(
            {
                "target_context_tokens": 128,
                "seed": seed,
                "intervention_class": "gold_evidence",
                "gold_nll": 1.0 + seed,
                "gold_ppl": math.exp(1.0 + seed),
                "gold_probability": math.exp(-(1.0 + seed)),
                "gold_full_vocab_margin": 0.0,
                "gold_conflict_margin": 1.0 - seed,
                "next_token_correct": int(seed == 0),
                "delta_gold_nll": float(seed),
                "gold_ppl_ratio": 1.0,
                "delta_gold_ppl": float(seed),
                "delta_gold_full_vocab_margin": -float(seed),
                "delta_gold_conflict_margin": -2.0 * seed,
            }
        )
    summary = target.intervention_summary(
        rows,
        bootstrap_replicates=300,
        bootstrap_seed=17,
        minimum_bootstrap_seeds=4,
    )
    selected = next(
        row
        for row in summary
        if row["target_context_tokens"] == 128
        and row["intervention_class"] == "gold_evidence"
    )

    assert selected["mean_delta_gold_nll"] == 1.5
    assert selected["gold_ppl_exp_mean_nll"] == pytest.approx(math.exp(2.5))
    assert selected["mean_delta_gold_conflict_margin"] == -3.0
    assert selected["mean_delta_gold_nll_ci95_low"] <= 1.5
    assert selected["mean_delta_gold_nll_ci95_high"] >= 1.5
    assert selected["mean_delta_gold_conflict_margin_ci95_low"] <= -3.0
    assert selected["mean_delta_gold_conflict_margin_ci95_high"] >= -3.0
    assert selected["delta_gold_nll_bootstrap_status"] == "ok"
    assert selected["delta_gold_conflict_margin_bootstrap_status"] == "ok"
