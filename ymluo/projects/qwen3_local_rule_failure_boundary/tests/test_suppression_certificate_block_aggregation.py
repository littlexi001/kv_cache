from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import analyze_suppression_certificate_block_aggregation as target  # noqa: E402


def _write_case(
    raw: Path,
    *,
    seed: int,
    gold_base: float,
    conflict_base: float,
    length: int = 128,
) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    stem = f"length_{length}_seed_{seed}"
    sample_path = raw / f"{stem}_certificate_samples.jsonl"
    result_path = raw / f"{stem}_result.json"
    class_tokens = {
        "gold_evidence": ((11, 0), (15, 1)),
        "conflict_evidence": ((31, 0), (35, 1)),
        "lexical_format_distractor": ((51, 0), (55, 1)),
        "filler": ((70, 0), (71, 0)),
    }
    class_base = {
        "gold_evidence": gold_base,
        "conflict_evidence": conflict_base,
        "lexical_format_distractor": 0.5,
        "filler": -0.5,
    }
    metric_offsets = {
        "pre_score": 0.0,
        "post_score": 0.2,
        "pre_suppression": 0.4,
        "grid_envelope_suppression": 0.6,
    }
    rows = []
    for layer in range(2):
        for head in range(2):
            layer_head_offset = 0.1 * layer + 0.01 * head
            for category, tokens in class_tokens.items():
                for sample_index, (position, decisive) in enumerate(tokens):
                    token_offset = 0.001 * sample_index
                    row = {
                        "target_context_tokens": length,
                        "seed": seed,
                        "layer": layer,
                        "head": head,
                        "class": category,
                        "sample_index": sample_index,
                        "token_position": position,
                        "relative_distance": length - 1 - position,
                        "is_decisive_token": decisive,
                    }
                    for metric, offset in metric_offsets.items():
                        row[metric] = (
                            class_base[category]
                            + offset
                            + layer_head_offset
                            + token_offset
                        )
                    rows.append(row)
    sample_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = {
        "case": {
            "records": [
                {
                    "category": "gold_evidence",
                    "text": "Gold evidence line.\n",
                    "span": [10, 20],
                },
                {
                    "category": "conflict_evidence",
                    "text": "Conflict evidence line.\n",
                    "span": [30, 40],
                },
                {
                    "category": "lexical_format_distractor",
                    "text": "Lexical distractor line.\n",
                    "span": [50, 60],
                },
            ]
        }
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return sample_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_quantile_and_tie_aware_auroc_are_dependency_free() -> None:
    assert target.quantile([0.0, 10.0], 0.9) == pytest.approx(9.0)
    assert target.binary_auroc([2.0], [1.0]) == 1.0
    assert target.binary_auroc([1.0], [1.0]) == 0.5
    assert math.isnan(target.binary_auroc([], [1.0]))
    source = (SRC / "analyze_suppression_certificate_block_aggregation.py").read_text(
        encoding="utf-8"
    )
    assert "import torch" not in source
    assert "sklearn" not in source


def test_analysis_reports_token_auc_line_pairs_and_decisive_scope(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard"
    raw = shard / "raw"
    _write_case(raw, seed=0, gold_base=4.0, conflict_base=1.0)
    _write_case(raw, seed=1, gold_base=1.0, conflict_base=4.0)
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "summary.json").write_text(
        json.dumps({"source_dirs": [str(shard)]}), encoding="utf-8"
    )

    output = tmp_path / "analysis"
    summary = target.analyze([merged], output)

    assert summary["cpu_only"] is True
    assert summary["raw_sample_count"] == 64
    assert summary["duplicate_row_count"] == 0
    assert summary["line_metadata_source_counts"][
        "result_json_record_span"
    ] == 48
    assert summary["line_metadata_source_counts"][
        "class_fallback_position_outside_record_spans"
    ] == 16
    assert summary["loso_combination_enabled"] is False
    assert not (output / "loso_line_combination.csv").exists()
    for name in (
        "token_aggregates.csv",
        "token_aurocs.csv",
        "line_aggregates.csv",
        "paired_line_comparisons.csv",
        "paired_line_summary.csv",
        "summary.json",
    ):
        assert (output / name).exists()

    token_rows = _read_csv(output / "token_aggregates.csv")
    # 16 all-sampled token cases plus six decisive evidence token cases.
    assert len(token_rows) == 22
    selected_token = next(
        row
        for row in token_rows
        if row["target_context_tokens"] == "128"
        and row["seed"] == "0"
        and row["scope"] == "all_sampled"
        and row["class"] == "gold_evidence"
        and row["token_position"] == "11"
    )
    assert float(selected_token["pre_score__mean"]) == pytest.approx(4.055)
    assert float(selected_token["pre_score__max"]) == pytest.approx(4.11)
    assert float(selected_token["pre_score__q90"]) == pytest.approx(4.107)
    assert float(selected_token["pre_score__positive_fraction"]) == 1.0
    assert selected_token["line_id"] == "record_000_10_20"

    auroc_rows = _read_csv(output / "token_aurocs.csv")
    seed0 = next(
        row
        for row in auroc_rows
        if row["evaluation_level"] == "within_seed"
        and row["target_context_tokens"] == "128"
        and row["seed"] == "0"
        and row["scope"] == "all_sampled"
        and row["contrast"] == "gold_vs_conflict"
        and row["metric"] == "pre_score"
        and row["layer_head_reducer"] == "mean"
    )
    seed1 = next(
        row
        for row in auroc_rows
        if row["evaluation_level"] == "within_seed"
        and row["target_context_tokens"] == "128"
        and row["seed"] == "1"
        and row["scope"] == "all_sampled"
        and row["contrast"] == "gold_vs_conflict"
        and row["metric"] == "pre_score"
        and row["layer_head_reducer"] == "mean"
    )
    macro = next(
        row
        for row in auroc_rows
        if row["evaluation_level"] == "macro_mean_of_within_seed_aurocs"
        and row["target_context_tokens"] == "128"
        and row["scope"] == "all_sampled"
        and row["contrast"] == "gold_vs_conflict"
        and row["metric"] == "pre_score"
        and row["layer_head_reducer"] == "mean"
    )
    assert float(seed0["auroc"]) == 1.0
    assert float(seed1["auroc"]) == 0.0
    assert float(macro["auroc"]) == 0.5
    decisive_auc = next(
        row
        for row in auroc_rows
        if row["evaluation_level"] == "within_seed"
        and row["target_context_tokens"] == "128"
        and row["seed"] == "0"
        and row["scope"] == "decisive_only"
        and row["contrast"] == "gold_vs_all_nongold"
        and row["metric"] == "pre_score"
        and row["layer_head_reducer"] == "mean"
    )
    assert int(decisive_auc["positive_token_count"]) == 1
    assert int(decisive_auc["negative_token_count"]) == 2

    line_rows = _read_csv(output / "line_aggregates.csv")
    gold_line = next(
        row
        for row in line_rows
        if row["seed"] == "0"
        and row["scope"] == "all_sampled"
        and row["class"] == "gold_evidence"
        and row["metric"] == "pre_score"
        and row["line_reducer"] == "mean"
    )
    assert int(gold_line["sampled_token_count"]) == 2
    assert int(gold_line["line_score_row_count"]) == 8
    paired = _read_csv(output / "paired_line_comparisons.csv")
    assert any(row["scope"] == "decisive_only" for row in paired)
    pair_summary = _read_csv(output / "paired_line_summary.csv")
    selected_pair = next(
        row
        for row in pair_summary
        if row["target_context_tokens"] == "128"
        and row["scope"] == "all_sampled"
        and row["metric"] == "pre_score"
        and row["line_reducer"] == "mean"
    )
    assert int(selected_pair["paired_seed_count"]) == 2
    assert float(selected_pair["paired_win_rate_ties_half"]) == 0.5
    assert float(selected_pair["mean_gold_minus_conflict_gap"]) == 0.0


def test_optional_loso_is_explicit_and_never_uses_held_out_scaling(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "shard" / "raw"
    _write_case(raw, seed=0, gold_base=4.0, conflict_base=1.0)
    _write_case(raw, seed=1, gold_base=1.0, conflict_base=4.0)
    _write_case(raw, seed=2, gold_base=3.0, conflict_base=2.0)
    output = tmp_path / "analysis"

    summary = target.analyze(
        [raw.parent], output, enable_loso_combination=True
    )

    assert summary["loso_combination_enabled"] is True
    rows = _read_csv(output / "loso_line_combination.csv")
    assert rows
    for row in rows:
        held_out = row["held_out_seed"]
        training_ids = row["training_seed_ids"].split(",")
        assert held_out not in training_ids
        assert int(row["training_seed_count"]) == 2
        assert row["feature_weight_policy"] == "fixed_equal_positive_weights"
        assert (
            row["standardization_policy"]
            == "unlabeled_training_seeds_only_mean_and_pstdev"
        )
        assert int(row["uses_labels_for_weights"]) == 0


def test_missing_result_json_is_explicit_class_fallback(tmp_path: Path) -> None:
    sample_path = _write_case(
        tmp_path, seed=0, gold_base=2.0, conflict_base=1.0
    )
    sample_path.with_name("length_128_seed_0_result.json").unlink()
    samples, summary = target.load_samples(
        [sample_path], target.DEFAULT_METRICS
    )

    assert {
        row["line_metadata_source"] for row in samples
    } == {"class_fallback_no_result_json"}
    assert summary["line_metadata_source_counts"] == {
        "class_fallback_no_result_json": 32
    }
