from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import analyze_value_mediated_noop_seed_cluster as target  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path, *, context_length: int = 8192) -> Path:
    merged = root / "merged"
    case_rows: list[dict] = []
    classes = target.CLASS_ORDER
    for seed in range(2):
        correct = int(seed == 0)
        for baseline in ("native_baseline", "instrumented_baseline", "custom_noop_baseline"):
            case_rows.append(
                {
                    "seed": seed,
                    "target_context_tokens": context_length,
                    "intervention_class": baseline,
                    "plan_kind": "none",
                    "intervention_scope": "noop" if baseline == "custom_noop_baseline" else "none",
                    "prediction_token_id": 9 if correct else 4,
                    "final_hidden_source_dtype": "bfloat16",
                }
            )
        for class_index, category in enumerate(classes):
            for kind in ("target", "random"):
                sign = 1.0 if category == "gold_evidence" else -1.0
                scale = 0.2 + 0.05 * seed if kind == "target" else 0.01
                predicted = sign * scale if category in target.EVIDENCE_CLASSES else 0.01
                actual = predicted * (1.0 + 0.05 * seed)
                case_rows.append(
                    {
                        "seed": seed,
                        "target_context_tokens": context_length,
                        "intervention_class": category,
                        "plan_kind": kind,
                        "intervention_scope": "singleton",
                        "uniform_score_lift": 0.25,
                        "pair_id": f"{category}_001",
                        target.PREDICTED: predicted,
                        target.ACTUAL: actual,
                        "first_order_sign_match": int(predicted * actual > 0),
                        "first_order_symmetric_closure_error": abs(predicted - actual)
                        / (abs(predicted) + abs(actual)),
                        "first_order_absolute_closure_error": abs(predicted - actual),
                        "delta_gold_nll": -actual / 10.0,
                    }
                )
    write_csv(merged / "case_rows.csv", case_rows)

    sample_rows: list[dict] = []
    for seed in range(2):
        for category in classes:
            for sample_index in range(2):
                direction = 1.0 if category == "gold_evidence" else -1.0 if category == "conflict_evidence" else 0.0
                sample_rows.append(
                    {
                        "seed": seed,
                        "layer": 0,
                        "head": 0,
                        "class": category,
                        "sample_index": sample_index,
                        "token_position": 10 + sample_index,
                        "dm_dscore": direction * (seed + 1),
                        "direct_ov_centered_margin_derivative": direction * 2.0,
                        "attention_probability": 0.1,
                        "suppression_gap": 1.0,
                    }
                )
    write_csv(merged / "value_samples.csv", sample_rows)
    wrong_merge = {
        "lengths": "8192,32768",
        "num_seeds": 4,
        "class_sample_count": 8,
        "singleton_top_n": 16,
    }
    (merged / "merge_config.json").write_text(
        json.dumps(wrong_merge), encoding="utf-8"
    )

    for seed in range(2):
        raw = root / ("shard_gpu6" if seed == 0 else "shard_gpu7") / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        candidates = []
        for category in classes:
            decisive = int(category in target.EVIDENCE_CLASSES)
            candidates.append(
                {
                    "class": category,
                    "target": {"is_decisive_token": decisive},
                    "random": {"is_decisive_token": 0},
                }
            )
        result = {
            "schema_version": 3,
            "experiment": "fixture",
            "singleton_top_n_per_class": 1,
            "singleton_candidate_ranking_metric": "fixture_metric",
            "case_replay_audit": {"passed": True},
            "prefix_cache_immutable": True,
            "custom_noop_delta_from_native": {"delta_gold_conflict_margin": 0.1},
            "custom_noop_delta_from_instrumented": {
                "delta_gold_conflict_margin": 0.0,
                "delta_gold_nll": 0.0,
            },
            "frozen_singleton_candidates": candidates,
        }
        (raw / f"length_{context_length}_seed_{seed}_result.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
    return root


def test_seed_cluster_analysis_is_reproducible_and_preserves_merge_config(
    tmp_path: Path,
) -> None:
    root = build_fixture(tmp_path / "fixture")
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        "CUDA_VISIBLE_DEVICES=6 CUDA_VISIBLE_DEVICES=7 --dtype bfloat16\n",
        encoding="utf-8",
    )
    original_merge = (root / "merged" / "merge_config.json").read_bytes()

    first = target.run_analysis(
        root, resamples=200, rng_seed=20260801, launcher=launcher
    )
    second = target.run_analysis(
        root, resamples=200, rng_seed=20260801, launcher=launcher
    )

    assert first == second
    audit, provenance, report = first
    assert audit["methodology"]["independent_unit"] == "seed"
    assert audit["intervention_summaries"]["evidence_only_target"]["n_seeds"] == 2
    assert audit["intervention_summaries"]["evidence_only_target"]["n_events"] == 4
    assert audit["sample_seed_macro"]["gold_minus_conflict"]["dm_dscore"][
        "gold_minus_conflict_seed_macro_mean"
    ] > 0
    assert audit["integrity"]["case_row_count"] == audit["integrity"][
        "case_unique_key_count"
    ]
    assert provenance["context_lengths"] == [8192]
    assert provenance["num_seeds"] == 2
    assert provenance["class_sample_count_per_layer_head_class_inferred"] == [2]
    assert provenance["singleton_top_n_per_class"] == [1]
    merge_provenance = provenance["merge_provenance"]
    assert merge_provenance["status"] == "legacy_merge_invocation_defaults"
    assert merge_provenance["warning_required"] is True
    assert provenance["merge_config_discrepancies"]["singleton_top_n"][
        "reported"
    ] == 16
    assert (root / "merged" / "merge_config.json").read_bytes() == original_merge
    assert "独立统计单位是 **seed**" in report
    assert "first_order_prediction_summary" in report
    assert "旧 merge 调用/默认参数快照" in report


def test_shard_derived_merge_config_is_not_misreported_as_legacy(
    tmp_path: Path,
) -> None:
    context_length = 16384
    root = build_fixture(tmp_path / "fixture", context_length=context_length)
    merged = root / "merged"
    shard_derived_config = {
        "merge_schema_version": 2,
        "shared_config": {
            "class_sample_count": 2,
            "singleton_top_n": 1,
            "model_weight_mode": "unquantized_bfloat16",
        },
        "shards": [
            {
                "source_dir": "shard_gpu6",
                "config": {
                    "resolved_lengths": [context_length],
                    "seed_start": 0,
                    "num_seeds": 1,
                },
            },
            {
                "source_dir": "shard_gpu7",
                "config": {
                    "resolved_lengths": [context_length],
                    "seed_start": 1,
                    "num_seeds": 1,
                },
            },
        ],
    }
    current_path = merged / "merge_config.json"
    current_path.write_text(json.dumps(shard_derived_config), encoding="utf-8")
    (merged / "merge_config_legacy_incorrect.json").write_text(
        json.dumps({"lengths": "8192,32768", "num_seeds": 4}),
        encoding="utf-8",
    )
    original_merge = current_path.read_bytes()

    _, provenance, report = target.run_analysis(
        root, resamples=50, rng_seed=20260801, launcher=None
    )

    merge_provenance = provenance["merge_provenance"]
    assert provenance["context_lengths"] == [context_length]
    assert merge_provenance["status"] == "shard_derived_schema"
    assert merge_provenance["warning_required"] is False
    assert merge_provenance["discrepancies"] == {}
    assert merge_provenance["legacy_incorrect_files_present"] == [
        "merge_config_legacy_incorrect.json"
    ]
    assert current_path.read_bytes() == original_merge
    assert "上下文长度 16,384 tokens" in report
    assert "仅为旧错误配置的归档，不参与本报告统计" in report
    assert "旧 merge 调用/默认参数快照" not in report


def test_percentile_and_tie_aware_correlation() -> None:
    assert target.percentile([0.0, 10.0], 0.5) == 5.0
    assert target.correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert target.average_ranks([3.0, 1.0, 1.0]) == [3.0, 1.5, 1.5]
