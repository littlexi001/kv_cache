from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


PUNCTUATION_VARIANTS = (
    "period_with_distractors",
    "comma_with_distractors",
    "question_with_distractors",
    "semicolon_with_distractors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--existing-points-csv", required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def token_name(key: str) -> str:
    fields = key.split(":", 1)
    return fields[1] if len(fields) == 2 else key


def format_total_kib(value: int | None) -> str:
    return "—" if value is None else f"{value / 1024:.3f}K"


def scan_summary(
    rows: list[dict[str, str]],
    *,
    stride: int,
) -> dict[str, Any]:
    failures = np.array(
        [not bool_value(row["top_is_gold"]) for row in rows],
        dtype=np.bool_,
    )
    totals = np.array(
        [int(row["total_tokens"]) for row in rows],
        dtype=np.int64,
    )
    first_failure = (
        int(totals[np.flatnonzero(failures)[0]])
        if failures.any()
        else None
    )

    consecutive_points = max(1, math.ceil(64 / stride))
    first_persistent = None
    if len(failures) >= consecutive_points:
        rolling = np.convolve(
            failures.astype(np.int64),
            np.ones(consecutive_points, dtype=np.int64),
            mode="valid",
        )
        matches = np.flatnonzero(rolling == consecutive_points)
        if len(matches):
            first_persistent = int(totals[matches[0]])

    window_points = max(1, math.ceil(512 / stride))
    first_majority = None
    if len(failures) >= window_points:
        rolling_rate = np.convolve(
            failures.astype(np.float64),
            np.ones(window_points, dtype=np.float64),
            mode="valid",
        ) / window_points
        matches = np.flatnonzero(rolling_rate > 0.5)
        if len(matches):
            first_majority = int(
                totals[matches[0] + window_points - 1]
            )

    failure_winners = collections.Counter(
        row["top_token_label"]
        for row, failed in zip(rows, failures)
        if failed
    )
    window_competitors = {}
    for lower_k, upper_k, include_upper in (
        (136, 138, False),
        (138, 140, False),
        (140, 142, False),
        (142, 144, True),
    ):
        selected = [
            row
            for row in rows
            if int(row["total_tokens"]) >= lower_k * 1024
            and (
                int(row["total_tokens"]) <= upper_k * 1024
                if include_upper
                else int(row["total_tokens"]) < upper_k * 1024
            )
        ]
        counts = collections.Counter(
            row["strongest_competitor_token_label"]
            for row in selected
        )
        winner_counts = collections.Counter(
            row["top_token_label"]
            for row in selected
            if not bool_value(row["top_is_gold"])
        )
        window_competitors[f"{lower_k}k_to_{upper_k}k"] = {
            "point_count": len(selected),
            "dominant_competitor": (
                counts.most_common(1)[0][0] if counts else None
            ),
            "competitor_counts": dict(counts.most_common()),
            "dominant_failure_winner": (
                winner_counts.most_common(1)[0][0]
                if winner_counts
                else None
            ),
            "failure_winner_counts": dict(
                winner_counts.most_common()
            ),
        }
    return {
        "failure_rate": float(failures.mean()),
        "first_failure_total_tokens": first_failure,
        "first_failure_kib_tokens": (
            first_failure / 1024 if first_failure is not None else None
        ),
        "first_64_token_persistent_failure_start": first_persistent,
        "first_64_token_persistent_failure_start_kib": (
            first_persistent / 1024
            if first_persistent is not None
            else None
        ),
        "first_512_token_window_majority_failure_end": first_majority,
        "first_512_token_window_majority_failure_end_kib": (
            first_majority / 1024
            if first_majority is not None
            else None
        ),
        "correct_failure_flip_count": int(
            np.count_nonzero(failures[1:] != failures[:-1])
        ),
        "mean_gold_probability": float(
            np.mean(
                [
                    float(row["gold_exact_probability"])
                    for row in rows
                ]
            )
        ),
        "mean_output_margin": float(
            np.mean(
                [
                    float(row["gold_exact_vs_competitor_margin"])
                    for row in rows
                ]
            )
        ),
        "mean_strongest_competitor_probability": float(
            np.mean(
                [
                    float(row["strongest_competitor_probability"])
                    for row in rows
                ]
            )
        ),
        "mean_top_probability": float(
            np.mean(
                [float(row["top_probability"]) for row in rows]
            )
        ),
        "failure_winner_counts": dict(
            failure_winners.most_common()
        ),
        "competitor_by_2k_window": window_competitors,
    }


def regression(
    features: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    prediction = x @ coefficients
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return {
        "r2": r2,
        "coefficients": coefficients.tolist(),
        "prediction": prediction,
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def punctuation_summary(root: Path) -> dict[str, Any]:
    output = {}
    for name in PUNCTUATION_VARIANTS:
        manifest = load_json(root / name / "manifest.json")
        rows = read_rows(root / name / "points.csv")
        total = int(manifest["completed_points"])
        competitor_counts = manifest["strongest_competitor_counts"]
        dominant = (
            max(competitor_counts, key=competitor_counts.get)
            if competitor_counts
            else None
        )
        by_total = {
            int(row["total_tokens"]): row for row in rows
        }
        anchors = {}
        for anchor_total in (
            136 * 1024,
            140 * 1024,
            142 * 1024,
            144 * 1024,
        ):
            row = by_total[anchor_total]
            anchors[str(anchor_total)] = {
                "gold_probability": float(
                    row["gold_exact_probability"]
                ),
                "output_margin": float(
                    row["gold_exact_vs_competitor_margin"]
                ),
                "top_token_label": row["top_token_label"],
                "strongest_competitor_token_label": row[
                    "strongest_competitor_token_label"
                ],
                "strongest_competitor_probability": float(
                    row["strongest_competitor_probability"]
                ),
            }
        output[name] = {
            "filler_text": manifest["filler_text"],
            "sample_stride": manifest["stride"],
            "point_count": total,
            "competitor_counts": competitor_counts,
            "competitor_shares": {
                key: value / total
                for key, value in competitor_counts.items()
            },
            "dominant_competitor": dominant,
            "anchor_metrics": anchors,
            **scan_summary(
                rows,
                stride=int(manifest["stride"]),
            ),
        }
    return output


def generation_summary(root: Path) -> dict[str, Any]:
    result = load_json(
        root
        / "period_multitoken_generation"
        / "results.json"
    )
    selection = load_json(
        root
        / "period_multitoken_generation"
        / "selection.json"
    )
    rows = result["rows"]
    wrong_first = [
        row for row in rows if not row["top_is_gold"]
    ]
    by_top: dict[str, dict[str, int]] = {}
    for row in rows:
        label = row["top_token_label"]
        entry = by_top.setdefault(
            label,
            {
                "count": 0,
                "contains_nine": 0,
                "first_number_nine": 0,
                "no_number": 0,
            },
        )
        generation = row["generation"]
        normalized = generation["normalized"]
        number_matches = re.findall(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
            r"1|2|3|4|5|6|7|8|9|10)\b",
            normalized,
        )
        normalized_numbers = [
            "nine" if value == "9" else value
            for value in number_matches
        ]
        contains_nine = "nine" in normalized_numbers
        first_number_nine = bool(
            normalized_numbers
            and normalized_numbers[0] == "nine"
        )
        entry["count"] += 1
        entry["contains_nine"] += int(contains_nine)
        entry["first_number_nine"] += int(
            first_number_nine
        )
        entry["no_number"] += int(
            not normalized_numbers
        )
        row["_analysis_contains_nine_or_9"] = contains_nine
        row["_analysis_first_number_nine_or_9"] = first_number_nine
        row["_analysis_number_mentions"] = normalized_numbers
    return {
        "selection_source": selection["source"],
        "selected_added_tokens": selection[
            "selected_added_tokens"
        ],
        "selected_count": len(rows),
        "wrong_first_token_count": len(wrong_first),
        "wrong_first_then_contains_nine": sum(
            row["_analysis_contains_nine_or_9"]
            for row in wrong_first
        ),
        "wrong_first_then_first_number_nine": sum(
            row["_analysis_first_number_nine_or_9"]
            for row in wrong_first
        ),
        "all_generations_contain_nine": sum(
            row["_analysis_contains_nine_or_9"]
            for row in rows
        ),
        "all_generations_first_number_nine": sum(
            row["_analysis_first_number_nine_or_9"]
            for row in rows
        ),
        "multi_token_first_number_accuracy": (
            sum(
                row["_analysis_first_number_nine_or_9"]
                for row in rows
            )
            / len(rows)
        ),
        "by_first_token": by_top,
        "rows": rows,
    }


def no_distractor_summary(root: Path) -> dict[str, Any]:
    manifest = load_json(
        root
        / "period_without_distractors"
        / "manifest.json"
    )
    baseline = load_json(
        root
        / "period_with_distractors"
        / "manifest.json"
    )
    output = {}
    for label, name, item in (
        (
            "without_distractors",
            "period_without_distractors",
            manifest,
        ),
        (
            "with_distractors_stride8",
            "period_with_distractors",
            baseline,
        ),
    ):
        rows = read_rows(root / name / "points.csv")
        total = int(item["completed_points"])
        competitor_counts = item["strongest_competitor_counts"]
        output[label] = {
            "point_count": total,
            "competitor_counts": competitor_counts,
            "competitor_shares": {
                key: value / total
                for key, value in competitor_counts.items()
            },
            **scan_summary(
                rows,
                stride=int(item["stride"]),
            ),
        }
        if label == "without_distractors":
            matched_rows = [
                row
                for row in rows
                if int(row["added_tokens"]) % 8 == 0
            ]
            output["without_distractors_stride8_matched"] = {
                "point_count": len(matched_rows),
                **scan_summary(matched_rows, stride=8),
            }
    return output


def q_mean_summary(root: Path) -> dict[str, Any]:
    rows = read_rows(root / "q_rope_probe" / "q_mean_heads.csv")
    grouped: dict[int, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        grouped[int(row["total_tokens"])].append(row)
    curve = {}
    for total, group in sorted(grouped.items()):
        global_cos = np.array(
            [float(row["q_global_cosine"]) for row in group]
        )
        period_cos = np.array(
            [float(row["q_period_cosine"]) for row in group]
        )
        global_l2 = np.array(
            [float(row["q_global_relative_l2"]) for row in group]
        )
        period_l2 = np.array(
            [float(row["q_period_relative_l2"]) for row in group]
        )
        curve[str(total)] = {
            "kib_tokens": total / 1024,
            "q_global_cosine_mean": float(global_cos.mean()),
            "q_global_cosine_median": float(
                np.median(global_cos)
            ),
            "q_period_cosine_mean": float(period_cos.mean()),
            "q_period_cosine_median": float(
                np.median(period_cos)
            ),
            "q_global_relative_l2_mean": float(global_l2.mean()),
            "q_period_relative_l2_mean": float(period_l2.mean()),
            "head_count": len(group),
        }
    first_total = min(grouped)
    last_total = max(grouped)
    first = curve[str(first_total)]
    last = curve[str(last_total)]
    first_by_head = {
        (int(row["layer"]), int(row["head"])): row
        for row in grouped[first_total]
    }
    last_by_head = {
        (int(row["layer"]), int(row["head"])): row
        for row in grouped[last_total]
    }
    pairs = sorted(set(first_by_head) & set(last_by_head))
    global_cos_increased = np.array(
        [
            float(last_by_head[pair]["q_global_cosine"])
            > float(first_by_head[pair]["q_global_cosine"])
            for pair in pairs
        ]
    )
    period_cos_increased = np.array(
        [
            float(last_by_head[pair]["q_period_cosine"])
            > float(first_by_head[pair]["q_period_cosine"])
            for pair in pairs
        ]
    )
    global_l2_decreased = np.array(
        [
            float(last_by_head[pair]["q_global_relative_l2"])
            < float(first_by_head[pair]["q_global_relative_l2"])
            for pair in pairs
        ]
    )
    period_l2_decreased = np.array(
        [
            float(last_by_head[pair]["q_period_relative_l2"])
            < float(first_by_head[pair]["q_period_relative_l2"])
            for pair in pairs
        ]
    )
    anchor_totals = (136 * 1024, 140 * 1024, 142 * 1024, 144 * 1024)
    anchors = {
        str(total): curve[str(total)]
        for total in anchor_totals
    }
    return {
        "sample_stride": (
            sorted(grouped)[1] - sorted(grouped)[0]
            if len(grouped) > 1
            else None
        ),
        "sample_count": len(grouped),
        "curve": curve,
        "anchors": anchors,
        "change_136k_to_144k": {
            key: last[key] - first[key]
            for key in (
                "q_global_cosine_mean",
                "q_period_cosine_mean",
                "q_global_relative_l2_mean",
                "q_period_relative_l2_mean",
            )
        },
        "headwise_approach_fraction_136k_to_144k": {
            "global_cosine_increased": float(
                global_cos_increased.mean()
            ),
            "global_relative_l2_decreased": float(
                global_l2_decreased.mean()
            ),
            "global_both": float(
                (global_cos_increased & global_l2_decreased).mean()
            ),
            "period_cosine_increased": float(
                period_cos_increased.mean()
            ),
            "period_relative_l2_decreased": float(
                period_l2_decreased.mean()
            ),
            "period_both": float(
                (period_cos_increased & period_l2_decreased).mean()
            ),
            "head_count": len(pairs),
        },
    }


def head_decomposition_summary(root: Path) -> dict[str, Any]:
    rows = read_rows(
        root / "q_rope_probe" / "probe_head_points.csv"
    )
    grouped: dict[tuple[int, int], list[dict[str, str]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)

    def window_mean(
        group: list[dict[str, str]],
        field: str,
        lower: int,
        upper: int,
        *,
        include_upper: bool = False,
    ) -> float:
        values = [
            float(row[field])
            for row in group
            if int(row["total_tokens"]) >= lower
            and (
                int(row["total_tokens"]) <= upper
                if include_upper
                else int(row["total_tokens"]) < upper
            )
        ]
        return float(np.mean(values))

    summaries = []
    for (layer, head), group in sorted(grouped.items()):
        early_actual = window_mean(
            group,
            "weighted_actual",
            136 * 1024,
            140 * 1024,
        )
        failure_actual = window_mean(
            group,
            "weighted_actual",
            140 * 1024,
            141 * 1024,
        )
        recovery_actual = window_mean(
            group,
            "weighted_actual",
            142 * 1024,
            144 * 1024,
            include_upper=True,
        )
        summaries.append(
            {
                "layer": layer,
                "head": head,
                "weight": float(group[0]["weight"]),
                "early_actual": early_actual,
                "failure_actual": failure_actual,
                "recovery_actual": recovery_actual,
                "drop_early_to_140_141": (
                    failure_actual - early_actual
                ),
                "recovery_140_141_to_142_144": (
                    recovery_actual - failure_actual
                ),
                "failure_window_position_delta": window_mean(
                    group,
                    "weighted_position_delta",
                    140 * 1024,
                    141 * 1024,
                ),
                "failure_window_content_delta": window_mean(
                    group,
                    "weighted_content_delta",
                    140 * 1024,
                    141 * 1024,
                ),
                "failure_window_interaction": window_mean(
                    group,
                    "weighted_interaction",
                    140 * 1024,
                    141 * 1024,
                ),
            }
        )
    return {
        "head_count": len(summaries),
        "largest_drop_heads": sorted(
            summaries,
            key=lambda row: row["drop_early_to_140_141"],
        )[:10],
        "largest_recovery_heads": sorted(
            summaries,
            key=lambda row: row[
                "recovery_140_141_to_142_144"
            ],
            reverse=True,
        )[:10],
        "_plot": summaries,
    }


def rope_alignment_summary(
    root: Path,
    existing_points_path: Path,
) -> dict[str, Any]:
    band_names = (
        "high_0_15",
        "mid_16_31",
        "low_32_47",
        "ultralow_48_63",
    )
    legacy_rows = read_rows(existing_points_path)
    legacy_by_total = {
        int(row["total_tokens"]): row for row in legacy_rows
    }
    current_rows = read_rows(
        root / "period_with_distractors" / "points.csv"
    )
    current_by_total = {
        int(row["total_tokens"]): row for row in current_rows
    }
    rope_rows = read_rows(
        root / "q_rope_probe" / "fixed_rope_curve.csv"
    )
    rope_totals = np.array(
        [int(row["total_tokens"]) for row in rope_rows],
        dtype=np.int64,
    )
    fixed_all = np.array(
        [float(row["fixed_q_rope_qk"]) for row in rope_rows]
    )
    bands_all = np.column_stack(
        [
            np.array([float(row[name]) for row in rope_rows])
            for name in band_names
        ]
    )
    rope_index = {
        int(total): index
        for index, total in enumerate(rope_totals)
    }

    probe_rows = read_rows(
        root / "q_rope_probe" / "probe_points.csv"
    )
    probe_totals = np.array(
        [int(row["total_tokens"]) for row in probe_rows],
        dtype=np.int64,
    )
    probe_indices = np.array(
        [rope_index[int(total)] for total in probe_totals]
    )
    fixed_probe = fixed_all[probe_indices]
    bands_probe = bands_all[probe_indices]
    actual_qk = np.array(
        [
            float(row["critical_qk_actual"])
            for row in probe_rows
        ]
    )
    probe_nine_probability = np.array(
        [
            float(row["gold_probability"])
            for row in probe_rows
        ]
    )
    probe_output_margin = np.array(
        [
            float(row["gold_vs_competitor_margin"])
            for row in probe_rows
        ]
    )
    fixed_to_actual = regression(
        fixed_probe.reshape(-1, 1),
        actual_qk,
    )
    bands_to_qk = regression(bands_probe, actual_qk)
    qk_to_margin = regression(
        actual_qk.reshape(-1, 1),
        probe_output_margin,
    )

    current_totals = np.array(
        sorted(current_by_total),
        dtype=np.int64,
    )
    current_indices = np.array(
        [rope_index[int(total)] for total in current_totals]
    )
    fixed_current = fixed_all[current_indices]
    bands_current = bands_all[current_indices]
    current_nine_probability = np.array(
        [
            float(
                current_by_total[int(total)][
                    "gold_exact_probability"
                ]
            )
            for total in current_totals
        ]
    )
    current_output_margin = np.array(
        [
            float(
                current_by_total[int(total)][
                    "gold_exact_vs_competitor_margin"
                ]
            )
            for total in current_totals
        ]
    )
    bands_to_margin = regression(
        bands_current,
        current_output_margin,
    )
    bands_to_probability = regression(
        bands_current,
        current_nine_probability,
    )

    legacy_actual_qk = np.array(
        [
            float(
                legacy_by_total[int(total)][
                    "critical_qk_weighted"
                ]
            )
            for total in rope_totals
        ]
    )
    legacy_nine_probability = np.array(
        [
            float(
                legacy_by_total[int(total)][
                    "nine_probability"
                ]
            )
            for total in rope_totals
        ]
    )
    legacy_output_margin = np.array(
        [
            float(
                legacy_by_total[int(total)][
                    "full_vocab_margin"
                ]
            )
            for total in rope_totals
        ]
    )
    legacy_bands_to_margin = regression(
        bands_all,
        legacy_output_margin,
    )
    probe_by_total = {
        int(row["total_tokens"]): row for row in probe_rows
    }
    anchors = {}
    for total in (136 * 1024, 140 * 1024, 142 * 1024, 144 * 1024):
        row = probe_by_total[total]
        anchor = {
            key: float(row[key])
            for key in (
                "critical_qk_actual",
                "critical_qk_position_only",
                "critical_qk_content_only",
                "critical_qk_baseline",
                "critical_qk_interaction",
                "gold_probability",
                "gold_vs_competitor_margin",
            )
        }
        anchor["position_delta_from_136k"] = (
            anchor["critical_qk_position_only"]
            - anchor["critical_qk_baseline"]
        )
        anchor["content_delta_from_136k"] = (
            anchor["critical_qk_content_only"]
            - anchor["critical_qk_baseline"]
        )
        anchor["actual_delta_from_136k"] = (
            anchor["critical_qk_actual"]
            - anchor["critical_qk_baseline"]
        )
        anchor["reconstructed_delta_from_136k"] = (
            anchor["position_delta_from_136k"]
            + anchor["content_delta_from_136k"]
            + anchor["critical_qk_interaction"]
        )
        anchor["top_token_label"] = row["top_token_label"]
        anchors[str(total)] = anchor

    windows = {}
    for label, lower, upper, include_upper in (
        ("136k_to_140k", 136 * 1024, 140 * 1024, False),
        ("140k_to_141k", 140 * 1024, 141 * 1024, False),
        ("141k_to_142k", 141 * 1024, 142 * 1024, False),
        ("142k_to_144k", 142 * 1024, 144 * 1024, True),
    ):
        selected = [
            row
            for row in probe_rows
            if int(row["total_tokens"]) >= lower
            and (
                int(row["total_tokens"]) <= upper
                if include_upper
                else int(row["total_tokens"]) < upper
            )
        ]
        window = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "critical_qk_actual",
                "critical_qk_position_only",
                "critical_qk_content_only",
                "critical_qk_interaction",
                "gold_probability",
                "gold_vs_competitor_margin",
            )
        }
        window["point_count"] = len(selected)
        windows[label] = window

    band_correlations = {}
    for index, name in enumerate(band_names):
        band_correlations[name] = {
            "with_actual_qk": correlation(
                bands_probe[:, index],
                actual_qk,
            ),
            "with_current_output_margin": correlation(
                bands_current[:, index],
                current_output_margin,
            ),
            "with_current_nine_probability": correlation(
                bands_current[:, index],
                current_nine_probability,
            ),
        }
    return {
        "probe_point_count": len(probe_totals),
        "current_output_point_count": len(current_totals),
        "fixed_rope_vs_actual_qk_correlation": correlation(
            fixed_probe,
            actual_qk,
        ),
        "fixed_rope_to_actual_qk_r2": fixed_to_actual["r2"],
        "bands_to_actual_qk_r2": bands_to_qk["r2"],
        "bands_to_output_margin_r2": bands_to_margin["r2"],
        "bands_to_nine_probability_r2": bands_to_probability["r2"],
        "actual_qk_to_output_margin_r2": qk_to_margin["r2"],
        "actual_qk_vs_nine_probability_correlation": correlation(
            actual_qk,
            probe_nine_probability,
        ),
        "fixed_rope_vs_current_nine_probability_correlation": (
            correlation(
                fixed_current,
                current_nine_probability,
            )
        ),
        "fixed_rope_vs_current_output_margin_correlation": (
            correlation(
                fixed_current,
                current_output_margin,
            )
        ),
        "legacy_reference": {
            "execution_note": (
                "Legacy tokenwise trace may use a different GPU "
                "partition; retained only as a reference."
            ),
            "fixed_rope_vs_actual_qk_correlation": correlation(
                fixed_all,
                legacy_actual_qk,
            ),
            "actual_qk_vs_nine_probability_correlation": (
                correlation(
                    legacy_actual_qk,
                    legacy_nine_probability,
                )
            ),
            "bands_to_output_margin_r2": (
                legacy_bands_to_margin["r2"]
            ),
        },
        "band_coefficients_for_output_margin": {
            name: coefficient
            for name, coefficient in zip(
                ("intercept", *band_names),
                bands_to_margin["coefficients"],
            )
        },
        "band_correlations": band_correlations,
        "anchor_decomposition": anchors,
        "window_decomposition": windows,
        "_plot": {
            "totals": current_totals,
            "fixed": fixed_current,
            "bands": bands_current,
            "probe_totals": probe_totals,
            "actual_qk": actual_qk,
            "nine_probability": current_nine_probability,
            "output_margin": current_output_margin,
            "band_margin_prediction": bands_to_margin["prediction"],
            "probe_rows": probe_rows,
        },
    }


def plot_results(
    root: Path,
    punctuation: dict[str, Any],
    q_mean: dict[str, Any],
    rope: dict[str, Any],
    head_decomposition: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "period_with_distractors": "period",
        "comma_with_distractors": "comma",
        "question_with_distractors": "question",
        "semicolon_with_distractors": "semicolon",
    }
    global_competitor_counts: collections.Counter[str] = (
        collections.Counter()
    )
    for value in punctuation.values():
        global_competitor_counts.update(value["competitor_counts"])
    all_competitors = [
        key
        for key, _ in global_competitor_counts.most_common(8)
    ]
    x = np.arange(len(PUNCTUATION_VARIANTS))
    bottom = np.zeros(len(x))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for key in all_competitors:
        shares = np.array(
            [
                punctuation[name]["competitor_shares"].get(key, 0.0)
                for name in PUNCTUATION_VARIANTS
            ]
        )
        ax.bar(
            x,
            shares,
            bottom=bottom,
            label=token_name(key)
            .replace("␠", "[space]")
            .replace("↵", "\\n"),
        )
        bottom += shares
    other_shares = np.array(
        [
            1.0
            - sum(
                punctuation[name]["competitor_shares"].get(
                    key,
                    0.0,
                )
                for key in all_competitors
            )
            for name in PUNCTUATION_VARIANTS
        ]
    )
    if np.any(other_shares > 1e-12):
        ax.bar(
            x,
            other_shares,
            bottom=bottom,
            label="other",
        )
    ax.set_xticks(
        x,
        [labels[name] for name in PUNCTUATION_VARIANTS],
    )
    ax.set_ylabel("Share as strongest competitor")
    ax.set_ylim(0, 1)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    fig.savefig(
        analysis_dir / "punctuation_competitor_shares.png",
        dpi=180,
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )
    for name in PUNCTUATION_VARIANTS:
        rows = read_rows(root / name / "points.csv")
        length_k = np.array(
            [int(row["total_tokens"]) / 1024 for row in rows]
        )
        axes[0].plot(
            length_k,
            [
                100 * float(row["gold_exact_probability"])
                for row in rows
            ],
            linewidth=0.9,
            label=labels[name],
        )
        axes[1].plot(
            length_k,
            [
                float(row["gold_exact_vs_competitor_margin"])
                for row in rows
            ],
            linewidth=0.9,
            label=labels[name],
        )
    axes[0].set_ylabel("P(nine) (%)")
    axes[0].legend(frameon=False, ncol=4)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Total length (Ki tokens)")
    axes[1].set_ylabel("Gold vs strongest competitor margin")
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(
        analysis_dir / "punctuation_gold_probability_curves.png",
        dpi=180,
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )
    for name, label in (
        ("period_with_distractors", "with distractors"),
        ("period_without_distractors", "without distractors"),
    ):
        rows = read_rows(root / name / "points.csv")
        length_k = np.array(
            [int(row["total_tokens"]) / 1024 for row in rows]
        )
        axes[0].plot(
            length_k,
            [
                100 * float(row["gold_exact_probability"])
                for row in rows
            ],
            linewidth=0.8,
            label=label,
        )
        axes[1].plot(
            length_k,
            [
                float(row["gold_exact_vs_competitor_margin"])
                for row in rows
            ],
            linewidth=0.8,
            label=label,
        )
    axes[0].set_ylabel("P(nine) (%)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Total length (Ki tokens)")
    axes[1].set_ylabel("Gold vs strongest competitor margin")
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(
        analysis_dir / "no_distractor_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    curve_values = list(q_mean["curve"].values())
    curve_x = [value["kib_tokens"] for value in curve_values]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(
        curve_x,
        [value["q_global_cosine_mean"] for value in curve_values],
        linewidth=1.0,
        label="Q vs all-prefix Q mean",
    )
    ax.plot(
        curve_x,
        [value["q_period_cosine_mean"] for value in curve_values],
        linewidth=1.0,
        label="Q vs period-token Q mean",
    )
    ax.set_xlabel("Total length (Ki tokens)")
    ax.set_ylabel("Mean cosine across layer-heads")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(
        analysis_dir / "q_mean_cosine_136k_144k.png",
        dpi=180,
    )
    plt.close(fig)

    plot_data = rope.pop("_plot")
    totals_k = plot_data["totals"] / 1024
    bands = plot_data["bands"]
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        totals_k,
        100 * plot_data["nine_probability"],
        linewidth=0.8,
        label="P(nine)",
    )
    axes[0].set_ylabel("P(nine) (%)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(
        totals_k,
        plot_data["output_margin"],
        linewidth=0.8,
        label="actual output margin",
    )
    axes[1].plot(
        totals_k,
        plot_data["band_margin_prediction"],
        linewidth=1.0,
        alpha=0.85,
        label="4-band linear fit",
    )
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].set_ylabel("Output margin")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    axes[2].plot(
        totals_k,
        plot_data["fixed"],
        linewidth=1.0,
        label="fixed-Q RoPE total",
    )
    for index, name in enumerate(
        ("high", "mid", "low", "ultralow")
    ):
        axes[2].plot(
            totals_k,
            bands[:, index],
            linewidth=0.8,
            label=name,
        )
    axes[2].plot(
        plot_data["probe_totals"] / 1024,
        plot_data["actual_qk"],
        linewidth=0.8,
        alpha=0.8,
        label="actual weighted QK",
    )
    axes[2].set_xlabel("Total length (Ki tokens)")
    axes[2].set_ylabel("Weighted QK contribution")
    axes[2].legend(frameon=False, ncol=3)
    axes[2].grid(axis="y", alpha=0.2)
    fig.savefig(
        analysis_dir / "rope_band_alignment.png",
        dpi=180,
    )
    plt.close(fig)

    probe_rows = plot_data["probe_rows"]
    probe_x = np.array(
        [int(row["total_tokens"]) / 1024 for row in probe_rows]
    )
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for field, label in (
        ("critical_qk_actual", "actual"),
        ("critical_qk_position_only", "position-only"),
        ("critical_qk_content_only", "content-only"),
        ("critical_qk_interaction", "interaction"),
    ):
        ax.plot(
            probe_x,
            [float(row[field]) for row in probe_rows],
            linewidth=1.0,
            label=label,
        )
    ax.set_xlabel("Total length (Ki tokens)")
    ax.set_ylabel("Weighted critical-head QK")
    ax.legend(frameon=False, ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(
        analysis_dir / "qk_position_content_decomposition.png",
        dpi=180,
    )
    plt.close(fig)

    head_rows = head_decomposition.pop("_plot")
    ordered = sorted(
        head_rows,
        key=lambda row: row["drop_early_to_140_141"],
    )
    labels = [
        f"L{row['layer']}H{row['head']}"
        for row in ordered
    ]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.bar(
        x - 0.2,
        [row["drop_early_to_140_141"] for row in ordered],
        width=0.4,
        label="136–140K → 140–141K",
    )
    ax.bar(
        x + 0.2,
        [
            row["recovery_140_141_to_142_144"]
            for row in ordered
        ],
        width=0.4,
        label="140–141K → 142–144K",
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x, labels, rotation=70)
    ax.set_ylabel("Change in weighted QK contribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(
        analysis_dir / "qk_head_drop_recovery.png",
        dpi=180,
    )
    plt.close(fig)


def write_report(
    path: Path,
    punctuation: dict[str, Any],
    generation: dict[str, Any],
    q_mean: dict[str, Any],
    no_distractor: dict[str, Any],
    rope: dict[str, Any],
    head_decomposition: dict[str, Any],
) -> None:
    lines = [
        "# 单跳长上下文夜间实验汇总",
        "",
        "## 1. 无关填充标点与最强竞争 token",
        "",
        "| 填充 | 失败率 | 平均 P(nine) | 平均 margin | 首次失败 | 512-token 窗口多数失败 | 最强非 gold token | 占比 | 失败时最常见赢家 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for name in PUNCTUATION_VARIANTS:
        row = punctuation[name]
        dominant = row["dominant_competitor"]
        share = (
            row["competitor_shares"].get(dominant, 0.0)
            if dominant is not None
            else 0.0
        )
        failure_winner = (
            max(
                row["failure_winner_counts"],
                key=row["failure_winner_counts"].get,
            )
            if row["failure_winner_counts"]
            else ""
        )
        lines.append(
            f"| `{row['filler_text']}` | "
            f"{100 * row['failure_rate']:.2f}% | "
            f"{100 * row['mean_gold_probability']:.2f}% | "
            f"{row['mean_output_margin']:.3f} | "
            f"{format_total_kib(row['first_failure_total_tokens'])} | "
            f"{format_total_kib(row['first_512_token_window_majority_failure_end'])} | "
            f"`{token_name(dominant or '')}` | "
            f"{100 * share:.2f}% | "
            f"`{failure_winner}` |"
        )
    lines.extend(
        [
            "",
            "这里“最强竞争 token”统计每个长度点中概率最高的非 gold "
            "token，不要求该点已经预测失败；真正获胜的错误 token "
            "另存于 `summary.json` 的 `failure_winner_counts`。",
            "",
            "四组在同一总长度下，开头的真实证据位置、真实 `nine` Key "
            "以及证据到查询的相对距离完全相同。因果模型中未来 filler "
            "也不能反向改变开头证据的 Key。因此组间差异不能归因于"
            "证据 Key 或 RoPE 距离本身，只能来自 filler 改变了后续 "
            "K/V 竞争集合，并通过深层注意力改变了最终查询的 pre-RoPE Q。"
            "如果问号本身成为输出赢家，则还表明存在明显的复制/续写先验。",
            "",
            "| 填充 | 136K P(nine) | 140K | 142K | 144K |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in PUNCTUATION_VARIANTS:
        row = punctuation[name]
        lines.append(
            f"| `{row['filler_text']}` | "
            + " | ".join(
                f"{100 * row['anchor_metrics'][str(total)]['gold_probability']:.2f}%"
                for total in (
                    136 * 1024,
                    140 * 1024,
                    142 * 1024,
                    144 * 1024,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 2. RoPE 频带能否解释概率波动",
            "",
            f"- 固定 Q 的纯 RoPE 曲线与实际关键 QK 的相关系数："
            f"**{rope['fixed_rope_vs_actual_qk_correlation']:.3f}**。",
            f"- 单条固定-Q RoPE 总曲线拟合实际 QK 的 "
            f"R²：**{rope['fixed_rope_to_actual_qk_r2']:.3f}**。",
            f"- 四个 RoPE 频带联合拟合实际 QK 的 "
            f"R²：**{rope['bands_to_actual_qk_r2']:.3f}**。",
            f"- 四个频带拟合最终输出 margin 的 "
            f"R²：**{rope['bands_to_output_margin_r2']:.3f}**。",
            f"- 四个频带拟合 `P(nine)` 的 "
            f"R²：**{rope['bands_to_nine_probability_r2']:.3f}**。",
            f"- 实际关键 QK 单独解释输出 margin 的 "
            f"R²：**{rope['actual_qk_to_output_margin_r2']:.3f}**。",
            "",
            "## 3. 首 token 错误后，多生成能否恢复 nine",
            "",
            "抽样位置覆盖旧逐-token轨迹中的正确区、换行失败区和其他"
            "竞争者失败区；每个位置的首 token 与后续生成均在本次同一"
            "模型、同一 GPU 配置上重新计算。",
            f"- 抽查 {generation['selected_count']} 个位置，其中首 token "
            f"错误 {generation['wrong_first_token_count']} 个。",
            f"- 首 token 错误后，后续 32 token 中出现 `nine` 或数字 `9`："
            f"**{generation['wrong_first_then_contains_nine']} / "
            f"{generation['wrong_first_token_count']}** 个。",
            f"- 首个明确年龄值为 `nine`/`9`："
            f"**{generation['wrong_first_then_first_number_nine']} / "
            f"{generation['wrong_first_token_count']}** 个。",
            f"- 全部抽样点按“生成中的首个年龄值”计算，语义准确率："
            f"**{100 * generation['multi_token_first_number_accuracy']:.2f}%**。",
            "",
            "| 首 token | 样本数 | 后续出现 nine/9 | 首个年龄为 nine/9 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, row in generation["by_first_token"].items():
        lines.append(
            f"| `{label}` | {row['count']} | "
            f"{row['contains_nine']} | "
            f"{row['first_number_nine']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Query Q 是否趋近均值",
            "",
        ]
    )
    for total, row in q_mean["anchors"].items():
        lines.append(
            f"- {row['kib_tokens']:.0f}K："
            f"cos(Q, 全前缀 Q 均值)="
            f"{row['q_global_cosine_mean']:.4f}；"
            f"cos(Q, 句号 Q 均值)="
            f"{row['q_period_cosine_mean']:.4f}。"
        )
    q_change = q_mean["change_136k_to_144k"]
    approach = q_mean["headwise_approach_fraction_136k_to_144k"]
    lines.extend(
        [
            "",
            f"- 136K→144K，全前缀均值 cosine 的均值变化："
            f"**{q_change['q_global_cosine_mean']:+.4f}**；"
            f"句号均值 cosine 的均值变化："
            f"**{q_change['q_period_cosine_mean']:+.4f}**。",
            f"- 同时满足“cosine 上升且相对 L2 距离下降”的 head："
            f"全前缀均值 **{100 * approach['global_both']:.1f}%**，"
            f"句号均值 **{100 * approach['period_both']:.1f}%**。",
            "- 只有 cosine 上升且相对 L2 距离下降，才记为该 head "
            "向相应均值靠近；不能仅凭单个总体均值判断。",
            "",
            "## 5. 移除年龄干扰句",
            "",
            "本对照保持总 token 长度不变。删除年龄干扰句后，腾出的"
            "位置由同一种句号 filler 补齐，因此它严格比较的是"
            "“年龄干扰句 + 句号”与“更高比例的重复句号”，而不是把"
            "干扰句直接删除后缩短上下文。若无干扰组更差，不能直接"
            "推论年龄干扰句在语义上有益；更可能说明极端重复标点本身"
            "造成了复制/格式退化。",
            f"- 有干扰（stride=8）失败率："
            f"**{100 * no_distractor['with_distractors_stride8']['failure_rate']:.2f}%**。",
            f"- 无干扰、同样 stride=8 下采样的失败率："
            f"**{100 * no_distractor['without_distractors_stride8_matched']['failure_rate']:.2f}%**。",
            f"- 无干扰全量逐 token 失败率："
            f"**{100 * no_distractor['without_distractors']['failure_rate']:.2f}%**"
            "（用于精确寻找边界，不直接与 stride=8 的比例混比）。",
            f"- 有干扰首次失败："
            f"**{format_total_kib(no_distractor['with_distractors_stride8']['first_failure_total_tokens'])}**；"
            f"无干扰首次失败："
            f"**{format_total_kib(no_distractor['without_distractors']['first_failure_total_tokens'])}**。",
            f"- 有干扰 512-token 窗口多数失败边界："
            f"**{format_total_kib(no_distractor['with_distractors_stride8']['first_512_token_window_majority_failure_end'])}**；"
            f"无干扰对应边界："
            f"**{format_total_kib(no_distractor['without_distractors']['first_512_token_window_majority_failure_end'])}**。",
            f"- 有干扰最常见竞争者："
            f"`{token_name(max(no_distractor['with_distractors_stride8']['competitor_counts'], key=no_distractor['with_distractors_stride8']['competitor_counts'].get))}`；"
            f"无干扰最常见竞争者："
            f"`{token_name(max(no_distractor['without_distractors']['competitor_counts'], key=no_distractor['without_distractors']['competitor_counts'].get))}`。",
            "",
            "## 6. 140K 下降与 142K 恢复的机制分解",
            "",
        ]
    )
    for total in (136 * 1024, 140 * 1024, 142 * 1024, 144 * 1024):
        row = rope["anchor_decomposition"][str(total)]
        lines.append(
            f"- {total / 1024:.0f}K：实际 QK "
            f"{row['critical_qk_actual']:.3f}；"
            f"相对 136K 的位置项 "
            f"{row['position_delta_from_136k']:+.3f}；"
            f"Q 内容项 {row['content_delta_from_136k']:+.3f}；"
            f"交互项 {row['critical_qk_interaction']:+.3f}；"
            f"首 token `{row['top_token_label']}`。"
        )
    lines.extend(
        [
            "",
            "分段均值（避免单个锚点恰好落在局部峰值）：",
            "",
            "| 区间 | 实际 QK | 仅位置 | 仅 Q 内容 | 交互项 | P(nine) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in rope["window_decomposition"].items():
        lines.append(
            f"| {label.replace('_', ' ')} | "
            f"{row['critical_qk_actual']:.3f} | "
            f"{row['critical_qk_position_only']:.3f} | "
            f"{row['critical_qk_content_only']:.3f} | "
            f"{row['critical_qk_interaction']:+.3f} | "
            f"{100 * row['gold_probability']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "对聚合下降贡献最大的关键 head：",
            "",
        ]
    )
    for row in head_decomposition["largest_drop_heads"][:5]:
        lines.append(
            f"- L{row['layer']}H{row['head']}："
            f"加权 QK 下降 "
            f"{row['drop_early_to_140_141']:+.4f}；"
            f"在 140–141K 窗口中，位置项 "
            f"{row['failure_window_position_delta']:+.4f}，"
            f"Q 内容项 {row['failure_window_content_delta']:+.4f}，"
            f"交互项 {row['failure_window_interaction']:+.4f}。"
        )
    lines.extend(
        [
            "",
            "恢复贡献最大的关键 head：",
            "",
        ]
    )
    for row in head_decomposition["largest_recovery_heads"][:5]:
        lines.append(
            f"- L{row['layer']}H{row['head']}："
            f"加权 QK 恢复 "
            f"{row['recovery_140_141_to_142_144']:+.4f}。"
        )
    lines.extend(
        [
            "",
            "数学上，对固定证据 Key，有",
            "",
            "$$",
            "s(p)=\\frac{\\langle R_p q, k_g\\rangle}{\\sqrt d}",
            "$$",
            "",
            "这里 $q$ 是最后一个查询 token 的 pre-RoPE Query，"
            "$R_p$ 是查询位置 $p$ 对应的 RoPE 旋转，"
            "$k_g$ 是缓存中真实 `nine` 证据的 post-RoPE Key。",
            "",
            "把 136K 的 Query 记为 $q_0$，当前位置 Query 记为 "
            "$q_p$，则总变化可以拆成：",
            "",
            "$$",
            "s(q_p,p)-s(q_0,p_0)="
            "\\Delta_{\\mathrm{position}}+"
            "\\Delta_{\\mathrm{content}}+"
            "\\Delta_{\\mathrm{interaction}}.",
            "$$",
            "",
            "其中",
            "",
            "$$",
            "\\Delta_{\\mathrm{position}}="
            "s(q_0,p)-s(q_0,p_0),",
            "$$",
            "",
            "$$",
            "\\Delta_{\\mathrm{content}}="
            "s(q_p,p_0)-s(q_0,p_0),",
            "$$",
            "",
            "$$",
            "\\Delta_{\\mathrm{interaction}}="
            "s(q_p,p)-s(q_0,p)-s(q_p,p_0)+s(q_0,p_0).",
            "$$",
            "",
            "因此，140K 的骤降是否主要来自 RoPE 相位、Query 内容漂移，"
            "还是两者相互作用，可以直接由上面的三项数值判定；"
            "142K 的恢复也使用同一口径验证，而不是仅凭曲线形状推测。",
            "",
            "## 图表",
            "",
            "- `punctuation_competitor_shares.png`",
            "- `punctuation_gold_probability_curves.png`",
            "- `no_distractor_comparison.png`",
            "- `rope_band_alignment.png`",
            "- `q_mean_cosine_136k_144k.png`",
            "- `qk_position_content_decomposition.png`",
            "- `qk_head_drop_recovery.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    punctuation = punctuation_summary(root)
    generation = generation_summary(root)
    no_distractor = no_distractor_summary(root)
    q_mean = q_mean_summary(root)
    head_decomposition = head_decomposition_summary(root)
    rope = rope_alignment_summary(
        root,
        Path(args.existing_points_csv),
    )
    plot_results(
        root,
        punctuation,
        q_mean,
        rope,
        head_decomposition,
    )
    summary = {
        "schema_version": 1,
        "punctuation": punctuation,
        "generation": {
            key: value
            for key, value in generation.items()
            if key != "rows"
        },
        "q_mean": q_mean,
        "no_distractor": no_distractor,
        "rope_alignment_and_qk_decomposition": rope,
        "critical_head_decomposition": head_decomposition,
    }
    write_json(analysis_dir / "summary.json", summary)
    write_report(
        analysis_dir / "report.md",
        punctuation,
        generation,
        q_mean,
        no_distractor,
        rope,
        head_decomposition,
    )
    print(
        json.dumps(
            {
                "punctuation": {
                    name: value["dominant_competitor"]
                    for name, value in punctuation.items()
                },
                "generation_wrong_then_nine": generation[
                    "wrong_first_then_contains_nine"
                ],
                "no_distractor_failure_rate": no_distractor[
                    "without_distractors"
                ]["failure_rate"],
                "rope_fixed_correlation": rope[
                    "fixed_rope_vs_actual_qk_correlation"
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
