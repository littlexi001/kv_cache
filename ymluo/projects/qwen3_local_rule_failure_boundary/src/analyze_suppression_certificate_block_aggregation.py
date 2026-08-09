from __future__ import annotations

"""CPU-only block/line aggregation for suppression-certificate samples.

The safety runner reports one row per sampled token, layer, and query head.
This analyzer first collapses layer/head measurements at a fixed token, then
asks whether that representation separates gold evidence from conflict or all
non-gold tokens. It also aggregates all sampled rows belonging to an evidence
line and performs a paired gold-line versus conflict-line comparison within
each length/seed case.

No model is loaded and no server is contacted. The optional LOSO combination
uses fixed equal weights; means and standard deviations are fitted on training
seeds only. It is disabled by default.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_METRICS = (
    "pre_score",
    "post_score",
    "pre_suppression",
    "grid_envelope_suppression",
)
REDUCERS = ("mean", "max", "q90", "positive_fraction")
SCOPES = ("all_sampled", "decisive_only")
CONTRASTS = {
    "gold_vs_conflict": (
        ("gold_evidence",),
        ("conflict_evidence",),
    ),
    "gold_vs_all_nongold": (
        ("gold_evidence",),
        (
            "conflict_evidence",
            "lexical_format_distractor",
            "filler",
        ),
    ),
}
IDENTITY_FIELDS = (
    "target_context_tokens",
    "seed",
    "layer",
    "head",
    "class",
    "token_position",
)


def rounded(value: float, digits: int = 10) -> float:
    return float(f"{float(value):.{digits}g}")


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile needs at least one value")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("quantile probability must be in [0,1]")
    ordered = sorted(map(float, values))
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_values(
    values: Sequence[float], positive_threshold: float
) -> dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate an empty score collection")
    numeric = list(map(float, values))
    return {
        "mean": statistics.fmean(numeric),
        "max": max(numeric),
        "q90": quantile(numeric, 0.90),
        "positive_fraction": sum(
            value > float(positive_threshold) for value in numeric
        )
        / len(numeric),
    }


def binary_auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Tie-aware AUROC, equal to a normalized Mann--Whitney statistic."""

    if not positives or not negatives:
        return float("nan")
    ranked = sorted(
        [(float(value), 1) for value in positives]
        + [(float(value), 0) for value in negatives]
    )
    wins = 0.0
    negatives_before = 0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        tied = ranked[index:end]
        tied_positives = sum(label == 1 for _, label in tied)
        tied_negatives = len(tied) - tied_positives
        wins += tied_positives * (
            negatives_before + 0.5 * tied_negatives
        )
        negatives_before += tied_negatives
        index = end
    return wins / (len(positives) * len(negatives))


def _finite_or_na(value: float) -> float | str:
    return rounded(value) if math.isfinite(value) else "NA"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _metadata_source_paths(metadata_path: Path) -> list[Path]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_sources: list[str] = []
    for field in ("source_dirs", "resolved_sources"):
        values = payload.get(field, [])
        if isinstance(values, list):
            raw_sources.extend(str(value) for value in values)
    paths: list[Path] = []
    for raw in raw_sources:
        candidate = Path(raw).expanduser()
        alternatives = [candidate]
        if not candidate.is_absolute():
            alternatives.extend(
                (
                    metadata_path.parent / candidate,
                    Path.cwd() / candidate,
                )
            )
        resolved = next((path for path in alternatives if path.exists()), None)
        if resolved is not None:
            paths.append(resolved)
    return paths


def discover_sample_files(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve files, shard directories, or merged outputs recursively."""

    found: set[Path] = set()
    visited: set[Path] = set()

    def visit(raw_path: str | Path) -> None:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"input does not exist: {path}")
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        if resolved.is_file():
            if not resolved.name.endswith("_certificate_samples.jsonl"):
                raise ValueError(
                    "sample file must end with _certificate_samples.jsonl: "
                    f"{resolved}"
                )
            found.add(resolved)
            return
        for sample_path in resolved.rglob("*_certificate_samples.jsonl"):
            found.add(sample_path.resolve())
        for metadata_name in ("summary.json", "merge_config.json"):
            metadata_path = resolved / metadata_name
            if metadata_path.exists():
                for source in _metadata_source_paths(metadata_path):
                    visit(source)

    for input_path in inputs:
        visit(input_path)
    if not found:
        raise FileNotFoundError(
            "no *_certificate_samples.jsonl files found in the supplied inputs"
        )
    return sorted(found, key=str)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def _matching_result_path(sample_path: Path) -> Path:
    suffix = "_certificate_samples.jsonl"
    stem = sample_path.name[: -len(suffix)]
    return sample_path.with_name(f"{stem}_result.json")


def _load_case_records(sample_path: Path) -> tuple[list[dict[str, Any]], str]:
    result_path = _matching_result_path(sample_path)
    if not result_path.exists():
        return [], "class_fallback_no_result_json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid result JSON: {result_path}") from error
    records = result.get("case", {}).get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"case.records is not a list in {result_path}")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        span = record.get("span", [])
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(
                f"record {index} has invalid span in {result_path}"
            )
        normalized.append(
            {
                "record_index": int(index),
                "category": str(record.get("category", "")),
                "text": str(record.get("text", "")),
                "span_start": int(span[0]),
                "span_end": int(span[1]),
            }
        )
    return normalized, "result_json_record_span"


def _line_metadata(
    row: dict[str, Any],
    records: Sequence[dict[str, Any]],
    metadata_source: str,
) -> dict[str, Any]:
    position = int(row["token_position"])
    category = str(row["class"])
    matches = [
        record
        for record in records
        if int(record["span_start"]) <= position < int(record["span_end"])
    ]
    if len(matches) > 1:
        raise ValueError(
            f"token position {position} belongs to multiple evidence records"
        )
    if matches:
        record = matches[0]
        if str(record["category"]) != category:
            raise ValueError(
                f"token class {category} conflicts with record class "
                f"{record['category']} at position {position}"
            )
        return {
            "line_id": (
                f"record_{int(record['record_index']):03d}_"
                f"{int(record['span_start'])}_{int(record['span_end'])}"
            ),
            "line_span_start": int(record["span_start"]),
            "line_span_end": int(record["span_end"]),
            "line_text": str(record["text"]).replace("\n", "\\n"),
            "line_metadata_source": metadata_source,
        }
    return {
        "line_id": f"class_fallback:{category}",
        "line_span_start": "",
        "line_span_end": "",
        "line_text": "",
        "line_metadata_source": (
            metadata_source
            if not records
            else "class_fallback_position_outside_record_spans"
        ),
    }


def load_samples(
    sample_files: Sequence[Path], metrics: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, validate, line-annotate, and de-duplicate sample rows."""

    rows_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicate_count = 0
    metadata_source_counts: dict[str, int] = defaultdict(int)
    for sample_path in sample_files:
        records, metadata_source = _load_case_records(sample_path)
        for raw in _read_jsonl(sample_path):
            missing = [
                field
                for field in (*IDENTITY_FIELDS, "is_decisive_token", *metrics)
                if field not in raw
            ]
            if missing:
                raise ValueError(
                    f"{sample_path} row is missing required fields: {missing}"
                )
            for metric in metrics:
                value = float(raw[metric])
                if not math.isfinite(value):
                    raise ValueError(
                        f"{sample_path} contains non-finite {metric}: {value}"
                    )
            identity = tuple(raw[field] for field in IDENTITY_FIELDS)
            if identity in rows_by_identity:
                existing = rows_by_identity[identity]
                if any(existing.get(field) != value for field, value in raw.items()):
                    raise ValueError(
                        "conflicting duplicate certificate row for identity "
                        f"{identity}"
                    )
                duplicate_count += 1
                continue
            enriched = dict(raw)
            line = _line_metadata(raw, records, metadata_source)
            enriched.update(line)
            enriched["source_sample_file"] = str(sample_path)
            rows_by_identity[identity] = enriched
            metadata_source_counts[str(line["line_metadata_source"])] += 1
    rows = sorted(
        rows_by_identity.values(),
        key=lambda row: (
            int(row["target_context_tokens"]),
            int(row["seed"]),
            str(row["class"]),
            int(row["token_position"]),
            int(row["layer"]),
            int(row["head"]),
        ),
    )
    return rows, {
        "deduplicated_row_count": len(rows),
        "duplicate_row_count": duplicate_count,
        "line_metadata_source_counts": dict(
            sorted(metadata_source_counts.items())
        ),
    }


def _scope_rows(
    samples: Sequence[dict[str, Any]], scope: str
) -> Iterable[dict[str, Any]]:
    if scope == "all_sampled":
        return samples
    if scope == "decisive_only":
        return (
            row
            for row in samples
            if int(row.get("is_decisive_token", 0)) == 1
        )
    raise ValueError(f"unknown scope: {scope}")


def token_aggregates(
    samples: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    *,
    positive_threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope in SCOPES:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in _scope_rows(samples, scope):
            key = (
                int(row["target_context_tokens"]),
                int(row["seed"]),
                str(row["class"]),
                int(row["token_position"]),
            )
            groups[key].append(row)
        for (length, seed, category, position), rows in sorted(groups.items()):
            line_ids = {str(row["line_id"]) for row in rows}
            if len(line_ids) != 1:
                raise ValueError("one token position maps to multiple line IDs")
            result: dict[str, Any] = {
                "target_context_tokens": length,
                "seed": seed,
                "scope": scope,
                "class": category,
                "token_position": position,
                "relative_distance": int(rows[0].get("relative_distance", -1)),
                "sample_index": int(rows[0].get("sample_index", -1)),
                "is_decisive_token": int(
                    rows[0].get("is_decisive_token", 0)
                ),
                "line_id": rows[0]["line_id"],
                "line_span_start": rows[0]["line_span_start"],
                "line_span_end": rows[0]["line_span_end"],
                "line_text": rows[0]["line_text"],
                "line_metadata_source": rows[0]["line_metadata_source"],
                "layer_count": len({int(row["layer"]) for row in rows}),
                "head_count": len({int(row["head"]) for row in rows}),
                "layer_head_count": len(rows),
            }
            for metric in metrics:
                reduced = aggregate_values(
                    [float(row[metric]) for row in rows],
                    positive_threshold,
                )
                for reducer, score in reduced.items():
                    result[f"{metric}__{reducer}"] = rounded(score)
            output.append(result)
    return output


def _auc_row(
    rows: Sequence[dict[str, Any]],
    *,
    evaluation_level: str,
    length: int | str,
    seed: int | str,
    scope: str,
    contrast: str,
    positive_classes: Sequence[str],
    negative_classes: Sequence[str],
    metric: str,
    reducer: str,
) -> dict[str, Any]:
    field = f"{metric}__{reducer}"
    positives = [
        float(row[field]) for row in rows if row["class"] in positive_classes
    ]
    negatives = [
        float(row[field]) for row in rows if row["class"] in negative_classes
    ]
    value = binary_auroc(positives, negatives)
    return {
        "evaluation_level": evaluation_level,
        "target_context_tokens": length,
        "seed": seed,
        "scope": scope,
        "contrast": contrast,
        "metric": metric,
        "layer_head_reducer": reducer,
        "positive_token_count": len(positives),
        "negative_token_count": len(negatives),
        "auroc": _finite_or_na(value),
    }


def token_auroc_rows(
    tokens: Sequence[dict[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    lengths = sorted({int(row["target_context_tokens"]) for row in tokens})
    within_seed_rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        scope_rows = [row for row in tokens if row["scope"] == scope]
        for length in lengths:
            length_rows = [
                row
                for row in scope_rows
                if int(row["target_context_tokens"]) == length
            ]
            seeds = sorted({int(row["seed"]) for row in length_rows})
            for contrast, (positive_classes, negative_classes) in CONTRASTS.items():
                for metric in metrics:
                    for reducer in REDUCERS:
                        output.append(
                            _auc_row(
                                length_rows,
                                evaluation_level="pooled_seeds_by_length",
                                length=length,
                                seed="all",
                                scope=scope,
                                contrast=contrast,
                                positive_classes=positive_classes,
                                negative_classes=negative_classes,
                                metric=metric,
                                reducer=reducer,
                            )
                        )
                        for seed in seeds:
                            seed_rows = [
                                row
                                for row in length_rows
                                if int(row["seed"]) == seed
                            ]
                            within = _auc_row(
                                seed_rows,
                                evaluation_level="within_seed",
                                length=length,
                                seed=seed,
                                scope=scope,
                                contrast=contrast,
                                positive_classes=positive_classes,
                                negative_classes=negative_classes,
                                metric=metric,
                                reducer=reducer,
                            )
                            output.append(within)
                            if within["auroc"] != "NA":
                                within_seed_rows.append(within)

        for contrast, (positive_classes, negative_classes) in CONTRASTS.items():
            for metric in metrics:
                for reducer in REDUCERS:
                    output.append(
                        _auc_row(
                            scope_rows,
                            evaluation_level="pooled_all_lengths",
                            length="all",
                            seed="all",
                            scope=scope,
                            contrast=contrast,
                            positive_classes=positive_classes,
                            negative_classes=negative_classes,
                            metric=metric,
                            reducer=reducer,
                        )
                    )

    macro_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in within_seed_rows:
        macro_groups[
            (
                row["target_context_tokens"],
                row["scope"],
                row["contrast"],
                row["metric"],
                row["layer_head_reducer"],
            )
        ].append(row)
        macro_groups[
            (
                "all",
                row["scope"],
                row["contrast"],
                row["metric"],
                row["layer_head_reducer"],
            )
        ].append(row)
    for (length, scope, contrast, metric, reducer), rows in sorted(
        macro_groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        values = [float(row["auroc"]) for row in rows]
        output.append(
            {
                "evaluation_level": "macro_mean_of_within_seed_aurocs",
                "target_context_tokens": length,
                "seed": "all",
                "scope": scope,
                "contrast": contrast,
                "metric": metric,
                "layer_head_reducer": reducer,
                "positive_token_count": sum(
                    int(row["positive_token_count"]) for row in rows
                ),
                "negative_token_count": sum(
                    int(row["negative_token_count"]) for row in rows
                ),
                "auroc": rounded(statistics.fmean(values)),
                "seed_case_count": len(rows),
                "seed_auroc_std": rounded(
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["target_context_tokens"]),
            str(row["scope"]),
            str(row["contrast"]),
            str(row["metric"]),
            str(row["layer_head_reducer"]),
            str(row["evaluation_level"]),
            str(row["seed"]),
        ),
    )


def line_aggregates(
    samples: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    *,
    positive_threshold: float,
) -> list[dict[str, Any]]:
    """Aggregate every sampled token/layer/head row belonging to one line."""

    output: list[dict[str, Any]] = []
    for scope in SCOPES:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in _scope_rows(samples, scope):
            key = (
                int(row["target_context_tokens"]),
                int(row["seed"]),
                str(row["class"]),
                str(row["line_id"]),
            )
            groups[key].append(row)
        for (length, seed, category, line_id), rows in sorted(groups.items()):
            common = {
                "target_context_tokens": length,
                "seed": seed,
                "scope": scope,
                "class": category,
                "line_id": line_id,
                "line_span_start": rows[0]["line_span_start"],
                "line_span_end": rows[0]["line_span_end"],
                "line_text": rows[0]["line_text"],
                "line_metadata_source": rows[0]["line_metadata_source"],
                "sampled_token_count": len(
                    {int(row["token_position"]) for row in rows}
                ),
                "layer_count": len({int(row["layer"]) for row in rows}),
                "head_count": len({int(row["head"]) for row in rows}),
                "line_score_row_count": len(rows),
            }
            for metric in metrics:
                reduced = aggregate_values(
                    [float(row[metric]) for row in rows],
                    positive_threshold,
                )
                for reducer, score in reduced.items():
                    output.append(
                        {
                            **common,
                            "metric": metric,
                            "line_reducer": reducer,
                            "score": rounded(score),
                        }
                    )
    return output


def paired_line_comparisons(
    lines: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in lines:
        if row["class"] not in ("gold_evidence", "conflict_evidence"):
            continue
        key = (
            int(row["target_context_tokens"]),
            int(row["seed"]),
            str(row["scope"]),
            str(row["metric"]),
            str(row["line_reducer"]),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for (length, seed, scope, metric, reducer), rows in sorted(groups.items()):
        gold = [row for row in rows if row["class"] == "gold_evidence"]
        conflict = [
            row for row in rows if row["class"] == "conflict_evidence"
        ]
        if not gold or not conflict:
            continue
        gold_score = statistics.fmean(float(row["score"]) for row in gold)
        conflict_score = statistics.fmean(
            float(row["score"]) for row in conflict
        )
        gap = gold_score - conflict_score
        win_value = 1.0 if gap > 0.0 else (0.5 if gap == 0.0 else 0.0)
        output.append(
            {
                "target_context_tokens": length,
                "seed": seed,
                "scope": scope,
                "metric": metric,
                "line_reducer": reducer,
                "gold_line_count": len(gold),
                "conflict_line_count": len(conflict),
                "gold_line_ids": ",".join(sorted(str(row["line_id"]) for row in gold)),
                "conflict_line_ids": ",".join(
                    sorted(str(row["line_id"]) for row in conflict)
                ),
                "gold_line_score": rounded(gold_score),
                "conflict_line_score": rounded(conflict_score),
                "gold_minus_conflict_gap": rounded(gap),
                "gold_strict_win": int(gap > 0.0),
                "tie": int(gap == 0.0),
                "paired_win_value": win_value,
            }
        )
    return output


def summarize_paired_lines(
    comparisons: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        base = (
            str(row["scope"]),
            str(row["metric"]),
            str(row["line_reducer"]),
        )
        groups[(int(row["target_context_tokens"]), *base)].append(row)
        groups[("all", *base)].append(row)
    output: list[dict[str, Any]] = []
    for (length, scope, metric, reducer), rows in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        gaps = [float(row["gold_minus_conflict_gap"]) for row in rows]
        output.append(
            {
                "target_context_tokens": length,
                "scope": scope,
                "metric": metric,
                "line_reducer": reducer,
                "paired_seed_count": len(rows),
                "paired_win_rate_ties_half": rounded(
                    statistics.fmean(
                        float(row["paired_win_value"]) for row in rows
                    )
                ),
                "strict_win_rate": rounded(
                    statistics.fmean(float(row["gold_strict_win"]) for row in rows)
                ),
                "tie_rate": rounded(
                    statistics.fmean(float(row["tie"]) for row in rows)
                ),
                "mean_gold_minus_conflict_gap": rounded(
                    statistics.fmean(gaps)
                ),
                "median_gold_minus_conflict_gap": rounded(
                    statistics.median(gaps)
                ),
                "q10_gold_minus_conflict_gap": rounded(quantile(gaps, 0.10)),
                "q90_gold_minus_conflict_gap": rounded(quantile(gaps, 0.90)),
            }
        )
    return output


def loso_equal_weight_line_combination(
    lines: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Optional label-free LOSO combination with train-fold-only scaling."""

    vectors: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in lines:
        if row["class"] not in ("gold_evidence", "conflict_evidence"):
            continue
        key = (
            int(row["target_context_tokens"]),
            str(row["scope"]),
            int(row["seed"]),
            str(row["class"]),
            str(row["line_id"]),
        )
        feature = f"{row['metric']}__{row['line_reducer']}"
        vectors[key][feature] = float(row["score"])
    output: list[dict[str, Any]] = []
    length_scopes = sorted(
        {(key[0], key[1]) for key in vectors}, key=lambda item: (item[0], item[1])
    )
    for length, scope in length_scopes:
        case_vectors = {
            key: features
            for key, features in vectors.items()
            if key[0] == length and key[1] == scope
        }
        seeds = sorted({key[2] for key in case_vectors})
        if len(seeds) < 2:
            continue
        for held_out_seed in seeds:
            training = {
                key: features
                for key, features in case_vectors.items()
                if key[2] != held_out_seed
            }
            held_out = {
                key: features
                for key, features in case_vectors.items()
                if key[2] == held_out_seed
            }
            if not training or not held_out:
                continue
            # Feature eligibility is determined from the training fold only.
            # The held-out fold may invalidate a fold through missing fields,
            # but it can never add or remove a feature based on its values.
            common_features = set.intersection(
                *(set(features) for features in training.values())
            )
            if any(
                not common_features.issubset(features)
                for features in held_out.values()
            ):
                continue
            scales: dict[str, tuple[float, float]] = {}
            for feature in sorted(common_features):
                values = [features[feature] for features in training.values()]
                std = statistics.pstdev(values) if len(values) > 1 else 0.0
                if std > 1e-12:
                    scales[feature] = (statistics.fmean(values), std)
            if not scales:
                continue
            combined_by_class: dict[str, list[float]] = defaultdict(list)
            for key, features in held_out.items():
                combined = statistics.fmean(
                    (features[feature] - mean) / std
                    for feature, (mean, std) in scales.items()
                )
                combined_by_class[key[3]].append(combined)
            gold = combined_by_class.get("gold_evidence", [])
            conflict = combined_by_class.get("conflict_evidence", [])
            if not gold or not conflict:
                continue
            gold_score = statistics.fmean(gold)
            conflict_score = statistics.fmean(conflict)
            gap = gold_score - conflict_score
            output.append(
                {
                    "target_context_tokens": length,
                    "scope": scope,
                    "held_out_seed": held_out_seed,
                    "training_seed_count": len(seeds) - 1,
                    "training_seed_ids": ",".join(
                        str(seed) for seed in seeds if seed != held_out_seed
                    ),
                    "feature_count": len(scales),
                    "feature_weight_policy": "fixed_equal_positive_weights",
                    "standardization_policy": (
                        "unlabeled_training_seeds_only_mean_and_pstdev"
                    ),
                    "uses_labels_for_weights": 0,
                    "gold_combined_score": rounded(gold_score),
                    "conflict_combined_score": rounded(conflict_score),
                    "gold_minus_conflict_gap": rounded(gap),
                    "gold_strict_win": int(gap > 0.0),
                    "tie": int(gap == 0.0),
                    "paired_win_value": (
                        1.0 if gap > 0.0 else (0.5 if gap == 0.0 else 0.0)
                    ),
                }
            )
    return output


def summarize_loso(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (int(row["target_context_tokens"]), str(row["scope"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (length, scope), selected in sorted(groups.items()):
        gaps = [float(row["gold_minus_conflict_gap"]) for row in selected]
        output.append(
            {
                "target_context_tokens": length,
                "scope": scope,
                "held_out_case_count": len(selected),
                "paired_win_rate_ties_half": rounded(
                    statistics.fmean(
                        float(row["paired_win_value"]) for row in selected
                    )
                ),
                "mean_gold_minus_conflict_gap": rounded(
                    statistics.fmean(gaps)
                ),
                "feature_weight_policy": "fixed_equal_positive_weights",
                "standardization_policy": (
                    "unlabeled_training_seeds_only_mean_and_pstdev"
                ),
            }
        )
    return output


def analyze(
    inputs: Sequence[str | Path],
    output_dir: Path,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    positive_threshold: float = 0.0,
    enable_loso_combination: bool = False,
) -> dict[str, Any]:
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError("at least one metric is required")
    if len(set(metrics)) != len(metrics):
        raise ValueError("metric names must be unique")
    if not math.isfinite(float(positive_threshold)):
        raise ValueError("positive threshold must be finite")
    sample_files = discover_sample_files(inputs)
    samples, load_summary = load_samples(sample_files, metrics)
    tokens = token_aggregates(
        samples, metrics, positive_threshold=positive_threshold
    )
    token_aurocs = token_auroc_rows(tokens, metrics)
    lines = line_aggregates(
        samples, metrics, positive_threshold=positive_threshold
    )
    paired = paired_line_comparisons(lines)
    paired_summary = summarize_paired_lines(paired)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "token_aggregates.csv", tokens)
    write_csv(output_dir / "token_aurocs.csv", token_aurocs)
    write_csv(output_dir / "line_aggregates.csv", lines)
    write_csv(output_dir / "paired_line_comparisons.csv", paired)
    write_csv(output_dir / "paired_line_summary.csv", paired_summary)

    loso_rows: list[dict[str, Any]] = []
    loso_summary: list[dict[str, Any]] = []
    if enable_loso_combination:
        loso_rows = loso_equal_weight_line_combination(lines)
        loso_summary = summarize_loso(loso_rows)
        write_csv(output_dir / "loso_line_combination.csv", loso_rows)
        write_csv(
            output_dir / "loso_line_combination_summary.csv", loso_summary
        )

    summary = {
        "schema_version": 1,
        "experiment": "suppression_certificate_block_aggregation",
        "cpu_only": True,
        "sample_files": [str(path) for path in sample_files],
        "metrics": list(metrics),
        "layer_head_reducers": list(REDUCERS),
        "scopes": list(SCOPES),
        "positive_threshold": float(positive_threshold),
        "raw_sample_count": len(samples),
        "token_aggregate_count": len(tokens),
        "token_auroc_row_count": len(token_aurocs),
        "line_aggregate_count": len(lines),
        "paired_line_comparison_count": len(paired),
        "paired_line_summary": paired_summary,
        "loso_combination_enabled": bool(enable_loso_combination),
        "loso_combination_default": "disabled_to_avoid_leakage",
        "loso_combination_policy": (
            "fixed equal positive weights; unlabeled mean/std fitted only on "
            "non-held-out seeds"
        ),
        "loso_combination_rows": loso_rows,
        "loso_combination_summary": loso_summary,
        **load_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only token and evidence-line aggregation for suppression "
            "certificate JSONL shards"
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="sample JSONL files, shard directories, or a merged output",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="comma-separated score fields",
    )
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument(
        "--enable-loso-combination",
        action="store_true",
        help=(
            "opt in to fixed equal-weight LOSO line features; scaling is "
            "fitted on non-held-out seeds only"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(
        args.inputs,
        Path(args.output_dir),
        metrics=parse_csv(args.metrics),
        positive_threshold=float(args.positive_threshold),
        enable_loso_combination=bool(args.enable_loso_combination),
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "raw_sample_count": summary["raw_sample_count"],
                "token_aggregate_count": summary["token_aggregate_count"],
                "paired_line_comparison_count": summary[
                    "paired_line_comparison_count"
                ],
                "loso_combination_enabled": summary[
                    "loso_combination_enabled"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
