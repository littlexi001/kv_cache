#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value)
        if text == "" or text.lower() in {"nan", "none"}:
            return default
        return float(text)
    except Exception:
        return default


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("benchmark", ""), row.get("task", ""), row.get("sample_id", ""))


FEATURES: list[tuple[str, str, Callable[[float, float], bool]]] = [
    ("ours_score_gap2", "le", lambda value, threshold: value <= threshold),
    ("ours_score_gap3", "le", lambda value, threshold: value <= threshold),
    ("ours_score_entropy", "ge", lambda value, threshold: value >= threshold),
    ("ours_score_max", "le", lambda value, threshold: value <= threshold),
    ("ours_score_mean", "le", lambda value, threshold: value <= threshold),
    ("ours_score_risk_linear_value", "ge", lambda value, threshold: value >= threshold),
    ("raw_prefix_tokens", "ge", lambda value, threshold: value >= threshold),
]


SAFE_FEATURES: list[tuple[str, str, Callable[[float, float], bool]]] = [
    ("ours_score_gap2", "ge", lambda value, threshold: value >= threshold),
    ("ours_score_gap3", "ge", lambda value, threshold: value >= threshold),
    ("ours_score_entropy", "le", lambda value, threshold: value <= threshold),
    ("ours_score_max", "ge", lambda value, threshold: value >= threshold),
    ("ours_score_mean", "ge", lambda value, threshold: value >= threshold),
    ("ours_score_risk_linear_value", "le", lambda value, threshold: value <= threshold),
    ("raw_prefix_tokens", "le", lambda value, threshold: value <= threshold),
]


def metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y and p)
    fp = sum(1 for y, p in zip(y_true, y_pred) if not y and p)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y and not p)
    tn = sum(1 for y, p in zip(y_true, y_pred) if not y and not p)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trigger_rate": (tp + fp) / max(1, len(y_true)),
        "danger_rate": sum(y_true) / max(1, len(y_true)),
    }


def search_single_feature(task: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y_true = [int(row["danger"]) for row in rows]
    out: list[dict[str, Any]] = []
    for feature, direction, predicate in FEATURES:
        values = sorted({fnum(row.get(feature), 0.0) for row in rows})
        if not values:
            continue
        candidates = values
        if len(values) > 80:
            candidates = [values[int((len(values) - 1) * i / 79)] for i in range(80)]
        for threshold in candidates:
            y_pred = [int(predicate(fnum(row.get(feature), 0.0), threshold)) for row in rows]
            row = {
                "task": task,
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
            }
            row.update(metrics(y_true, y_pred))
            out.append(row)
    out.sort(key=lambda item: (item["f1"], item["recall"], -abs(item["trigger_rate"] - item["danger_rate"])), reverse=True)
    return out


def search_safe_certificate(task: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    y_true = [1 - int(row["danger"]) for row in rows]
    out: list[dict[str, Any]] = []
    for feature, direction, predicate in SAFE_FEATURES:
        values = sorted({fnum(row.get(feature), 0.0) for row in rows})
        if not values:
            continue
        candidates = values
        if len(values) > 80:
            candidates = [values[int((len(values) - 1) * i / 79)] for i in range(80)]
        for threshold in candidates:
            y_pred = [int(predicate(fnum(row.get(feature), 0.0), threshold)) for row in rows]
            row = {
                "task": task,
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
            }
            row.update(metrics(y_true, y_pred))
            # false_positive here means a dangerous example would be certified safe.
            row["unsafe_certified_rate"] = row["fp"] / max(1, sum(1 for item in y_true if item == 0))
            out.append(row)
    out.sort(
        key=lambda item: (
            item["precision"] >= 0.90,
            item["precision"],
            item["recall"],
            -item["unsafe_certified_rate"],
        ),
        reverse=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--reference_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {key(row): row for row in read_csv(Path(args.labels_csv))}
    ref_rows = read_csv(Path(args.reference_results))

    joined: list[dict[str, Any]] = []
    for ref in ref_rows:
        label = labels.get(key(ref))
        if label is None:
            continue
        row: dict[str, Any] = dict(ref)
        row["full_score"] = fnum(label.get("full_score"))
        row["safe_threshold"] = fnum(label.get("safe_threshold"))
        row["danger"] = 0 if int(fnum(label.get("has_safe_sparse_action"), 0)) else 1
        row["min_safe_label"] = label.get("min_safe_label", "")
        joined.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        grouped[str(row.get("task", ""))].append(row)

    summary: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    safe_candidates: list[dict[str, Any]] = []
    for task, subset in sorted(grouped.items()):
        if not subset:
            continue
        y_true = [int(row["danger"]) for row in subset]
        current_pred = [
            int(
                fnum(row.get("ours_score_risk_triggered"), 0.0) > 0.5
                or fnum(row.get("ours_coverage_risk_triggered"), 0.0) > 0.5
                or fnum(row.get("ours_retry_full_fallback_active"), 0.0) > 0.5
                or fnum(row.get("ours_consistency_full_fallback_active"), 0.0) > 0.5
            )
            for row in subset
        ]
        current = {"task": task, "rule": "current_v300_fallback"}
        current.update(metrics(y_true, current_pred))
        summary.append(current)
        task_candidates = search_single_feature(task, subset)[: max(1, args.top_k)]
        candidates.extend(task_candidates)
        if task_candidates:
            best = dict(task_candidates[0])
            best["rule"] = "best_single_feature"
            summary.append(best)
        task_safe_candidates = search_safe_certificate(task, subset)[: max(1, args.top_k)]
        safe_candidates.extend(task_safe_candidates)
        if task_safe_candidates:
            best_safe = dict(task_safe_candidates[0])
            best_safe["rule"] = "best_safe_certificate"
            summary.append(best_safe)

    write_csv(output_dir / "score_risk_rule_summary.csv", summary)
    write_csv(output_dir / "score_risk_rule_candidates.csv", candidates)
    write_csv(output_dir / "score_safe_rule_candidates.csv", safe_candidates)
    payload = {"summary_rows": len(summary), "candidate_rows": len(candidates), "output_dir": str(output_dir)}
    (output_dir / "score_risk_rule_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
