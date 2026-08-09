#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FEATURES = [
    "raw_prefix_tokens",
    "keep_fraction",
    "ours_score_max",
    "ours_score_mean",
    "ours_score_gap2",
    "ours_score_gap3",
    "ours_score_entropy",
    "ours_score_positive_fraction",
    "ours_query_coverage_terms",
    "ours_query_coverage_recall",
    "ours_span_repack_active",
    "ours_span_repack_tokens",
    "ours_span_repack_windows",
    "ours_bridge_active",
    "ours_full_fallback_active",
    "ours_score_risk_triggered",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def label_for_dir(path: Path) -> str:
    name = path.name
    if name.startswith("riskkv_v19_"):
        name = name[len("riskkv_v19_") :]
    suffixes = [
        "_20260711_b16_compressed_mid_smoke_m20_bDyn_pDyn",
        "_20260711_b16_manyblocks_xl_smoke_m20_bDyn_pDyn",
        "_20260711_bm25_bridge_smoke_m20_bDyn_pDyn",
        "_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn",
        "_20260711_qasper_bm25_budget_smoke_m20_bDyn_pDyn",
        "_20260711_b16_windowvote_sweep_m100_bDyn_pDyn",
        "_20260711_b16_microspan_sweep_m100_bDyn_pDyn",
        "_20260711_b16_purefine_sweep_m100_bDyn_pDyn",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        task = row.get("task", "")
        sample_id = row.get("sample_id", "")
        if task and sample_id:
            out[(task, sample_id)] = row
    return out


def summarize_joined(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["candidate"]), str(row["task"]))].append(row)
        grouped[(str(row["candidate"]), "ALL")].append(row)
    out: list[dict[str, Any]] = []
    for (candidate, task), subset in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] != "ALL", item[0][1])):
        deltas = [float(row["delta_vs_v300"]) for row in subset]
        full_deltas = [float(row["delta_vs_full"]) for row in subset]
        safe = [row for row in subset if float(row["delta_vs_v300"]) >= -1e-9]
        quality_ok = [row for row in subset if float(row["score"]) + 1e-9 >= 0.95 * float(row["v300_score"])]
        out.append(
            {
                "candidate": candidate,
                "task": task,
                "samples": len(subset),
                "score": mean([float(row["score"]) for row in subset]),
                "v300_score": mean([float(row["v300_score"]) for row in subset]),
                "full_score": mean([float(row["full_score"]) for row in subset]),
                "delta_vs_v300": mean(deltas),
                "delta_vs_full": mean(full_deltas),
                "safe_ge_v300_rate": len(safe) / max(1, len(subset)),
                "score_ge_95pct_v300_rate": len(quality_ok) / max(1, len(subset)),
                "kv_keep": mean([float(row["kv_keep"]) for row in subset]),
                "v300_kv_keep": mean([float(row["v300_kv_keep"]) for row in subset]),
                "online_seconds": mean([float(row["online_seconds"]) for row in subset]),
                "v300_online_seconds": mean([float(row["v300_online_seconds"]) for row in subset]),
                "online_speed_vs_v300": mean([float(row["v300_online_seconds"]) for row in subset])
                / max(1e-12, mean([float(row["online_seconds"]) for row in subset])),
                "matched": len(subset),
            }
        )
    return out


def threshold_rules(rows: list[dict[str, Any]], min_coverage: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["candidate"]), str(row["task"]))].append(row)
    for (candidate, task), subset in sorted(grouped.items()):
        total = len(subset)
        if total < min_coverage:
            continue
        for feature in FEATURES:
            values = sorted({float(row.get(feature, 0.0) or 0.0) for row in subset})
            if len(values) <= 1:
                continue
            quantiles = []
            for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
                quantiles.append(values[idx])
            for direction in ["<=", ">="]:
                for threshold in sorted(set(quantiles)):
                    if direction == "<=":
                        selected = [row for row in subset if float(row.get(feature, 0.0) or 0.0) <= threshold]
                    else:
                        selected = [row for row in subset if float(row.get(feature, 0.0) or 0.0) >= threshold]
                    if len(selected) < min_coverage:
                        continue
                    safe = [row for row in selected if float(row["delta_vs_v300"]) >= -1e-9]
                    ok95 = [row for row in selected if float(row["score"]) + 1e-9 >= 0.95 * float(row["v300_score"])]
                    out.append(
                        {
                            "candidate": candidate,
                            "task": task,
                            "rule": f"{feature} {direction} {threshold:.6g}",
                            "feature": feature,
                            "direction": direction,
                            "threshold": threshold,
                            "coverage": len(selected),
                            "coverage_rate": len(selected) / max(1, total),
                            "safe_ge_v300_rate": len(safe) / max(1, len(selected)),
                            "score_ge_95pct_v300_rate": len(ok95) / max(1, len(selected)),
                            "score": mean([float(row["score"]) for row in selected]),
                            "v300_score": mean([float(row["v300_score"]) for row in selected]),
                            "delta_vs_v300": mean([float(row["delta_vs_v300"]) for row in selected]),
                            "kv_keep": mean([float(row["kv_keep"]) for row in selected]),
                            "v300_kv_keep": mean([float(row["v300_kv_keep"]) for row in selected]),
                            "online_seconds": mean([float(row["online_seconds"]) for row in selected]),
                            "v300_online_seconds": mean([float(row["v300_online_seconds"]) for row in selected]),
                        }
                    )
    out.sort(
        key=lambda row: (
            -float(row["score_ge_95pct_v300_rate"]),
            -float(row["safe_ge_v300_rate"]),
            -float(row["coverage_rate"]),
            float(row["kv_keep"]),
        )
    )
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dir", required=True)
    parser.add_argument("--v300_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_rule_coverage", type=int, default=5)
    parser.add_argument("candidate_dirs", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = by_key(read_csv(Path(args.full_dir) / "task_results.csv"))
    v300 = by_key(read_csv(Path(args.v300_dir) / "task_results.csv"))

    joined: list[dict[str, Any]] = []
    for item in args.candidate_dirs:
        directory = Path(item)
        label = label_for_dir(directory)
        for row in read_csv(directory / "task_results.csv"):
            key = (row.get("task", ""), row.get("sample_id", ""))
            v300_row = v300.get(key)
            full_row = full.get(key)
            if v300_row is None or full_row is None:
                continue
            out: dict[str, Any] = {
                "candidate": label,
                "task": key[0],
                "sample_id": key[1],
                "score": fnum(row, "score"),
                "v300_score": fnum(v300_row, "score"),
                "full_score": fnum(full_row, "score"),
                "delta_vs_v300": fnum(row, "score") - fnum(v300_row, "score"),
                "delta_vs_full": fnum(row, "score") - fnum(full_row, "score"),
                "kv_keep": fnum(row, "keep_fraction"),
                "v300_kv_keep": fnum(v300_row, "keep_fraction"),
                "online_seconds": fnum(row, "online_seconds"),
                "v300_online_seconds": fnum(v300_row, "online_seconds"),
                "prediction": row.get("prediction", ""),
                "v300_prediction": v300_row.get("prediction", ""),
            }
            for feature in FEATURES:
                out[feature] = fnum(row, feature)
            joined.append(out)

    summary = summarize_joined(joined)
    rules = threshold_rules(joined, args.min_rule_coverage)
    write_csv(output_dir / "joined_candidate_actions.csv", joined)
    write_csv(output_dir / "candidate_action_summary.csv", summary)
    write_csv(output_dir / "candidate_action_rules.csv", rules)
    (output_dir / "candidate_action_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "candidate_action_rules.json").write_text(
        json.dumps(rules[:200], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"joined": len(joined), "summary": len(summary), "rules": len(rules), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
