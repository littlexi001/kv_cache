from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SIGNALS = ("max_score", "margin", "score_std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether frozen transition scores can gate deeper block reads."
    )
    parser.add_argument("--dev_generation_rows_path", required=True)
    parser.add_argument("--dev_selection_rows_path", required=True)
    parser.add_argument("--dev_retrieval_rows_path", default="")
    parser.add_argument("--test_generation_rows_path", required=True)
    parser.add_argument("--test_selection_rows_path", required=True)
    parser.add_argument("--test_retrieval_rows_path", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--method", default="heuristic_structured")
    parser.add_argument("--expansion_fractions", default="0.25,0.5,0.75")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(path: str) -> dict[int, dict[str, Any]]:
    return {int(row["query_id"]): row for row in read_jsonl(Path(path))}


def confidence_record(
    generation: dict[str, Any],
    selection: dict[str, Any],
    *,
    method: str,
    retrieval_rank: int = 0,
) -> dict[str, Any]:
    selected_index = int(selection[f"{method}_index"])
    scores = np.asarray(selection[f"{method}_scores"], dtype=np.float64)
    ordered = np.sort(scores)[::-1]
    selected_correct = bool(
        generation["branches"][selected_index]["target_hit"]
    )
    return {
        "query_id": int(generation["query_id"]),
        "selected_correct": selected_correct,
        "any_branch_correct": bool(generation["any_branch_target_hit"]),
        "max_score": float(ordered[0]),
        "margin": float(ordered[0] - ordered[1]) if len(ordered) >= 2 else 0.0,
        "score_std": float(scores.std()),
        "retrieval_rank": int(retrieval_rank),
        "expansion_opportunity": bool(
            not selected_correct and 4 <= int(retrieval_rank) <= 16
        ),
    }


def build_records(
    generation_path: str,
    selection_path: str,
    retrieval_path: str,
    *,
    method: str,
) -> list[dict[str, Any]]:
    generations = keyed(generation_path)
    selections = keyed(selection_path)
    retrieval = keyed(retrieval_path) if retrieval_path else {}
    if set(generations) != set(selections):
        raise ValueError("generation and selection rows do not align")
    if retrieval and set(generations) != set(retrieval):
        raise ValueError("retrieval rows do not align")
    return [
        confidence_record(
            generations[query_id],
            selections[query_id],
            method=method,
            retrieval_rank=(
                int(retrieval[query_id]["lexical_rank"]) if retrieval else 0
            ),
        )
        for query_id in sorted(generations)
    ]


def signal_metrics(
    rows: Sequence[dict[str, Any]], signal: str, target: str
) -> dict[str, float]:
    labels = np.asarray([bool(row[target]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row[signal]) for row in rows], dtype=np.float64)
    if len(set(labels.tolist())) < 2:
        return {"roc_auc": float("nan"), "average_precision": float(labels.mean())}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def gate_summary(
    rows: Sequence[dict[str, Any]], signal: str, threshold: float
) -> dict[str, Any]:
    expanded = [float(row[signal]) <= threshold for row in rows]
    high = [row for row, flag in zip(rows, expanded) if not flag]
    low = [row for row, flag in zip(rows, expanded) if flag]
    errors = [row for row in rows if not row["selected_correct"]]
    opportunities = [row for row in rows if row["expansion_opportunity"]]
    expanded_errors = sum(
        flag and not row["selected_correct"] for row, flag in zip(rows, expanded)
    )
    expanded_opportunities = sum(
        flag and row["expansion_opportunity"] for row, flag in zip(rows, expanded)
    )
    baseline_correct = sum(row["selected_correct"] for row in rows)
    return {
        "threshold": threshold,
        "expansion_fraction": sum(expanded) / max(1, len(rows)),
        "high_confidence_queries": len(high),
        "high_confidence_accuracy": (
            sum(row["selected_correct"] for row in high) / len(high) if high else 0.0
        ),
        "low_confidence_accuracy": (
            sum(row["selected_correct"] for row in low) / len(low) if low else 0.0
        ),
        "error_capture": expanded_errors / max(1, len(errors)),
        "expansion_opportunity_capture": expanded_opportunities
        / max(1, len(opportunities)),
        "expansion_opportunities": len(opportunities),
        "perfect_expansion_upper_accuracy": (
            baseline_correct + expanded_opportunities
        )
        / max(1, len(rows)),
    }


def main() -> None:
    args = parse_args()
    dev_rows = build_records(
        args.dev_generation_rows_path,
        args.dev_selection_rows_path,
        args.dev_retrieval_rows_path,
        method=args.method,
    )
    test_rows = build_records(
        args.test_generation_rows_path,
        args.test_selection_rows_path,
        args.test_retrieval_rows_path,
        method=args.method,
    )
    dev_metrics = {
        signal: {
            target: signal_metrics(dev_rows, signal, target)
            for target in ("selected_correct", "any_branch_correct")
        }
        for signal in SIGNALS
    }
    selected_signal = max(
        SIGNALS,
        key=lambda signal: (
            dev_metrics[signal]["selected_correct"]["roc_auc"], signal
        ),
    )
    fractions = [
        float(item.strip())
        for item in args.expansion_fractions.split(",")
        if item.strip()
    ]
    gates = []
    dev_values = np.asarray(
        [float(row[selected_signal]) for row in dev_rows], dtype=np.float64
    )
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError("expansion fractions must be between zero and one")
        threshold = float(np.quantile(dev_values, fraction))
        gates.append(
            {
                "dev_target_expansion_fraction": fraction,
                "dev": gate_summary(dev_rows, selected_signal, threshold),
                "test": gate_summary(test_rows, selected_signal, threshold),
            }
        )
    payload = {
        "source": "dev-selected confidence signal for selective deeper block reads",
        "selection_uses_test_labels": False,
        "method": args.method,
        "dev_queries": len(dev_rows),
        "test_queries": len(test_rows),
        "dev_signal_metrics": dev_metrics,
        "test_signal_metrics": {
            signal: {
                target: signal_metrics(test_rows, signal, target)
                for target in ("selected_correct", "any_branch_correct")
            }
            for signal in SIGNALS
        },
        "selected_signal": selected_signal,
        "dev_selected_accuracy": sum(row["selected_correct"] for row in dev_rows)
        / len(dev_rows),
        "test_selected_accuracy": sum(row["selected_correct"] for row in test_rows)
        / len(test_rows),
        "gates": gates,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
