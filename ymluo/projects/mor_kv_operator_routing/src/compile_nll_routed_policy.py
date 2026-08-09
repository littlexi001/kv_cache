from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a task-conditional operator policy from dev NLL and audit on test NLL."
    )
    parser.add_argument("--dev_nll_rows", required=True)
    parser.add_argument("--test_nll_rows", required=True)
    parser.add_argument("--router_predictions", required=True)
    parser.add_argument("--retrieval_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_methods", default="bm25_b4,single_hybrid_b4,mor_kv_b4")
    parser.add_argument("--baseline_method", default="bm25_b4")
    parser.add_argument("--output_method", default="mor_kv_nll_routed_b4")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [item.strip() for item in args.candidate_methods.split(",") if item.strip()]
    dev_rows = read_csv(Path(args.dev_nll_rows))
    test_rows = read_csv(Path(args.test_nll_rows))
    predictions = {
        int(row["query_id"]): row["predicted_task"]
        for row in read_csv(Path(args.router_predictions))
        if row["split"] == "test"
    }
    retrieval_rows = read_csv(Path(args.retrieval_results))

    tasks = sorted({row["task_type"] for row in dev_rows})
    policy: dict[str, str] = {}
    dev_table: list[dict[str, Any]] = []
    for task in tasks:
        scored: list[tuple[float, str]] = []
        for method in candidates:
            group = [
                row
                for row in dev_rows
                if row["task_type"] == task and row["method"] == method
            ]
            if not group:
                raise ValueError(f"Missing dev NLL rows for {task}/{method}")
            mean_nll = statistics.fmean(float(row["answer_nll"]) for row in group)
            scored.append((mean_nll, method))
            dev_table.append(
                {"task_type": task, "method": method, "dev_mean_answer_nll": mean_nll}
            )
        policy[task] = min(scored)[1]

    test_by_key = {
        (int(row["query_id"]), row["method"]): row for row in test_rows
    }
    routed_rows: list[dict[str, Any]] = []
    for query_id, predicted_task in sorted(predictions.items()):
        selected_method = policy[predicted_task]
        source = test_by_key[(query_id, selected_method)]
        baseline = test_by_key[(query_id, args.baseline_method)]
        routed_rows.append(
            {
                **source,
                "method": args.output_method,
                "routed_task": predicted_task,
                "selected_source_method": selected_method,
                "nll_delta_vs_baseline": float(source["answer_nll"])
                - float(baseline["answer_nll"]),
            }
        )

    selected_retrieval = {
        (int(row["query_id"]), row["method"]): row
        for row in retrieval_rows
        if row["split"] == "test"
    }
    routed_retrieval: list[dict[str, Any]] = []
    for query_id, predicted_task in sorted(predictions.items()):
        selected_method = policy[predicted_task]
        source = selected_retrieval[(query_id, selected_method)]
        routed_retrieval.append(
            {
                **source,
                "method": args.output_method,
                "routed_task": predicted_task,
                "selected_source_method": selected_method,
            }
        )

    task_summary: list[dict[str, Any]] = []
    for task in ["all", *sorted({row["task_type"] for row in routed_rows})]:
        group = [
            row for row in routed_rows if task == "all" or row["task_type"] == task
        ]
        task_summary.append(
            {
                "task_type": task,
                "queries": len(group),
                "mean_answer_nll": statistics.fmean(float(row["answer_nll"]) for row in group),
                "mean_nll_delta_vs_baseline": statistics.fmean(
                    float(row["nll_delta_vs_baseline"]) for row in group
                ),
                "win_rate_vs_baseline": statistics.fmean(
                    float(row["nll_delta_vs_baseline"]) < 0.0 for row in group
                ),
                "mean_gold_hits": statistics.fmean(float(row["gold_hits"]) for row in group),
                "mean_hard_negative_hits": statistics.fmean(
                    float(row["hard_negative_hits"]) for row in group
                ),
            }
        )

    rng = np.random.default_rng(args.seed)
    comparisons: list[dict[str, Any]] = []
    for reference in candidates:
        deltas = np.asarray(
            [
                float(row["answer_nll"])
                - float(test_by_key[(int(row["query_id"]), reference)]["answer_nll"])
                for row in routed_rows
            ],
            dtype=np.float64,
        )
        bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
        for index in range(args.bootstrap_samples):
            sample = rng.integers(0, deltas.size, size=deltas.size)
            bootstrap[index] = deltas[sample].mean()
        comparisons.append(
            {
                "reference_method": reference,
                "mean_nll_delta": float(deltas.mean()),
                "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                "win_rate": float(np.mean(deltas < 0.0)),
                "tie_rate": float(np.mean(deltas == 0.0)),
            }
        )

    plot_path: str | None = None
    try:
        import matplotlib.pyplot as plt

        method_means = {
            method: statistics.fmean(
                float(row["answer_nll"]) for row in test_rows if row["method"] == method
            )
            for method in candidates
        }
        method_means[args.output_method] = statistics.fmean(
            float(row["answer_nll"]) for row in routed_rows
        )
        labels = list(method_means)
        values = [method_means[label] for label in labels]
        colors = ["#9ecae1", "#6baed6", "#4292c6", "#08519c"][: len(labels)]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        ax.set_ylabel("Mean answer NLL (lower is better)")
        ax.set_title("Held-out budget-4 answer NLL")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / "nll_routed_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_path = str(path)
    except ImportError:
        pass

    write_csv(output_dir / "dev_candidate_nll.csv", dev_table, list(dev_table[0]))
    write_csv(output_dir / "routed_test_nll_rows.csv", routed_rows, list(routed_rows[0]))
    write_csv(output_dir / "routed_test_summary.csv", task_summary, list(task_summary[0]))
    write_csv(
        output_dir / "routed_query_results.csv", routed_retrieval, list(routed_retrieval[0])
    )
    summary = {
        "source": "dev-NLL-compiled task operator policy, frozen on test",
        "candidate_methods": candidates,
        "baseline_method": args.baseline_method,
        "policy": policy,
        "test": task_summary,
        "paired_comparisons": comparisons,
        "plot_path": plot_path,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
