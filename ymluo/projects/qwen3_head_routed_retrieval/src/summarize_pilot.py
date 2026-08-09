from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a head-routed retrieval pilot.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = read_csv(run_dir / "head_assignments.csv")
    metrics = read_csv(run_dir / "head_retriever_metrics.csv")
    aggregates = read_csv(run_dir / "aggregate_retriever_metrics.csv")
    gqa = read_csv(run_dir / "gqa_union_by_layer_group.csv")
    test_metrics = {
        (int(row["layer"]), int(row["head"]), row["method"]): row
        for row in metrics
        if row["split"] == "test"
    }
    layer_count = 1 + max(int(row["layer"]) for row in assignments)
    head_count = 1 + max(int(row["head"]) for row in assignments)
    methods = list(dict.fromkeys(row["method"] for row in metrics))
    method_index = {method: idx for idx, method in enumerate(methods)}

    route_matrix = [[0 for _ in range(head_count)] for _ in range(layer_count)]
    gain_matrix = [[0.0 for _ in range(head_count)] for _ in range(layer_count)]
    gains: list[dict[str, Any]] = []
    for row in assignments:
        layer = int(row["layer"])
        head = int(row["head"])
        method = row["train_best_method"]
        route_matrix[layer][head] = method_index[method]
        chosen = float(row["test_position_recall"])
        position = float(test_metrics[layer, head, "position"]["position_recall"])
        gain = chosen - position
        gain_matrix[layer][head] = gain
        gains.append(
            {
                "layer": layer,
                "head": head,
                "method": method,
                "test_recall": chosen,
                "position_recall": position,
                "gain": gain,
                "remote_recall": float(row["test_remote_position_recall"]),
            }
        )

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
        from matplotlib.patches import Patch

        colors = [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#B279A2",
            "#FFBF79",
            "#8CD17D",
            "#FF9D9A",
            "#D4A6C8",
            "#9D755D",
        ]
        cmap = ListedColormap(colors[: len(methods)])
        norm = BoundaryNorm(range(len(methods) + 1), cmap.N)
        fig, ax = plt.subplots(figsize=(10.5, 8.0))
        ax.imshow(np.asarray(route_matrix), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xlabel("Query head")
        ax.set_ylabel("Layer")
        ax.set_title("Train-selected external retriever for each layer/query-head")
        ax.set_xticks(range(head_count))
        ax.set_yticks(range(0, layer_count, 2))
        ax.legend(
            handles=[Patch(facecolor=colors[idx], label=method) for method, idx in method_index.items()],
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
        )
        fig.tight_layout()
        fig.savefig(output_dir / "head_route_map.png", dpi=args.dpi)
        plt.close(fig)

        gain_array = np.asarray(gain_matrix)
        bound = max(0.01, float(np.abs(gain_array).max()))
        fig, ax = plt.subplots(figsize=(10.5, 8.0))
        image = ax.imshow(
            gain_array,
            aspect="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
            interpolation="nearest",
        )
        ax.set_xlabel("Query head")
        ax.set_ylabel("Layer")
        ax.set_title("Held-out Top-2% recall gain of routed policy over position")
        ax.set_xticks(range(head_count))
        ax.set_yticks(range(0, layer_count, 2))
        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("Absolute position-recall gain")
        fig.tight_layout()
        fig.savefig(output_dir / "head_recall_gain_map.png", dpi=args.dpi)
        plt.close(fig)

        test_aggregates = [row for row in aggregates if row["split"] == "test"]
        labels = [row["policy"].replace("homogeneous:", "all:").replace("head_routed:", "routed:") for row in test_aggregates]
        position_values = [100.0 * float(row["position_recall"]) for row in test_aggregates]
        remote_values = [100.0 * float(row["remote_position_recall"]) for row in test_aggregates]
        y = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(labels))))
        ax.barh(y - 0.18, position_values, height=0.34, label="all oracle positions")
        ax.barh(y + 0.18, remote_values, height=0.34, label="remote oracle positions")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Held-out oracle Top-2% position recall (%)")
        ax.set_title("Equal-budget retriever imitation")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "retriever_policy_comparison.png", dpi=args.dpi)
        plt.close(fig)
    except ImportError as exc:
        print(f"plotting skipped: {exc}")

    route_counts = Counter(row["train_best_method"] for row in assignments)
    test_oracle_counts = Counter(row["diagnostic_test_oracle_best_method"] for row in assignments)
    stable = sum(
        row["train_best_method"] == row["diagnostic_test_oracle_best_method"]
        for row in assignments
    )
    position_reference = [
        row for row in metrics if row["split"] == "test" and row["method"] == "position"
    ]
    remote_events = sum(int(row["remote_oracle_events"]) for row in position_reference)
    oracle_events = sum(int(row["oracle_events"]) for row in position_reference)
    gqa_expansions = finite([float(row["union_vs_single_head_budget"]) for row in gqa])
    summary = {
        "head_count": len(assignments),
        "route_counts": dict(route_counts),
        "diagnostic_test_oracle_route_counts": dict(test_oracle_counts),
        "train_route_matches_test_oracle_best": stable,
        "train_route_matches_test_oracle_best_fraction": stable / len(assignments),
        "test_oracle_remote_event_fraction": remote_events / oracle_events,
        "heads_improved_vs_position": sum(row["gain"] > 1e-12 for row in gains),
        "heads_hurt_vs_position": sum(row["gain"] < -1e-12 for row in gains),
        "heads_equal_to_position": sum(abs(row["gain"]) <= 1e-12 for row in gains),
        "top_head_gains": sorted(gains, key=lambda row: row["gain"], reverse=True)[:20],
        "bottom_head_gains": sorted(gains, key=lambda row: row["gain"])[:10],
        "gqa_union_vs_single_head_budget": {
            "mean": sum(gqa_expansions) / len(gqa_expansions),
            "min": min(gqa_expansions),
            "max": max(gqa_expansions),
            "expanded_groups": sum(value > 1.001 for value in gqa_expansions),
            "group_count": len(gqa_expansions),
        },
        "test_aggregate_rows": [row for row in aggregates if row["split"] == "test"],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
