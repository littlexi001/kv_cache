from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize and plot the token-by-token two-hop first-token scan."
        )
    )
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def read_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "added_tokens": int(raw["added_tokens"]),
                    "total_tokens": int(raw["total_tokens"]),
                    "kib_tokens": float(raw["kib_tokens"]),
                    "gold_exact_probability": float(
                        raw["gold_exact_probability"]
                    ),
                    "gold_semantic_probability": float(
                        raw["gold_semantic_probability"]
                    ),
                    "gold_exact_vs_competitor_margin": float(
                        raw["gold_exact_vs_competitor_margin"]
                    ),
                    "top_token_id": int(raw["top_token_id"]),
                    "top_token_label": raw["top_token_label"],
                    "top_probability": float(
                        raw["top_probability"]
                    ),
                    "top_is_gold": read_bool(raw["top_is_gold"]),
                    "strongest_competitor_token_id": int(
                        raw["strongest_competitor_token_id"]
                    ),
                    "strongest_competitor_token_label": raw[
                        "strongest_competitor_token_label"
                    ],
                    "strongest_competitor_probability": float(
                        raw["strongest_competitor_probability"]
                    ),
                }
            )
    if not rows:
        raise RuntimeError("points CSV is empty")
    return rows


def failure_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        failed = not bool(row["top_is_gold"])
        if failed and start is None:
            start = index
        if start is not None and (
            not failed or index == len(rows) - 1
        ):
            end = index - 1 if not failed else index
            runs.append(
                {
                    "start_total_tokens": rows[start]["total_tokens"],
                    "end_total_tokens": rows[end]["total_tokens"],
                    "start_kib_tokens": rows[start]["kib_tokens"],
                    "end_kib_tokens": rows[end]["kib_tokens"],
                    "length_tokens": end - start + 1,
                }
            )
            start = None
    return runs


def label_key(token_id: int, label: str) -> str:
    return f"{token_id}:{label}"


def plot_label(label: str) -> str:
    return (
        label.replace("␠", "[space]")
        .replace("↵", "\\n")
        .replace("␍", "\\r")
        .replace("⇥", "\\t")
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    competitor_counts: collections.Counter[str] = collections.Counter()
    prediction_counts: collections.Counter[str] = collections.Counter()
    failures = 0
    flips = 0
    for index, row in enumerate(rows):
        competitor_counts[
            label_key(
                row["strongest_competitor_token_id"],
                row["strongest_competitor_token_label"],
            )
        ] += 1
        prediction_counts[
            label_key(
                row["top_token_id"],
                row["top_token_label"],
            )
        ] += 1
        failures += int(not row["top_is_gold"])
        if (
            index > 0
            and row["top_token_id"] != rows[index - 1]["top_token_id"]
        ):
            flips += 1
    first_failure = next(
        (row for row in rows if not row["top_is_gold"]),
        None,
    )
    runs = failure_runs(rows)
    return {
        "schema_version": 1,
        "experiment": "incremental_twohop_first_token",
        "point_count": len(rows),
        "start_total_tokens": rows[0]["total_tokens"],
        "end_total_tokens": rows[-1]["total_tokens"],
        "failure_count": failures,
        "failure_rate": failures / len(rows),
        "prediction_flip_count": flips,
        "first_failure": first_failure,
        "failure_run_count": len(runs),
        "longest_failure_run_tokens": max(
            (run["length_tokens"] for run in runs),
            default=0,
        ),
        "failure_runs": runs,
        "strongest_competitor_counts": dict(
            competitor_counts.most_common()
        ),
        "top_prediction_counts": dict(
            prediction_counts.most_common()
        ),
    }


def compact_plot_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
    competitors: dict[int, str] = {}
    predictions: dict[int, str] = {}
    for row in rows:
        competitors[
            row["strongest_competitor_token_id"]
        ] = row["strongest_competitor_token_label"]
        predictions[
            row["top_token_id"]
        ] = row["top_token_label"]
    return {
        "schema_version": 1,
        "x_total_tokens": [
            row["total_tokens"] for row in rows
        ],
        "gold_probability": [
            row["gold_exact_probability"] for row in rows
        ],
        "semantic_gold_probability": [
            row["gold_semantic_probability"] for row in rows
        ],
        "strongest_competitor_probability": [
            row["strongest_competitor_probability"] for row in rows
        ],
        "strongest_competitor_token_id": [
            row["strongest_competitor_token_id"] for row in rows
        ],
        "top_token_id": [
            row["top_token_id"] for row in rows
        ],
        "top_is_gold": [
            row["top_is_gold"] for row in rows
        ],
        "margin": [
            row["gold_exact_vs_competitor_margin"] for row in rows
        ],
        "competitor_labels": {
            str(key): value
            for key, value in sorted(competitors.items())
        },
        "prediction_labels": {
            str(key): value
            for key, value in sorted(predictions.items())
        },
    }


def plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x = [row["kib_tokens"] for row in rows]
    gold = [100.0 * row["gold_exact_probability"] for row in rows]
    competitor = [
        100.0 * row["strongest_competitor_probability"]
        for row in rows
    ]
    competitor_ids = [
        row["strongest_competitor_token_id"] for row in rows
    ]
    competitor_labels = {
        row["strongest_competitor_token_id"]: row[
            "strongest_competitor_token_label"
        ]
        for row in rows
    }
    unique_competitors = list(
        dict.fromkeys(competitor_ids)
    )
    palette = list(plt.get_cmap("tab20").colors)
    colors = {
        token_id: palette[index % len(palette)]
        for index, token_id in enumerate(unique_competitors)
    }

    fig, (state_ax, probability_ax) = plt.subplots(
        2,
        1,
        figsize=(16, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55, 5.0]},
        constrained_layout=True,
    )

    top_ids = [row["top_token_id"] for row in rows]
    top_labels = {
        row["top_token_id"]: row["top_token_label"]
        for row in rows
    }
    prediction_ids = list(dict.fromkeys(top_ids))
    prediction_colors: dict[int, Any] = {}
    for token_id in prediction_ids:
        if any(
            row["top_token_id"] == token_id
            and row["top_is_gold"]
            for row in rows
        ):
            prediction_colors[token_id] = "#15803d"
        elif token_id in colors:
            prediction_colors[token_id] = colors[token_id]
        else:
            prediction_colors[token_id] = palette[
                len(prediction_colors) % len(palette)
            ]
    state_ax.scatter(
        x,
        [0.0] * len(x),
        c=[prediction_colors[token_id] for token_id in top_ids],
        marker="|",
        s=18,
        linewidths=1.0,
    )
    state_ax.set_yticks([])
    state_ax.set_ylabel("Top-1")
    state_ax.set_ylim(-0.8, 0.8)
    state_ax.grid(False)

    probability_ax.plot(
        x,
        gold,
        color="#111827",
        linewidth=1.1,
        label="gold: [space]nine",
        zorder=3,
    )

    run_start = 0
    used_labels: set[int] = set()
    for index in range(1, len(rows) + 1):
        boundary = (
            index == len(rows)
            or competitor_ids[index] != competitor_ids[index - 1]
        )
        if not boundary:
            continue
        token_id = competitor_ids[index - 1]
        start = max(0, run_start - 1)
        end = index
        label = None
        if token_id not in used_labels:
            label = (
                "strongest competitor: "
                + plot_label(competitor_labels[token_id])
            )
            used_labels.add(token_id)
        probability_ax.plot(
            x[start:end],
            competitor[start:end],
            color=colors[token_id],
            linewidth=1.0,
            label=label,
            zorder=2,
        )
        if end - run_start == 1:
            probability_ax.scatter(
                [x[run_start]],
                [competitor[run_start]],
                color=colors[token_id],
                s=7,
                zorder=4,
            )
        run_start = index

    probability_ax.axhline(
        0.0,
        color="#9ca3af",
        linewidth=0.7,
    )
    probability_ax.set_ylabel("First-token probability (%)")
    probability_ax.set_xlabel("Total sequence length (Ki tokens)")
    probability_ax.set_xlim(x[0], x[-1])
    probability_ax.grid(
        axis="y",
        color="#d1d5db",
        linewidth=0.6,
        alpha=0.7,
    )
    probability_ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
    )

    state_handles = [
        Line2D(
            [0],
            [0],
            color=prediction_colors[token_id],
            marker="|",
            linestyle="None",
            markersize=10,
            label=plot_label(top_labels[token_id]),
        )
        for token_id in prediction_ids
    ]
    state_ax.legend(
        handles=state_handles,
        loc="upper right",
        frameon=False,
        ncol=min(6, max(1, len(state_handles))),
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.points_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "plot_data.json",
        compact_plot_data(rows),
    )
    plot(
        rows,
        output_dir
        / "first_token_probability_98k_120k.png",
    )
    print(
        json.dumps(
            {
                "point_count": summary["point_count"],
                "failure_rate": summary["failure_rate"],
                "prediction_flip_count": summary[
                    "prediction_flip_count"
                ],
                "competitors": summary[
                    "strongest_competitor_counts"
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
