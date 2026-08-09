from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASELINE = "qscore_qmse"
CANDIDATES = ("softmax_fisher", "value_jacobian")
METRICS = (
    "topk_recall",
    "attention_mass",
    "output_relative_l2",
    "output_cosine",
    "output_mse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict paired, hierarchical-bootstrap comparison of spectral "
            "rate-allocation objectives."
        )
    )
    parser.add_argument("--input_csv", type=Path, action="append", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def interval(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "lower_2p5": float(low),
        "median": float(median),
        "upper_97p5": float(high),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for path in args.input_csv:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    paired: dict[
        tuple[str, int, int, int, int, float],
        dict[str, dict[str, str]],
    ] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["label"]),
            int(row["layer"]),
            int(row["heldout_step"]),
            int(row["kv_head"]),
            int(row["query_group"]),
            float(row["selected_fraction"]),
        )
        paired[key][str(row["method"])] = row
    expected = {BASELINE, *CANDIDATES}
    incomplete = [key for key, methods in paired.items() if set(methods) != expected]
    if incomplete:
        raise ValueError(f"incomplete method pairs: {incomplete[:5]}")

    # Heads and GQA groups share the same layer/step state. Average them before
    # bootstrap so they are not treated as independent observations.
    block_values: dict[
        tuple[str, int, int, float, str, str],
        list[float],
    ] = defaultdict(list)
    for key, methods in paired.items():
        label, layer, step, _, _, fraction = key
        for candidate in CANDIDATES:
            for metric in METRICS:
                difference = (
                    float(methods[candidate][metric])
                    - float(methods[BASELINE][metric])
                )
                block_values[
                    (label, layer, step, fraction, candidate, metric)
                ].append(difference)

    blocks: dict[
        tuple[str, int, int, float, str, str],
        float,
    ] = {
        key: float(np.mean(values)) for key, values in block_values.items()
    }
    labels = sorted({key[0] for key in blocks})
    layers_by_label = {
        label: sorted({key[1] for key in blocks if key[0] == label})
        for label in labels
    }
    steps_by_label_layer = {
        (label, layer): sorted(
            {
                key[2]
                for key in blocks
                if key[0] == label and key[1] == layer
            }
        )
        for label in labels
        for layer in layers_by_label[label]
    }
    layer_count = len(layers_by_label[labels[0]])
    step_count = len(steps_by_label_layer[(labels[0], layers_by_label[labels[0]][0])])
    if any(len(layers_by_label[label]) != layer_count for label in labels):
        raise ValueError("hierarchical bootstrap requires balanced layer counts")
    if any(
        len(steps_by_label_layer[(label, layer)]) != step_count
        for label in labels
        for layer in layers_by_label[label]
    ):
        raise ValueError("hierarchical bootstrap requires balanced step counts")
    fractions = sorted({key[3] for key in blocks})
    rng = np.random.default_rng(args.seed)
    replicate_count = args.bootstrap_replicates
    label_draw = rng.integers(
        0,
        len(labels),
        size=(replicate_count, len(labels)),
    )
    layer_draw = rng.integers(
        0,
        layer_count,
        size=(replicate_count, len(labels), layer_count),
    )
    step_draw = rng.integers(
        0,
        step_count,
        size=(
            replicate_count,
            len(labels),
            layer_count,
            step_count,
        ),
    )
    comparisons: list[dict[str, Any]] = []

    for fraction in fractions:
        for candidate in CANDIDATES:
            for metric in METRICS:
                cube = np.empty(
                    (len(labels), layer_count, step_count),
                    dtype=np.float64,
                )
                for label_index, label in enumerate(labels):
                    for layer_index, layer in enumerate(
                        layers_by_label[label]
                    ):
                        for step_index, step in enumerate(
                            steps_by_label_layer[(label, layer)]
                        ):
                            cube[label_index, layer_index, step_index] = blocks[
                                (
                                    label,
                                    layer,
                                    step,
                                    fraction,
                                    candidate,
                                    metric,
                                )
                            ]
                sampled = cube[
                    label_draw[:, :, None, None],
                    layer_draw[:, :, :, None],
                    step_draw,
                ]
                bootstrap = sampled.mean(axis=(1, 2, 3))
                observed = cube.reshape(-1)
                comparisons.append(
                    {
                        "candidate": candidate,
                        "baseline": BASELINE,
                        "selected_fraction": fraction,
                        "metric": metric,
                        "candidate_minus_baseline": float(observed.mean()),
                        "hierarchical_bootstrap_95_percent": interval(bootstrap),
                        "probability_candidate_better": float(
                            np.mean(
                                bootstrap < 0.0
                                if metric
                                in {"output_relative_l2", "output_mse"}
                                else bootstrap > 0.0
                            )
                        ),
                        "blocks": int(observed.size),
                    }
                )

    output = {
        "protocol": {
            "strict_query_pairs": len(paired),
            "raw_rows": len(rows),
            "labels": labels,
            "layers_per_label": {
                label: layers_by_label[label] for label in labels
            },
            "bootstrap": (
                "resample labels, layers within labels, then held-out steps; "
                "KV heads and GQA groups are averaged within each layer-step"
            ),
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
