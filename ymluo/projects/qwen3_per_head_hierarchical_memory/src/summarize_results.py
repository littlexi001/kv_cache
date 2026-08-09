#!/usr/bin/env python3
"""Create compact, machine-readable summaries for the hierarchical-memory runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    outputs = args.project_dir / "outputs"
    rows: list[dict[str, Any]] = []

    for recent in (256, 384, 448, 480):
        data = read_json(outputs / f"war16k_recent{recent}_v2_20260716" / "summary.json")
        for policy, metrics in data["test_metrics"].items():
            rows.append(
                {
                    "experiment": f"oracle_recall_recent{recent}",
                    "policy": policy,
                    "test_tokens": 64,
                    "active_heads": 448 if policy != "sink_recent_500" else 0,
                    "mean_nll": "",
                    "ppl": "",
                    "oracle_position_recall": metrics["oracle_position_recall"],
                    "oracle_mass_recall": metrics["oracle_mass_recall"],
                    "remote_oracle_position_recall": metrics[
                        "remote_oracle_position_recall"
                    ],
                    "gqa_union_vs_single_l0": data[
                        "test_mean_gqa_union_vs_single_l0"
                    ][policy],
                }
            )

    sparse = read_json(
        outputs / "sparse_ppl_war16k_recent448_v1_20260716" / "summary.json"
    )
    for result in sparse["results"]:
        rows.append(
            {
                "experiment": "sparse_ppl_uniform_64q",
                "policy": result["policy"],
                "test_tokens": result["test_tokens"],
                "active_heads": 448 if "function" in result["policy"] else 0,
                "mean_nll": result["test_mean_nll"],
                "ppl": result["test_ppl"],
                "oracle_position_recall": "",
                "oracle_mass_recall": "",
                "remote_oracle_position_recall": "",
                "gqa_union_vs_single_l0": "",
            }
        )

    for category in ("semantic_evidence", "lexical_copy", "structural_anchor"):
        data = read_json(
            outputs
            / f"sparse_ppl_war16k_gated_{category}_v2_20260716"
            / "summary.json"
        )
        result = data["results"][0]
        rows.append(
            {
                "experiment": f"sparse_ppl_gated_{category}_64q",
                "policy": result["policy"],
                "test_tokens": result["test_tokens"],
                "active_heads": (
                    data["invariants"]["heads_with_promotions"]
                    if result["policy"] == "hier_function_500"
                    else 0
                ),
                "mean_nll": result["test_mean_nll"],
                "ppl": result["test_ppl"],
                "oracle_position_recall": "",
                "oracle_mass_recall": "",
                "remote_oracle_position_recall": "",
                "gqa_union_vs_single_l0": "",
            }
        )

    for suffix in ("full", "recent500", "semantic_memory"):
        data = read_json(
            outputs / f"sparse_ppl_war16k_512q_{suffix}_20260716" / "summary.json"
        )
        result = data["results"][0]
        rows.append(
            {
                "experiment": "sparse_ppl_semantic_confirmation_512q",
                "policy": result["policy"],
                "test_tokens": result["test_tokens"],
                "active_heads": (
                    data["invariants"]["heads_with_promotions"]
                    if result["policy"] == "hier_function_500"
                    else 0
                ),
                "mean_nll": result["test_mean_nll"],
                "ppl": result["test_ppl"],
                "oracle_position_recall": "",
                "oracle_mass_recall": "",
                "remote_oracle_position_recall": "",
                "gqa_union_vs_single_l0": "",
            }
        )

    for test_tokens in (64, 512):
        run_dir = outputs / f"oracle_top2_war16k_aligned_{test_tokens}q_20260716"
        with (run_dir / "ppl_by_mode.csv").open(encoding="utf-8", newline="") as handle:
            oracle_rows = list(csv.DictReader(handle))
        for result in oracle_rows:
            rows.append(
                {
                    "experiment": f"oracle_top2_aligned_{test_tokens}q",
                    "policy": result["mode"],
                    "test_tokens": result["token_count"],
                    "active_heads": 448 if result["mode"] == "top2" else 0,
                    "mean_nll": result["loss"],
                    "ppl": result["ppl"],
                    "oracle_position_recall": "",
                    "oracle_mass_recall": "",
                    "remote_oracle_position_recall": "",
                    "gqa_union_vs_single_l0": "",
                }
            )
    write_csv(args.project_dir / "analysis" / "experiment_summary.csv", rows)

    recall_dir = outputs / "war16k_512q_semantic_recall_20260716"
    with (recall_dir / "head_function_mixture.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        routes = {
            (int(row["layer"]), int(row["head"])): row
            for row in csv.DictReader(handle)
        }
    with (recall_dir / "per_head_memory_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        metrics = {
            (row["split"], row["policy"], int(row["layer"]), int(row["head"])): row
            for row in csv.DictReader(handle)
        }
    head_rows: list[dict[str, Any]] = []
    for (layer, head), route in routes.items():
        if int(route["promotion_slots"]) <= 0:
            continue
        output: dict[str, Any] = {
            "head_id": route["head_id"],
            "confidence": route["confidence"],
            "promotion_slots": route["promotion_slots"],
        }
        for split in ("train", "test"):
            baseline = metrics[(split, "sink_recent_500", layer, head)]
            hierarchy = metrics[(split, "hier_function_500", layer, head)]
            output[f"{split}_delta_position_recall"] = float(
                hierarchy["oracle_position_recall"]
            ) - float(baseline["oracle_position_recall"])
            output[f"{split}_delta_mass_recall"] = float(
                hierarchy["oracle_mass_recall"]
            ) - float(baseline["oracle_mass_recall"])
        output["test_remote_position_recall"] = metrics[
            ("test", "hier_function_500", layer, head)
        ]["remote_oracle_position_recall"]
        head_rows.append(output)
    write_csv(args.project_dir / "analysis" / "semantic_head_recall_deltas.csv", head_rows)

    print(f"wrote {len(rows)} experiment rows and {len(head_rows)} head rows")


if __name__ == "__main__":
    main()
