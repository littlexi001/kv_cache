from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge exact head-distortion teacher shards and compile risk-constrained actions."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--relative_error_thresholds", default="0.02,0.05,0.1")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for raw_path in args.input:
        with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                rows.append(
                    {
                        **raw,
                        "query_id": int(raw["query_id"]),
                        "layer": int(raw["layer"]),
                        "query_head": int(raw["query_head"]),
                        "kv_head": int(raw["kv_head"]),
                        "query_position": int(raw["query_position"]),
                        "selected_blocks": int(raw["selected_blocks"]),
                        "omitted_mass": float(raw["omitted_mass"]),
                        "output_l2": float(raw["output_l2"]),
                        "relative_output_l2": float(raw["relative_output_l2"]),
                        "output_cosine": float(raw["output_cosine"]),
                        "max_value_norm": float(raw["max_value_norm"]),
                        "mass_bound": float(raw["mass_bound"]),
                        "bound_satisfied": float(raw["bound_satisfied"]),
                    }
                )
    rows.sort(
        key=lambda row: (
            row["query_id"],
            row["layer"],
            row["query_head"],
            row["query_position"],
            row["action"],
        )
    )
    write_csv(output_dir / "distortion_rows.csv", rows)
    action_rows: list[dict[str, Any]] = []
    for action in sorted({row["action"] for row in rows}):
        group = [row for row in rows if row["action"] == action]
        action_rows.append(
            {
                "action": action,
                "rows": len(group),
                "mean_selected_blocks": statistics.fmean(row["selected_blocks"] for row in group),
                "mean_omitted_mass": statistics.fmean(row["omitted_mass"] for row in group),
                "p95_omitted_mass": float(np.quantile([row["omitted_mass"] for row in group], 0.95)),
                "mean_relative_output_l2": statistics.fmean(
                    row["relative_output_l2"] for row in group
                ),
                "p95_relative_output_l2": float(
                    np.quantile([row["relative_output_l2"] for row in group], 0.95)
                ),
                "mean_output_cosine": statistics.fmean(row["output_cosine"] for row in group),
                "bound_satisfaction_rate": statistics.fmean(
                    row["bound_satisfied"] for row in group
                ),
            }
        )
    write_csv(output_dir / "action_summary.csv", action_rows)

    deployable_actions = ["streaming", "lexical_blocks", "uniform", "qk_top_blocks"]
    costs = {"streaming": 2, "lexical_blocks": 8, "uniform": 8, "qk_top_blocks": 8, "full": 16}
    thresholds = [float(item) for item in args.relative_error_thresholds.split(",") if item]
    grouped: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["query_id"], row["layer"], row["query_head"], row["query_position"])
        grouped[key][row["action"]] = row
    compiled_rows: list[dict[str, Any]] = []
    stability_summary: dict[str, Any] = {}
    for threshold in thresholds:
        choices: dict[tuple[int, int], list[str]] = defaultdict(list)
        within_query_choices: dict[tuple[int, int, int], list[tuple[int, str]]] = defaultdict(list)
        for key, actions in grouped.items():
            feasible = [
                action
                for action in deployable_actions
                if actions[action]["relative_output_l2"] <= threshold
            ]
            chosen = min(
                feasible,
                key=lambda action: (
                    actions[action]["selected_blocks"],
                    actions[action]["relative_output_l2"],
                    action,
                ),
            ) if feasible else "full"
            query_id, layer, query_head, query_position = key
            choices[(layer, query_head)].append(chosen)
            within_query_choices[(query_id, layer, query_head)].append(
                (query_position, chosen)
            )
            compiled_rows.append(
                {
                    "threshold": threshold,
                    "query_id": query_id,
                    "layer": layer,
                    "query_head": query_head,
                    "query_position": query_position,
                    "chosen_action": chosen,
                    "selected_blocks": actions[chosen]["selected_blocks"],
                    "relative_output_l2": actions[chosen]["relative_output_l2"],
                    "omitted_mass": actions[chosen]["omitted_mass"],
                }
            )
        head_agreements: list[float] = []
        stable_heads = 0
        head_majority_actions: Counter[str] = Counter()
        for head_choices in choices.values():
            counts = Counter(head_choices)
            majority_action, majority_count = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            agreement = majority_count / len(head_choices)
            head_agreements.append(agreement)
            stable_heads += int(agreement >= 0.8)
            head_majority_actions[majority_action] += 1
        within_query_agreements: list[float] = []
        adjacent_matches: list[float] = []
        for positioned_choices in within_query_choices.values():
            ordered = [
                action for _, action in sorted(positioned_choices, key=lambda item: item[0])
            ]
            counts = Counter(ordered)
            within_query_agreements.append(max(counts.values()) / len(ordered))
            adjacent_matches.extend(
                float(left == right) for left, right in zip(ordered, ordered[1:])
            )
        current = [row for row in compiled_rows if row["threshold"] == threshold]
        stability_summary[str(threshold)] = {
            "action_counts": dict(sorted(Counter(row["chosen_action"] for row in current).items())),
            "mean_selected_blocks": statistics.fmean(row["selected_blocks"] for row in current),
            "mean_relative_output_l2": statistics.fmean(
                row["relative_output_l2"] for row in current
            ),
            "p95_relative_output_l2": float(
                np.quantile([row["relative_output_l2"] for row in current], 0.95)
            ),
            "mean_head_action_agreement": statistics.fmean(head_agreements),
            "heads_with_at_least_80pct_agreement": stable_heads,
            "heads_below_80pct_agreement": len(choices) - stable_heads,
            "head_majority_action_counts": dict(sorted(head_majority_actions.items())),
            "total_heads": len(choices),
            "within_query_groups": len(within_query_choices),
            "within_query_mean_action_agreement": statistics.fmean(
                within_query_agreements
            ),
            "within_query_adjacent_action_agreement": (
                statistics.fmean(adjacent_matches) if adjacent_matches else None
            ),
        }
    write_csv(output_dir / "compiled_actions.csv", compiled_rows)
    omitted = np.asarray([row["omitted_mass"] for row in rows])
    errors = np.asarray([row["output_l2"] for row in rows])
    summary = {
        "source": "merged exact head-distortion teacher",
        "queries": len({row["query_id"] for row in rows}),
        "layers": sorted({row["layer"] for row in rows}),
        "heads": len({(row["layer"], row["query_head"]) for row in rows}),
        "rows": len(rows),
        "action_summary": action_rows,
        "omitted_mass_output_l2_pearson": float(np.corrcoef(omitted, errors)[0, 1]),
        "risk_constrained_compilation": stability_summary,
        "note": (
            "Compiled actions choose the lowest-block deployable operator satisfying an exact "
            "per-head relative output-error threshold; full is the fallback."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
