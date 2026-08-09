from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


Case = dict[str, Any]
RiskFunction = Callable[[Case], float]


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(part) for part in value.split(",") if part.strip()})
    if not values:
        raise ValueError("expected at least one float")
    return values


def read_method_rows(
    path: Path, base_method: str, enhanced_method: str
) -> list[Case]:
    paired: dict[tuple[str, int, int, int], dict[str, dict[str, str]]] = defaultdict(
        dict
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if method not in {base_method, enhanced_method}:
                continue
            key = (
                row["topic"],
                int(row["record_index"]),
                int(row["layer"]),
                int(row["query_head"]),
            )
            paired[key][method] = row

    cases: list[Case] = []
    required_features = (
        "query_energy_coverage",
        "base_proxy_top2_mass",
        "base_tail_score_std",
        "base_margin_to_2k_sigma",
        "base_margin_to_4k_sigma",
        "base_boundary_band_1sigma_ratio",
    )
    metrics = ("top2_attention_mass_recall", "top2_recall")
    for key, methods in paired.items():
        if set(methods) != {base_method, enhanced_method}:
            raise ValueError(f"unpaired methods for case {key}")
        base = methods[base_method]
        enhanced = methods[enhanced_method]
        case: Case = {
            "topic": key[0],
            "record_index": key[1],
            "layer": key[2],
            "query_head": key[3],
        }
        case.update({name: float(base[name]) for name in required_features})
        for metric in metrics:
            case[f"base_{metric}"] = float(base[metric])
            case[f"enhanced_{metric}"] = float(enhanced[metric])
        cases.append(case)
    if not cases:
        raise ValueError(f"no paired rows found in {path}")
    return cases


RISK_FUNCTIONS: dict[str, RiskFunction] = {
    "residual_energy": lambda case: 1.0 - case["query_energy_coverage"],
    "proxy_diffuse": lambda case: -case["base_proxy_top2_mass"],
    "boundary_band": lambda case: case["base_boundary_band_1sigma_ratio"],
    "inverse_margin_2k": lambda case: -case["base_margin_to_2k_sigma"],
    "inverse_margin_4k": lambda case: -case["base_margin_to_4k_sigma"],
    "energy_x_boundary": lambda case: (
        (1.0 - case["query_energy_coverage"])
        * case["base_boundary_band_1sigma_ratio"]
    ),
    "oracle_gain": lambda case: (
        case["enhanced_top2_attention_mass_recall"]
        - case["base_top2_attention_mass_recall"]
    ),
}


def routed_case_indices(
    cases: list[Case], route_fraction: float, risk_function: RiskFunction
) -> set[int]:
    if not 0.0 <= route_fraction <= 1.0:
        raise ValueError("route_fraction must be in [0, 1]")
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        groups[(case["topic"], case["record_index"], case["layer"])].append(index)

    selected: set[int] = set()
    for indices in groups.values():
        count = min(len(indices), math.ceil(route_fraction * len(indices)))
        if count == 0:
            continue
        ranked = sorted(indices, key=lambda index: risk_function(cases[index]), reverse=True)
        selected.update(ranked[:count])
    return selected


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_policy(
    cases: list[Case],
    policy: str,
    route_fraction: float,
    base_rank: int,
    enhanced_rank: int,
) -> dict[str, Any]:
    selected = routed_case_indices(cases, route_fraction, RISK_FUNCTIONS[policy])
    output: dict[str, Any] = {
        "policy": policy,
        "route_fraction_target": route_fraction,
        "route_fraction_actual": len(selected) / len(cases),
        "average_rank": (
            base_rank
            + (enhanced_rank - base_rank) * len(selected) / len(cases)
        ),
        "cases": len(cases),
    }
    for metric in ("top2_attention_mass_recall", "top2_recall"):
        values = [
            case[f"enhanced_{metric}"] if index in selected else case[f"base_{metric}"]
            for index, case in enumerate(cases)
        ]
        output[f"{metric}_mean"] = sum(values) / len(values)
        output[f"{metric}_p10"] = quantile(values, 0.1)
        output[f"{metric}_minimum"] = min(values)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare training-free anytime-PCA residual routing policies."
    )
    parser.add_argument("--input_csv", required=True, type=Path, nargs="+")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--base_method", default="fixed_rank_64")
    parser.add_argument("--enhanced_method", default="fixed_rank_96")
    parser.add_argument("--base_rank", type=int, default=64)
    parser.add_argument("--enhanced_rank", type=int, default=96)
    parser.add_argument("--route_fractions", default="0,0.25,0.5,0.75,1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases: list[Case] = []
    for path in args.input_csv:
        cases.extend(read_method_rows(path, args.base_method, args.enhanced_method))
    fractions = parse_float_list(args.route_fractions)
    if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
        raise ValueError("route fractions must be in [0, 1]")

    rows = [
        evaluate_policy(
            cases,
            policy,
            fraction,
            args.base_rank,
            args.enhanced_rank,
        )
        for fraction in fractions
        for policy in RISK_FUNCTIONS
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "policy_summary.csv", rows)
    report = {
        "input_csv": [str(path) for path in args.input_csv],
        "base_method": args.base_method,
        "enhanced_method": args.enhanced_method,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
