from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASELINE_METHOD = "fixed_alpha_1"
METHODS = (
    "fixed_alpha_1",
    "block_residual_mean_proxy",
    "block_conditional_residual_proxy_d8",
)
CASES = (
    "narrative32k",
    "narrative64k",
    "narrative128k",
    "lcc64k",
    "qmsum64k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--candidate_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_markdown", type=Path, required=True)
    return parser.parse_args()


def load_method(path: Path, method: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in payload["summary"]
        if row["candidate_mode"] == "proxy" and row["method"] == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {method} row in {path}, got {len(matches)}")
    return matches[0]


def projected_mean(path: Path, method: str) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row["projected_relative_l2"])
        for row in rows
        if row["candidate_mode"] == "proxy" and row["method"] == method
    ]
    if not values:
        raise RuntimeError(f"no projected rows for {method} in {path}")
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    baseline: dict[str, dict[str, float]] = {}
    for case_name in CASES:
        case_root = args.baseline_root / "analysis" / case_name
        row = load_method(case_root / "summary.json", BASELINE_METHOD)
        baseline[case_name] = {
            "relative_l2_mean": float(row["relative_l2_mean"]),
            "relative_l2_p90": float(row["relative_l2_p90"]),
            "projected_relative_l2_mean": projected_mean(
                case_root / "per_layer_output.csv", BASELINE_METHOD
            ),
        }

    rows: list[dict[str, Any]] = []
    for budget_root in sorted(args.candidate_root.glob("k[0-9]*")):
        if not budget_root.is_dir():
            continue
        top_k = int(budget_root.name[1:])
        for case_name in CASES:
            case_root = budget_root / case_name
            for method in METHODS:
                row = load_method(case_root / "summary.json", method)
                base = baseline[case_name]
                mean = float(row["relative_l2_mean"])
                p90 = float(row["relative_l2_p90"])
                projected = projected_mean(
                    case_root / "per_layer_output.csv", method
                )
                rows.append(
                    {
                        "top_k": top_k,
                        "case": case_name,
                        "method": method,
                        "relative_l2_mean": mean,
                        "relative_l2_p90": p90,
                        "projected_relative_l2_mean": projected,
                        "mean_ratio_vs_top1280": mean
                        / base["relative_l2_mean"],
                        "p90_ratio_vs_top1280": p90
                        / base["relative_l2_p90"],
                        "projected_ratio_vs_top1280": projected
                        / base["projected_relative_l2_mean"],
                        "passes_case": (
                            mean <= base["relative_l2_mean"]
                            and p90 <= base["relative_l2_p90"]
                            and projected
                            <= 1.05 * base["projected_relative_l2_mean"]
                        ),
                    }
                )

    configurations: list[dict[str, Any]] = []
    for top_k in sorted({int(row["top_k"]) for row in rows}):
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["top_k"] == top_k and row["method"] == method
            ]
            configurations.append(
                {
                    "top_k": top_k,
                    "method": method,
                    "cases": len(selected),
                    "passing_cases": sum(bool(row["passes_case"]) for row in selected),
                    "all_cases_pass": all(bool(row["passes_case"]) for row in selected),
                    "mean_ratio_vs_top1280": sum(
                        float(row["mean_ratio_vs_top1280"]) for row in selected
                    )
                    / len(selected),
                    "worst_mean_ratio_vs_top1280": max(
                        float(row["mean_ratio_vs_top1280"]) for row in selected
                    ),
                    "worst_p90_ratio_vs_top1280": max(
                        float(row["p90_ratio_vs_top1280"]) for row in selected
                    ),
                    "worst_projected_ratio_vs_top1280": max(
                        float(row["projected_ratio_vs_top1280"])
                        for row in selected
                    ),
                }
            )

    report = {
        "schema": "qksieve_conditional_residual_budget_v1",
        "baseline": {
            "top_k": 1280,
            "method": BASELINE_METHOD,
            "cases": baseline,
        },
        "rows": rows,
        "configurations": configurations,
        "claim_boundary": (
            "Real-QKV local and W_o-projected output errors only; a passing "
            "configuration still requires model-level PPL and CUDA timing."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 条件残差低预算结果",
        "",
        "参照是 top-1280 的当前全局 ValueSketch。比值小于 1 表示误差更低。",
        "",
        "| top-k | 方法 | 通过任务 | 平均误差比 | 最坏 mean 比 | 最坏 P90 比 | 最坏 W_o 比 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in configurations:
        lines.append(
            "| {top_k} | {method} | {passing_cases}/{cases} | "
            "{mean_ratio_vs_top1280:.3f} | {worst_mean_ratio_vs_top1280:.3f} | "
            "{worst_p90_ratio_vs_top1280:.3f} | "
            "{worst_projected_ratio_vs_top1280:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "该表只支持局部 attention 输出结论，不等同于 PPL、LongBench 或速度结论。",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
