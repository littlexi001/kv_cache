from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MARGIN_FIELDS = (
    "nine_newline_margin",
    "full_vocab_margin",
    "candidate_margin",
)
INTERNAL_FIELDS = (
    "critical_qk_weighted",
    "critical_pre_cosine_weighted",
    "critical_post_cosine_weighted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token-wise nine/newline boundary crossings and internal predictors."
    )
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stable-run", type=int, default=5)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--window-failure-rate", type=float, default=0.5)
    parser.add_argument("--sustained-horizon", type=int, default=256)
    parser.add_argument("--sustained-failure-rate", type=float, default=0.8)
    return parser.parse_args()


def rounded(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in (
                "added_tokens",
                "total_tokens",
                "added_token_id",
                "top_token_id",
            ):
                if row.get(field) not in ("", None):
                    row[field] = int(row[field])
            for field in (
                "nine_probability",
                "newline_probability",
                "gold_ppl",
                *MARGIN_FIELDS,
                *INTERNAL_FIELDS,
            ):
                row[field] = float(row[field])
            rows.append(row)
    if not rows:
        raise RuntimeError("points CSV is empty")
    return rows


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "added_tokens",
        "total_tokens",
        "added_token_text",
        "added_token_category",
        "nine_probability",
        "newline_probability",
        "nine_newline_margin",
        "gold_ppl",
        "full_vocab_margin",
        "top_token",
        "candidate_margin",
        *INTERNAL_FIELDS,
    )
    return {field: row[field] for field in fields}


def first_nonpositive(rows: list[dict[str, Any]], field: str) -> int | None:
    return next(
        (index for index, row in enumerate(rows) if float(row[field]) <= 0.0),
        None,
    )


def first_failure_run(
    rows: list[dict[str, Any]],
    field: str,
    run_length: int,
) -> int | None:
    run = 0
    for index, row in enumerate(rows):
        run = run + 1 if float(row[field]) <= 0.0 else 0
        if run >= run_length:
            return index - run_length + 1
    return None


def first_window_boundary(
    rows: list[dict[str, Any]],
    field: str,
    window: int,
    target_rate: float,
) -> tuple[int, float] | None:
    failures = [int(float(row[field]) <= 0.0) for row in rows]
    running = sum(failures[:window])
    if len(rows) >= window and running / window >= target_rate:
        return 0, running / window
    for end in range(window, len(rows)):
        running += failures[end] - failures[end - window]
        if running / window >= target_rate:
            return end - window + 1, running / window
    return None


def first_sustained_boundary(
    rows: list[dict[str, Any]],
    field: str,
    horizon: int,
    target_rate: float,
) -> tuple[int, float] | None:
    failures = np.asarray(
        [float(row[field]) <= 0.0 for row in rows],
        dtype=np.float64,
    )
    if len(failures) < horizon:
        return None
    kernel = np.ones(horizon, dtype=np.float64)
    rates = np.convolve(failures, kernel, mode="valid") / horizon
    indices = np.flatnonzero(rates >= target_rate)
    if not len(indices):
        return None
    index = int(indices[0])
    return index, float(rates[index])


def crossing_count(rows: list[dict[str, Any]], field: str) -> int:
    states = [float(row[field]) > 0.0 for row in rows]
    return sum(left != right for left, right in zip(states, states[1:]))


def best_univariate_threshold(
    values: np.ndarray,
    failures: np.ndarray,
) -> dict[str, Any]:
    order = np.argsort(values, kind="stable")
    x = values[order]
    y = failures[order].astype(np.int64)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return {
            "available": False,
            "reason": "both success and failure samples are required",
        }
    positive_prefix = np.concatenate(([0], np.cumsum(y)))
    negative_prefix = np.arange(len(y) + 1) - positive_prefix
    candidates: list[tuple[float, int, str]] = []
    for split in range(len(y) + 1):
        # Rule A: x <= threshold predicts failure.
        true_positive = int(positive_prefix[split])
        true_negative = negatives - int(negative_prefix[split])
        balanced = 0.5 * (
            true_positive / positives + true_negative / negatives
        )
        candidates.append((balanced, split, "failure_if_le"))
        # Rule B: x >= threshold predicts failure.
        true_positive = positives - int(positive_prefix[split])
        true_negative = int(negative_prefix[split])
        balanced = 0.5 * (
            true_positive / positives + true_negative / negatives
        )
        candidates.append((balanced, split, "failure_if_ge"))
    balanced, split, direction = max(candidates, key=lambda item: item[0])
    if split == 0:
        threshold = float(x[0]) - np.finfo(np.float64).eps
    elif split == len(x):
        threshold = float(x[-1]) + np.finfo(np.float64).eps
    else:
        threshold = 0.5 * float(x[split - 1] + x[split])
    return {
        "available": True,
        "direction": direction,
        "threshold": rounded(threshold, 10),
        "balanced_accuracy": rounded(balanced),
        "failure_samples": positives,
        "success_samples": negatives,
    }


def linear_relation(values: np.ndarray, margins: np.ndarray) -> dict[str, Any]:
    design = np.column_stack((np.ones(len(values)), values))
    coefficients, *_ = np.linalg.lstsq(design, margins, rcond=None)
    prediction = design @ coefficients
    residual = float(np.sum((margins - prediction) ** 2))
    total = float(np.sum((margins - margins.mean()) ** 2))
    slope = float(coefficients[1])
    threshold = (
        float(-coefficients[0] / slope)
        if abs(slope) > np.finfo(np.float64).eps
        else None
    )
    correlation = float(np.corrcoef(values, margins)[0, 1])
    return {
        "pearson_r": rounded(correlation),
        "r_squared": rounded(1.0 - residual / total) if total > 0 else None,
        "intercept": rounded(coefficients[0]),
        "slope": rounded(slope),
        "linear_zero_crossing": (
            rounded(threshold, 10) if threshold is not None else None
        ),
    }


def multivariate_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.asarray(
        [[float(row[field]) for field in INTERNAL_FIELDS] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [float(row["nine_newline_margin"]) for row in rows],
        dtype=np.float64,
    )
    means = x.mean(axis=0)
    standard_deviations = x.std(axis=0)
    nonzero = standard_deviations > 0
    z = np.zeros_like(x)
    z[:, nonzero] = (x[:, nonzero] - means[nonzero]) / standard_deviations[nonzero]
    design = np.column_stack((np.ones(len(z)), z))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "target": "nine_newline_margin",
        "r_squared": rounded(1.0 - residual / total) if total > 0 else None,
        "intercept": rounded(coefficients[0]),
        "standardized_coefficients": {
            field: rounded(value)
            for field, value in zip(INTERNAL_FIELDS, coefficients[1:])
        },
    }


def category_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for previous, current in zip(rows, rows[1:]):
        grouped[str(current["added_token_category"])].append(
            float(current["nine_newline_margin"])
            - float(previous["nine_newline_margin"])
        )
    output: dict[str, Any] = {}
    for category, deltas in sorted(grouped.items()):
        output[category] = {
            "count": len(deltas),
            "mean_delta_margin": rounded(statistics.fmean(deltas)),
            "median_delta_margin": rounded(statistics.median(deltas)),
            "mean_absolute_delta": rounded(
                statistics.fmean(abs(value) for value in deltas)
            ),
        }
    return output


def boundary_record(
    rows: list[dict[str, Any]],
    index: int | None,
    *,
    radius: int = 8,
) -> dict[str, Any] | None:
    if index is None:
        return None
    return {
        "index": index,
        "point": compact_row(rows[index]),
        "local_points": [
            compact_row(row)
            for row in rows[max(0, index - radius) : min(len(rows), index + radius + 1)]
        ],
    }


def build_report(result: dict[str, Any]) -> str:
    primary = result["boundaries"]["nine_newline_margin"]
    lines = [
        "# 136K→144K 逐 token：nine 被 newline 夺走主导权",
        "",
        "## 一眼结论",
        "",
        f"- 已分析 **{result['point_count']}** 个连续长度点，范围 "
        f"`{result['start_total_tokens']}`→`{result['end_total_tokens']}`。",
        f"- `nine` 相对 `newline` 的胜负翻转共 **{primary['crossing_count']}** 次；"
        "因此不能只把第一次翻转当作永久失败边界。",
    ]
    first = primary["first_nonpositive"]
    stable = primary["first_stable_run"]
    sustained = primary["first_sustained_window"]
    if first:
        point = first["point"]
        lines.append(
            f"- 首次零点/失败在新增 **{point['added_tokens']}** token："
            f"margin={point['nine_newline_margin']:.4f}，"
            f"`P(nine)={point['nine_probability']:.2%}`，"
            f"`P(newline)={point['newline_probability']:.2%}`。"
        )
    if stable:
        point = stable["point"]
        lines.append(
            f"- 首次连续失败段从新增 **{point['added_tokens']}** token 开始。"
        )
    if sustained:
        point = sustained["point"]
        lines.append(
            f"- 首个高失败率持续窗口从新增 **{point['added_tokens']}** token 开始。"
        )
    lines.extend(
        [
            "",
            "严格来说，输出切换的直接阈值就是",
            "",
            "$$",
            "\\Delta_{\\mathrm{out}}="
            "z_{\\mathrm{nine}}-z_{\\mathrm{newline}}=0.",
            "$$",
            "",
            "但实验真正检验的是：某一个内部指标是否也存在稳定阈值，"
            "足以预测这个零点。若单指标的 balanced accuracy 不高，"
            "就说明主导权不是由一个 head 或一个 QK 数值单独决定，"
            "而是多层残差累积后的联合结果。",
            "",
            "## 内部指标的单变量阈值",
            "",
            "| 指标 | 规则 | 阈值 | Balanced accuracy | 与输出 margin 的 r |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for field, data in result["internal_predictors"].items():
        threshold = data["classification_threshold"]
        relation = data["linear_relation"]
        if threshold["available"]:
            lines.append(
                f"| `{field}` | `{threshold['direction']}` | "
                f"{threshold['threshold']:.6f} | "
                f"{threshold['balanced_accuracy']:.3f} | "
                f"{relation['pearson_r']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 不同新增 token 类型造成的单步扰动",
            "",
            "| token 类别 | 次数 | 平均 Δmargin | 中位数 Δmargin | 平均绝对扰动 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, data in result["category_effects"].items():
        lines.append(
            f"| `{category}` | {data['count']} | "
            f"{data['mean_delta_margin']:+.4f} | "
            f"{data['median_delta_margin']:+.4f} | "
            f"{data['mean_absolute_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判定定义",
            "",
            f"- 瞬时边界：margin 首次 $\\le 0$。",
            f"- 稳定短段：连续 **{result['settings']['stable_run']}** 点 margin $\\le 0$。",
            f"- 统计窗口：连续 **{result['settings']['window']}** 点中失败率达到 "
            f"**{result['settings']['window_failure_rate']:.0%}**。",
            f"- 持续窗口：后续 **{result['settings']['sustained_horizon']}** 点中失败率达到 "
            f"**{result['settings']['sustained_failure_rate']:.0%}**。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.points_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries: dict[str, Any] = {}
    for field in MARGIN_FIELDS:
        first = first_nonpositive(rows, field)
        stable = first_failure_run(rows, field, args.stable_run)
        window = first_window_boundary(
            rows,
            field,
            args.window,
            args.window_failure_rate,
        )
        sustained = first_sustained_boundary(
            rows,
            field,
            args.sustained_horizon,
            args.sustained_failure_rate,
        )
        boundaries[field] = {
            "failure_fraction": rounded(
                statistics.fmean(float(row[field]) <= 0.0 for row in rows)
            ),
            "crossing_count": crossing_count(rows, field),
            "first_nonpositive": boundary_record(rows, first),
            "first_stable_run": boundary_record(rows, stable),
            "first_statistical_window": (
                {
                    **(boundary_record(rows, window[0]) or {}),
                    "observed_failure_rate": rounded(window[1]),
                }
                if window
                else None
            ),
            "first_sustained_window": (
                {
                    **(boundary_record(rows, sustained[0]) or {}),
                    "observed_failure_rate": rounded(sustained[1]),
                }
                if sustained
                else None
            ),
        }

    output_margin = np.asarray(
        [float(row["nine_newline_margin"]) for row in rows],
        dtype=np.float64,
    )
    failures = output_margin <= 0.0
    predictors: dict[str, Any] = {}
    for field in INTERNAL_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        predictors[field] = {
            "classification_threshold": best_univariate_threshold(values, failures),
            "linear_relation": linear_relation(values, output_margin),
        }

    result = {
        "schema_version": 1,
        "point_count": len(rows),
        "start_total_tokens": rows[0]["total_tokens"],
        "end_total_tokens": rows[-1]["total_tokens"],
        "settings": {
            "stable_run": args.stable_run,
            "window": args.window,
            "window_failure_rate": args.window_failure_rate,
            "sustained_horizon": args.sustained_horizon,
            "sustained_failure_rate": args.sustained_failure_rate,
        },
        "boundaries": boundaries,
        "internal_predictors": predictors,
        "combined_internal_linear_model": multivariate_relation(rows),
        "category_effects": category_effects(rows),
    }
    write_json(output_dir / "analysis.json", result)
    (output_dir / "report.md").write_text(
        build_report(result),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
