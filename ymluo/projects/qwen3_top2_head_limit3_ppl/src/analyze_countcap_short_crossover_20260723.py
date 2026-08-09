from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SPARSE_METHODS = ("countcap_fullprompt", "countcap_fullprompt_keypca")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the measured dense/CountCap short-context crossover."
    )
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--quality_floor", type=float, default=0.95)
    parser.add_argument("--speed_margin", type=float, default=1.03)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def macro_score(rows: list[dict[str, str]]) -> float:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(float(row["score"]))
    return statistics.mean(
        statistics.mean(scores) for scores in by_task.values()
    )


def paired_rows(
    rows: list[dict[str, str]], method: str
) -> list[tuple[dict[str, str], dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["task"], row["sample_id"])][row["method"]] = row
    return [
        (methods["full_kv"], methods[method])
        for methods in grouped.values()
        if "full_kv" in methods and method in methods
    ]


def ratio_of_sums(
    pairs: list[tuple[dict[str, str], dict[str, str]]], key: str
) -> float:
    denominator = sum(float(candidate[key]) for _, candidate in pairs)
    if denominator <= 0.0:
        return 0.0
    return sum(float(full[key]) for full, _ in pairs) / denominator


def median_decode_forward_ms(rows: list[dict[str, str]]) -> float | None:
    values = []
    for row in rows:
        generated = int(row["generated_tokens"])
        if generated > 1:
            values.append(1000.0 * float(row["decode_seconds"]) / (generated - 1))
    return statistics.median(values) if values else None


def fit_decode_cost(rows: list[dict[str, str]]) -> dict[str, float | None]:
    points = [
        (max(0, int(row["generated_tokens"]) - 1), float(row["decode_seconds"]))
        for row in rows
    ]
    if not points:
        return {
            "fixed_seconds": None,
            "step_seconds": None,
            "r_squared": None,
        }
    x_mean = statistics.mean(x for x, _ in points)
    y_mean = statistics.mean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0.0:
        positive = [y / x for x, y in points if x > 0]
        slope = statistics.median(positive) if positive else 0.0
        intercept = max(0.0, y_mean - slope * x_mean)
        r_squared = None
    else:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
        slope = max(0.0, slope)
        intercept = max(0.0, y_mean - slope * x_mean)
        predicted = [intercept + slope * x for x, _ in points]
        residual = sum((y - estimate) ** 2 for (_, y), estimate in zip(points, predicted))
        total = sum((y - y_mean) ** 2 for _, y in points)
        r_squared = 1.0 - residual / total if total > 0.0 else None
    return {
        "fixed_seconds": intercept,
        "step_seconds": slope,
        "r_squared": r_squared,
    }


def break_even_steps(
    dense: dict[str, float | None], sparse: dict[str, float | None]
) -> int | None:
    dense_fixed = dense["fixed_seconds"]
    dense_step = dense["step_seconds"]
    sparse_fixed = sparse["fixed_seconds"]
    sparse_step = sparse["step_seconds"]
    if None in {dense_fixed, dense_step, sparse_fixed, sparse_step}:
        return None
    assert dense_fixed is not None and dense_step is not None
    assert sparse_fixed is not None and sparse_step is not None
    if sparse_step >= dense_step:
        return None
    required = (sparse_fixed - dense_fixed) / (dense_step - sparse_step)
    return max(0, math.ceil(required))


def summarize_length(length_dir: Path) -> dict[str, Any]:
    rows = read_rows(length_dir / "sample_results.csv")
    full_rows = [row for row in rows if row["method"] == "full_kv"]
    full_macro = macro_score(full_rows)
    result: dict[str, Any] = {
        "configured_max_tokens": int(length_dir.name.removeprefix("length")),
        "mean_prompt_tokens": statistics.mean(
            int(row["prompt_tokens"]) for row in full_rows
        ),
        "samples": len(full_rows),
        "full_macro_score": full_macro,
        "full_decode_ms_per_forward": median_decode_forward_ms(full_rows),
        "full_decode_cost_model": fit_decode_cost(full_rows),
        "methods": {},
    }
    for method in SPARSE_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        pairs = paired_rows(rows, method)
        score = macro_score(method_rows)
        cost_model = fit_decode_cost(method_rows)
        result["methods"][method] = {
            "macro_score": score,
            "quality_retention": score / full_macro if full_macro > 0.0 else None,
            "query_speedup": ratio_of_sums(pairs, "query_seconds"),
            "decode_speedup": ratio_of_sums(pairs, "decode_seconds"),
            "online_speedup": ratio_of_sums(pairs, "online_seconds"),
            "total_speedup": ratio_of_sums(pairs, "total_seconds"),
            "decode_ms_per_forward": median_decode_forward_ms(method_rows),
            "decode_cost_model": cost_model,
            "break_even_generated_forwards": break_even_steps(
                result["full_decode_cost_model"], cost_model
            ),
        }
    return result


def choose_path(
    row: dict[str, Any], quality_floor: float, speed_margin: float
) -> tuple[str, float]:
    eligible = []
    for method, metrics in row["methods"].items():
        retention = metrics["quality_retention"]
        speed = metrics["online_speedup"]
        if retention is not None and retention >= quality_floor and speed >= speed_margin:
            eligible.append((speed, method))
    if not eligible:
        return "full_kv", 1.0
    speed, method = max(eligible)
    return method, speed


def estimate_crossover(
    rows: list[dict[str, Any]], quality_floor: float
) -> float | None:
    points = []
    for row in rows:
        best_speed = max(
            (
                metrics["online_speedup"]
                for metrics in row["methods"].values()
                if metrics["quality_retention"] is not None
                and metrics["quality_retention"] >= quality_floor
            ),
            default=0.0,
        )
        points.append((float(row["mean_prompt_tokens"]), best_speed))
    for (n0, s0), (n1, s1) in zip(points, points[1:]):
        if s0 < 1.0 <= s1 and s0 > 0.0 and n1 > n0:
            x0, x1 = math.log(n0), math.log(n1)
            y0, y1 = math.log(s0), math.log(s1)
            if not math.isclose(y0, y1):
                return math.exp(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))
    return None


def fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    quality_floor: float,
    speed_margin: float,
    crossover: float | None,
) -> None:
    lines = [
        "# CountCap 短序列交叉点实测",
        "",
        f"质量门槛：Full KV 的 {quality_floor:.0%}；速度切换余量：{speed_margin:.2f}x。",
        "",
        "| 实际平均长度 | Full 分数 | QK-metric 保持率 | QK-metric online | Key-PCA 保持率 | Key-PCA online | 推荐路径 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        qk = row["methods"]["countcap_fullprompt"]
        kp = row["methods"]["countcap_fullprompt_keypca"]
        selected, _ = choose_path(row, quality_floor, speed_margin)
        lines.append(
            "| {n:.0f} | {full:.4f} | {qr} | {qs}x | {kr} | {ks}x | {selected} |".format(
                n=row["mean_prompt_tokens"],
                full=row["full_macro_score"],
                qr=fmt(qk["quality_retention"]),
                qs=fmt(qk["online_speedup"]),
                kr=fmt(kp["quality_retention"]),
                ks=fmt(kp["online_speedup"]),
                selected=selected,
            )
        )
    lines.extend(["", "## 解析式门控", ""])
    if crossover is None:
        lines.append("当前测量区间内没有同时满足质量和速度条件的稀疏交叉点。")
    else:
        lines.append(
            f"按相邻实测点的对数插值，速度交叉点约为 **{crossover:.0f} tokens**。"
        )
    lines.extend(
        [
            "",
            "运行时只在质量合格且实测 T_build + G*T_sparse 小于 G*T_dense 时启用 CountCap；其余情况使用 Full SDPA。",
            "",
            "这里的门控只使用长度、预计剩余生成步数和硬件实测耗时，不训练 router。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    length_dirs = sorted(
        args.run_root.glob("length*"),
        key=lambda path: int(path.name.removeprefix("length")),
    )
    rows = [summarize_length(path) for path in length_dirs]
    for row in rows:
        selected, speed = choose_path(row, args.quality_floor, args.speed_margin)
        row["recommended_path"] = selected
        row["recommended_online_speedup"] = speed
    crossover = estimate_crossover(rows, args.quality_floor)
    output = {
        "quality_floor": args.quality_floor,
        "speed_margin": args.speed_margin,
        "estimated_speed_crossover_tokens": crossover,
        "lengths": rows,
    }
    (args.run_root / "crossover_analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(
        args.run_root / "crossover_analysis_zh.md",
        rows,
        args.quality_floor,
        args.speed_margin,
        crossover,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
