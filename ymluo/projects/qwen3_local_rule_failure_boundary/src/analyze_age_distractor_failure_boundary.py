from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


def rounded(value: float, digits: int = 10) -> float:
    return float(f"{value:.{digits}g}")


def mean_head_scalar(values: Sequence[Sequence[float]], layers: range | None = None) -> float:
    layer_indices = layers if layers is not None else range(len(values))
    return statistics.fmean(
        float(values[layer][head])
        for layer in layer_indices
        for head in range(len(values[layer]))
    )


def mean_head_category(
    values: Sequence[Sequence[Sequence[float]]],
    category_index: int,
    layers: range | None = None,
) -> float:
    layer_indices = layers if layers is not None else range(len(values))
    return statistics.fmean(
        float(values[layer][head][category_index])
        for layer in layer_indices
        for head in range(len(values[layer]))
    )


def load_series(input_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for completed in manifest["completed"]:
        payload = json.loads((input_dir / completed["file"]).read_text(encoding="utf-8"))
        attention = payload.get("attention")
        row: dict[str, Any] = {
            "series": input_dir.name,
            **completed,
            "filler_count": payload["case"]["filler_count"],
            "category_counts": payload["case"]["category_counts"],
            "prompt_sha256": payload["case"]["prompt_sha256"],
            "answer_only": bool(payload.get("answer_only", False)),
        }
        if attention is not None:
            order = attention["category_order"]
            for category_index, category in enumerate(order):
                row[f"{category}_mass"] = rounded(
                    mean_head_category(attention["head_category_mass"], category_index)
                )
                row[f"{category}_mean_attention"] = rounded(
                    mean_head_category(
                        attention["head_category_mean_attention"],
                        category_index,
                    )
                )
                row[f"{category}_enrichment"] = rounded(
                    mean_head_category(
                        attention["head_category_enrichment"],
                        category_index,
                    )
                )
                row[f"{category}_mean_logit"] = rounded(
                    mean_head_category(
                        attention["head_category_mean_logit"],
                        category_index,
                    )
                )
                for block, layers in (
                    ("early", range(0, 12)),
                    ("middle", range(12, 28)),
                    ("late", range(28, 36)),
                ):
                    row[f"{block}_{category}_mass"] = rounded(
                        mean_head_category(
                            attention["head_category_mass"],
                            category_index,
                            layers,
                        )
                    )
                    row[f"{block}_{category}_mean_logit"] = rounded(
                        mean_head_category(
                            attention["head_category_mean_logit"],
                            category_index,
                            layers,
                        )
                    )
            row["mean_head_logsumexp"] = rounded(
                mean_head_scalar(attention["head_logsumexp"])
            )
            row["mean_head_entropy"] = rounded(
                mean_head_scalar(attention["head_entropy"])
            )
            row["partition_mass_sum"] = rounded(
                sum(row[f"{category}_mass"] for category in order)
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row if key != "category_counts"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in keys
                }
            )


def transition_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    transitions = []
    for left, right in zip(rows, rows[1:]):
        if bool(left["full_vocab_correct"]) != bool(right["full_vocab_correct"]):
            transitions.append(
                {
                    "left": {
                        "total_tokens": left["total_tokens"],
                        "distractor_count": left["distractor_count"],
                        "margin": left["full_vocab_margin"],
                        "correct": left["full_vocab_correct"],
                    },
                    "right": {
                        "total_tokens": right["total_tokens"],
                        "distractor_count": right["distractor_count"],
                        "margin": right["full_vocab_margin"],
                        "correct": right["full_vocab_correct"],
                    },
                }
            )
    return transitions


def format_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Tokens | Distractors | Filler | Gold PPL | Margin | Top-1 | Gold age mass | Distractor age mass | Query mass |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        def pct(key: str) -> str:
            value = row.get(key)
            return "—" if value is None else f"{100 * float(value):.4f}%"

        lines.append(
            f"| {int(row['total_tokens']):,} | {int(row['distractor_count']):,} | "
            f"{int(row['filler_count']):,} | {float(row['gold_ppl']):.4f} | "
            f"{float(row['full_vocab_margin']):+.4f} | `{row['top_token']}` | "
            f"{pct('gold_age_mass')} | {pct('distractor_ages_mass')} | {pct('query_mass')} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    series: dict[str, list[dict[str, Any]]],
    all_rows: Sequence[dict[str, Any]],
) -> None:
    minimum = min(all_rows, key=lambda row: float(row["full_vocab_margin"]))
    failures = [row for row in all_rows if not bool(row["full_vocab_correct"])]
    lines = [
        "# 年龄干扰长度 × 数量失败边界扫描",
        "",
        "## 当前边界",
        "",
        (
            f"- 首次观测失败：{failures[0]['total_tokens']:,} tokens / "
            f"{failures[0]['distractor_count']:,} 条干扰。"
            if failures
            else "- 当前已完成点中尚未出现完整词表 top-1 失败。"
        ),
        (
            f"- 最小 margin：{float(minimum['full_vocab_margin']):+.4f}，"
            f"位于 {int(minimum['total_tokens']):,} tokens / "
            f"{int(minimum['distractor_count']):,} 条干扰。"
        ),
        "",
    ]
    for name, rows in series.items():
        lines.extend(
            [
                f"## {name}",
                "",
                format_table(rows),
                "",
            ]
        )
    lines.extend(
        [
            "## 指标定义",
            "",
            "- `Margin = log p(nine) - log p(最强非 nine token)`；小于等于 0 表示完整词表 top-1 失败。",
            "- 六个 attention 类别互斥并覆盖全部 prompt token，因此每个 Head 的类别 mass 之和为 1。",
            "- `Gold age mass` 只统计证据中的单 token `nine`。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    series: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for raw in args.input_dir:
        input_dir = Path(raw)
        rows = load_series(input_dir)
        series[input_dir.name] = rows
        all_rows.extend(rows)

    write_csv(output_dir / "boundary_points.csv", all_rows)
    summary = {
        "schema_version": 1,
        "point_count": len(all_rows),
        "failure_count": sum(not bool(row["full_vocab_correct"]) for row in all_rows),
        "minimum_margin_point": min(
            all_rows,
            key=lambda row: float(row["full_vocab_margin"]),
        ),
        "series": {
            name: {
                "point_count": len(rows),
                "transitions": transition_rows(rows),
            }
            for name, rows in series.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "boundary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", series, all_rows)


if __name__ == "__main__":
    main()
