from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries: list[tuple[int, dict[str, Any]]] = []
    for path in root.glob("gpus*/summary.json"):
        match = re.fullmatch(r"gpus(\d+)", path.parent.name)
        if match:
            summaries.append((int(match.group(1)), json.loads(path.read_text(encoding="utf-8"))))
    summaries.sort(key=lambda item: item[0])
    if not summaries:
        raise FileNotFoundError(f"No gpus*/summary.json under {root}")

    baseline: dict[str, float] = {}
    for gpu_count, summary in summaries:
        if gpu_count == 1:
            baseline = {row["method"]: float(row["median_seconds"]) for row in summary["methods"]}
            break
    rows: list[dict[str, Any]] = []
    for gpu_count, summary in summaries:
        for method in summary["methods"]:
            seconds = float(method["median_seconds"])
            base = baseline.get(method["method"], seconds)
            speedup = base / seconds
            rows.append(
                {
                    "gpus": gpu_count,
                    "method": method["method"],
                    "median_seconds": seconds,
                    "speedup_vs_1gpu": speedup,
                    "parallel_efficiency": speedup / gpu_count,
                    "queries_per_second": method["queries_per_second"],
                    "scanned_tokens_per_second": method["scanned_tokens_per_second"],
                    "oracle_block_recall": method["oracle_block_recall"],
                    "exact_block_mass_recall": method["exact_block_mass_recall"],
                    "answer_block_recall": method["answer_block_recall"],
                    "answer_block_mrr": method["answer_block_mrr"],
                }
            )

    fields = list(rows[0])
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 真实 Q/K 检索多卡结果",
        "",
        "全部向量来自 Qwen3-0.6B 对真实 LongBench 文本的前向过程；没有使用合成高斯向量。",
        "",
        "| GPU | 方法 | 时间(s) | 相对 1 卡加速 | 并行效率 | Full128 reference recall | Full128 block-mass proxy | Answer block recall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {gpus} | {method} | {median_seconds:.4f} | {speedup_vs_1gpu:.2f}x | "
            "{parallel_efficiency:.2%} | {oracle_block_recall:.2%} | "
            "{exact_block_mass_recall:.2%} | {answer_block_recall:.2%} |".format(**row)
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
