from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from analyze_countcap_short_crossover_20260723 import (
    break_even_steps,
    fit_decode_cost,
)


METHOD = "countcap_fullprompt_keypca"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ratio(full: dict[str, str], sparse: dict[str, str], key: str) -> float:
    denominator = float(sparse[key])
    return float(full[key]) / denominator if denominator > 0.0 else 0.0


def summarize(root: Path) -> dict[str, Any]:
    lengths = []
    for length_dir in sorted(
        root.glob("length*"),
        key=lambda path: int(path.name.removeprefix("length")),
    ):
        points = []
        full_cost_rows = []
        sparse_cost_rows = []
        for horizon_dir in sorted(
            length_dir.glob("g*"),
            key=lambda path: int(path.name.removeprefix("g")),
        ):
            rows = read_rows(horizon_dir / "sample_results.csv")
            by_method = {row["method"]: row for row in rows}
            if set(by_method) != {"full_kv", METHOD}:
                raise RuntimeError(
                    f"unexpected methods in {horizon_dir}: {sorted(by_method)}"
                )
            full = by_method["full_kv"]
            sparse = by_method[METHOD]
            full_cost_rows.append(full)
            sparse_cost_rows.append(sparse)
            points.append(
                {
                    "configured_generation_tokens": int(
                        horizon_dir.name.removeprefix("g")
                    ),
                    "prompt_tokens": int(full["prompt_tokens"]),
                    "full_generated_tokens": int(full["generated_tokens"]),
                    "sparse_generated_tokens": int(sparse["generated_tokens"]),
                    "full_score": float(full["score"]),
                    "sparse_score": float(sparse["score"]),
                    "quality_retention": (
                        float(sparse["score"]) / float(full["score"])
                        if float(full["score"]) > 0.0
                        else None
                    ),
                    "decode_speedup": ratio(full, sparse, "decode_seconds"),
                    "online_speedup": ratio(full, sparse, "online_seconds"),
                    "total_speedup": ratio(full, sparse, "total_seconds"),
                    "full_decode_seconds": float(full["decode_seconds"]),
                    "sparse_decode_seconds": float(sparse["decode_seconds"]),
                }
            )
        full_model = fit_decode_cost(full_cost_rows)
        sparse_model = fit_decode_cost(sparse_cost_rows)
        lengths.append(
            {
                "configured_prompt_tokens": int(
                    length_dir.name.removeprefix("length")
                ),
                "actual_prompt_tokens": points[0]["prompt_tokens"],
                "full_decode_cost_model": full_model,
                "methods": {
                    METHOD: {
                        "quality_retention": min(
                            point["quality_retention"]
                            for point in points
                            if point["quality_retention"] is not None
                        ),
                        "decode_cost_model": sparse_model,
                        "break_even_generated_forwards": break_even_steps(
                            full_model, sparse_model
                        ),
                    }
                },
                "points": points,
            }
        )
    return {
        "protocol": "single GovReport sample; exact prompt cap x generation horizon",
        "method": METHOD,
        "lengths": lengths,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CountCap 长度与生成时域二维成本网格",
        "",
        "同一个 GovReport 样本分别限制为 8K、16K、24K、32K，并生成 8、32、64 token。所有速度均包含 Key-PCA 建表和检索开销。",
        "",
        "| 实际 prompt | 生成上限 | Full decode | Key-PCA decode | Decode speed | Online speed | 质量保持率 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["lengths"]:
        for point in row["points"]:
            retention = point["quality_retention"]
            lines.append(
                "| {prompt} | {generation} | {full:.3f}s | {sparse:.3f}s | {decode:.3f}x | {online:.3f}x | {retention} |".format(
                    prompt=row["actual_prompt_tokens"],
                    generation=point["configured_generation_tokens"],
                    full=point["full_decode_seconds"],
                    sparse=point["sparse_decode_seconds"],
                    decode=point["decode_speedup"],
                    online=point["online_speedup"],
                    retention=(
                        "-" if retention is None else f"{100.0 * retention:.2f}%"
                    ),
                )
            )
    lines.extend(["", "## 解析交叉点", ""])
    for row in result["lengths"]:
        method = row["methods"][METHOD]
        crossing = method["break_even_generated_forwards"]
        lines.append(
            f"- {row['actual_prompt_tokens']} tokens："
            + (
                "当前范围内每步稀疏成本不低于 Full，禁用 CountCap。"
                if crossing is None
                else f"预计至少生成 {crossing + 1} tokens 后启用 CountCap。"
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_root)
    (args.run_root / "horizon_grid.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.run_root / "horizon_grid_zh.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
