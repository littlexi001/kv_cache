from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--initial-score", type=float, default=0.492948717948718)
    parser.add_argument("--native-baseline", type=float, default=0.8519230769230769)
    parser.add_argument("--initial-tokens", type=int, default=10027008)
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = [
        {
            "checkpoint": "initial_10m",
            "additional_steps": 0,
            "cumulative_tokens": args.initial_tokens,
            "ruler_score": args.initial_score,
            "exceeds_native": args.initial_score > args.native_baseline,
        }
    ]
    evaluations = args.run_dir / "checkpoint_evals"
    for path in sorted(evaluations.glob("step_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        item = next(row for row in summary if row["variant"] == "nope_every4_offset3")
        step = int(path.parent.name.split("_", 1)[1])
        score = float(item["official_score_mean"])
        rows.append(
            {
                "checkpoint": path.parent.name,
                "additional_steps": step,
                "cumulative_tokens": args.initial_tokens + step * args.tokens_per_step,
                "ruler_score": score,
                "exceeds_native": score > args.native_baseline,
            }
        )
    final_path = args.run_dir / "final_eval" / "summary.json"
    if final_path.exists():
        summary = json.loads(final_path.read_text(encoding="utf-8"))
        item = next(row for row in summary if row["variant"] == "nope_every4_offset3")
        config = json.loads((args.run_dir / "train" / "config.json").read_text(encoding="utf-8"))
        step = int(config["max_steps"])
        score = float(item["official_score_mean"])
        if not any(row["additional_steps"] == step for row in rows):
            rows.append(
                {
                    "checkpoint": "final",
                    "additional_steps": step,
                    "cumulative_tokens": args.initial_tokens + step * args.tokens_per_step,
                    "ruler_score": score,
                    "exceeds_native": score > args.native_baseline,
                }
            )
    rows.sort(key=lambda row: row["additional_steps"])
    with (args.run_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["ruler_score"])
    result = {
        "native_baseline": args.native_baseline,
        "points": rows,
        "best": best,
        "ever_exceeds_native": any(row["exceeds_native"] for row in rows),
    }
    (args.run_dir / "training_curve.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# RNoPE 训练量扫描",
        "",
        f"原生 RoPE baseline：{100 * args.native_baseline:.2f}%",
        "",
        "| Checkpoint | 累计训练 token | RULER-32K | 超过原生 RoPE |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['cumulative_tokens']:,} | "
            f"{100 * row['ruler_score']:.2f}% | {'是' if row['exceeds_native'] else '否'} |"
        )
    (args.run_dir / "training_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

