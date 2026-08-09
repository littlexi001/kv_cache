from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ["native", "deep_highfreq_drop", "slow_rope", "smooth_layer_frequency"]
DIRECTORIES = {variant: f"{variant}_v2" for variant in VARIANTS}


def read_last_final(path: Path) -> dict | None:
    if not path.exists():
        return None
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    finals = [record for record in records if record.get("final")]
    return finals[-1] if finals else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload: dict[str, dict] = {}
    train_payload: dict[str, dict] = {}
    for variant in VARIANTS:
        directory = args.outputs / DIRECTORIES[variant]
        final = read_last_final(directory / "eval.jsonl")
        if final is not None:
            payload[variant] = {str(row["length"]): row for row in final["rows"]}
        train_path = directory / "train.jsonl"
        if train_path.exists():
            train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if train_rows:
                train_payload[variant] = train_rows[-1]

    lengths = sorted({int(length) for rows in payload.values() for length in rows})
    lines = [
        "# 125M PE 预训练结果",
        "",
        "同一长度先看 Gold NLL（越低越好），再看准确率。只有完成 final eval 的条件才进入表格。",
        "",
        "| 长度 | 条件 | Accuracy | Gold NLL | Gold margin |",
        "|---:|---|---:|---:|---:|",
    ]
    for length in lengths:
        for variant in VARIANTS:
            row = payload.get(variant, {}).get(str(length))
            if row is None:
                continue
            lines.append(
                f"| {length:,} | `{variant}` | {row['accuracy']:.2%} | {row['gold_nll']:.4f} | {row['gold_margin']:+.4f} |"
            )

    if "native" in payload:
        lines.extend(
            [
                "",
                "## 相对原生 RoPE",
                "",
                "NLL 变化为 `变体 - native`，负数表示改善。准确率变化使用百分点。",
                "",
                "| 长度 | 条件 | Accuracy 变化 | Gold NLL 变化 |",
                "|---:|---|---:|---:|",
            ]
        )
        for length in lengths:
            baseline = payload["native"].get(str(length))
            if baseline is None:
                continue
            for variant in VARIANTS[1:]:
                row = payload.get(variant, {}).get(str(length))
                if row is None:
                    continue
                lines.append(
                    f"| {length:,} | `{variant}` | {(row['accuracy'] - baseline['accuracy']) * 100:+.2f} pp | {row['gold_nll'] - baseline['gold_nll']:+.4f} |"
                )

    lines.extend(
        [
            "",
            "## 训练状态",
            "",
            "| 条件 | 最后 step | tokens | train loss | tokens/s | 最大显存 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        row = train_payload.get(variant)
        if row is None:
            continue
        lines.append(
            f"| `{variant}` | {row['step']} | {row['tokens_seen']:,} | {row['loss']:.4f} | {row['tokens_per_second']:,.0f} | {row['max_memory_gib']:.2f} GiB |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
