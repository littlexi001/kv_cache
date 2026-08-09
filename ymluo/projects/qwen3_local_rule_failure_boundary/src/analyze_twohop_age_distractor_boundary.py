from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def read_rows(input_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    by_length: dict[int, dict[str, Any]] = {}
    for input_dir in input_dirs:
        manifest_path = input_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for completed in manifest.get("completed", []):
            payload_path = input_dir / completed["file"]
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            answer = payload["answer"]
            generation = payload.get("generation")
            row = {
                **completed,
                "source_dir": input_dir.name,
                "semantic_gold_probability": answer[
                    "semantic_gold_probability"
                ],
                "semantic_gold_ppl": answer["semantic_gold_ppl"],
                "semantic_full_vocab_margin": answer[
                    "semantic_full_vocab_margin"
                ],
                "semantic_full_vocab_correct": bool(
                    answer["semantic_full_vocab_correct"]
                ),
                "semantic_candidate_margin": answer[
                    "semantic_candidate_margin"
                ],
                "semantic_candidate_correct": bool(
                    answer["semantic_candidate_correct"]
                ),
                "semantic_candidate_prediction": answer[
                    "semantic_candidate_prediction"
                ],
                "top_token": answer["top_token"],
                "generation_correct": (
                    None
                    if generation is None
                    else bool(generation["answer_correct"])
                ),
                "generation_answer": (
                    None
                    if generation is None
                    else generation["first_number_word"]
                ),
                "generation_text": (
                    None if generation is None else generation["text"]
                ),
            }
            length = int(row["total_tokens"])
            previous = by_length.get(length)
            if previous is None:
                by_length[length] = row
            elif (
                previous["generation_correct"] is None
                and row["generation_correct"] is not None
            ):
                by_length[length] = row
    return [by_length[length] for length in sorted(by_length)]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "total_tokens",
        "distractor_count",
        "filler_count",
        "semantic_gold_probability",
        "semantic_gold_ppl",
        "semantic_full_vocab_margin",
        "semantic_full_vocab_correct",
        "semantic_candidate_margin",
        "semantic_candidate_correct",
        "semantic_candidate_prediction",
        "top_token",
        "generation_correct",
        "generation_answer",
        "generation_text",
        "source_dir",
        "file",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def contiguous_windows(
    rows: Sequence[dict[str, Any]],
    *,
    start: int,
    end: int,
    step: int,
    size: int,
) -> list[dict[str, Any]]:
    indexed = {
        int(row["total_tokens"]): row
        for row in rows
        if start <= int(row["total_tokens"]) <= end
    }
    output = []
    for left in range(start, end - (size - 1) * step + 1, step):
        lengths = [left + index * step for index in range(size)]
        if any(length not in indexed for length in lengths):
            continue
        window_rows = [indexed[length] for length in lengths]
        failures = sum(
            not bool(row["semantic_full_vocab_correct"])
            for row in window_rows
        )
        output.append(
            {
                "start_tokens": lengths[0],
                "center_tokens": lengths[size // 2],
                "end_tokens": lengths[-1],
                "point_count": size,
                "failure_count": failures,
                "failure_rate": failures / size,
                "states": [
                    {
                        "total_tokens": int(row["total_tokens"]),
                        "correct": bool(
                            row["semantic_full_vocab_correct"]
                        ),
                    }
                    for row in window_rows
                ],
            }
        )
    return output


def kib(tokens: int) -> str:
    return (
        f"{tokens / 1024:.0f}K"
        if tokens % 1024 == 0
        else f"{tokens:,}"
    )


def build_report(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    selected_lengths = {
        81920,
        86016,
        90112,
        98304,
        99328,
        100352,
        101376,
        102400,
        106496,
        110592,
        114688,
        118784,
        122880,
        131072,
        139264,
        147456,
        163840,
    }
    table_rows = [
        row for row in rows if int(row["total_tokens"]) in selected_lengths
    ]
    lines = [
        "# 两跳年龄检索：干扰信息下的失败边界",
        "",
        "## 结论",
        "",
        "- 模型：Qwen3-8B；RoPE/YaRN factor 固定为 8。",
        "- 干扰信息与之前单跳实验相同：合理但无关的“某人的年龄”事实，约每 32 token 一条；其余为单-token 句号。",
        "- 两跳证据：`Xiaohong → Xiaoming → nine`。",
        "- **首次局部翻转位于 99K–100K：99K 正确，100K 错误。**",
        "- **约 100K 开始进入高频抖动区；它不是永久失败阈值。** 104K 会恢复，108K/112K 再次失败，116K 后又恢复。",
        "- 4K 网格的五点窗口中，96–112K 与 100–116K 的失败率均为 60%；104–120K 降为 40%。",
        "- Greedy 验证表明 100K、108K、112K 都没有在后续 16 token 中补出正确年龄，因此不是仅由换行或大小写造成的假失败。",
        "",
        "## 边界附近结果",
        "",
        "| 长度 | 干扰句 | 正确 | Gold 概率 | PPL | Top-1 margin | Top token | 候选年龄 | Greedy |",
        "|---:|---:|:---:|---:|---:|---:|---|---|---|",
    ]
    for row in table_rows:
        generation = "—"
        if row["generation_correct"] is True:
            generation = f"对（{row['generation_answer']}）"
        elif row["generation_correct"] is False:
            generation = "错"
        lines.append(
            f"| {kib(int(row['total_tokens']))} | "
            f"{int(row['distractor_count']):,} | "
            f"{'✓' if row['semantic_full_vocab_correct'] else '✗'} | "
            f"{100 * float(row['semantic_gold_probability']):.2f}% | "
            f"{float(row['semantic_gold_ppl']):.3f} | "
            f"{float(row['semantic_full_vocab_margin']):+.3f} | "
            f"`{str(row['top_token']).replace(chr(10), '↵')}` | "
            f"`{row['semantic_candidate_prediction']}` | "
            f"{generation} |"
        )
    lines.extend(
        [
            "",
            "## 如何理解“边界”",
            "",
            "这个实验不存在满足单调性的硬长度阈值，因此不能把普通二分的某个错误点解释成“此后永远失败”。更准确的定义有两层：",
            "",
            "1. **首次局部翻转下沿：99K–100K。**",
            "2. **高频失败带：约 100K–116K。** 在这里，相邻 4K 探针形成的五点窗口失败率达到 60%。",
            "",
            "100K 点包含 3,199 条年龄干扰句。此处 Gold 概率从 99K 的约 "
            f"{100 * float(summary['point_99k']['semantic_gold_probability']):.2f}% "
            "降到 "
            f"{100 * float(summary['point_100k']['semantic_gold_probability']):.2f}%，"
            "margin 从 "
            f"{float(summary['point_99k']['semantic_full_vocab_margin']):+.3f} "
            "变为 "
            f"{float(summary['point_100k']['semantic_full_vocab_margin']):+.3f}。",
            "",
            "## 与单跳实验的近似比较",
            "",
            "- 单跳此前的多数失败窗口约在 140K。",
            "- 两跳本次约在 100K 进入高频抖动区，约提前 40K，所能容忍的上下文长度约少 29%。",
            "- 这是近似比较：单跳使用过逐 token 增量扫描，本次使用粗扫、1K 二分和 4K 局部窗口；两者模型、RoPE factor、干扰模板和干扰密度一致。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dirs = [Path(value) for value in args.input_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_dirs)
    failures = [
        row for row in rows if not row["semantic_full_vocab_correct"]
    ]
    first_failure = min(
        failures,
        key=lambda row: int(row["total_tokens"]),
    )
    predecessor = max(
        (
            row
            for row in rows
            if row["semantic_full_vocab_correct"]
            and int(row["total_tokens"])
            < int(first_failure["total_tokens"])
        ),
        key=lambda row: int(row["total_tokens"]),
    )
    windows = contiguous_windows(
        rows,
        start=96 * 1024,
        end=120 * 1024,
        step=4 * 1024,
        size=5,
    )
    high_frequency = [
        window for window in windows if window["failure_rate"] > 0.5
    ]
    indexed = {int(row["total_tokens"]): row for row in rows}
    summary = {
        "schema_version": 1,
        "point_count": len(rows),
        "failure_count": len(failures),
        "first_failure_tokens": int(first_failure["total_tokens"]),
        "last_observed_correct_before_first_failure_tokens": int(
            predecessor["total_tokens"]
        ),
        "local_flip_interval_tokens": [
            int(predecessor["total_tokens"]),
            int(first_failure["total_tokens"]),
        ],
        "high_frequency_definition": (
            "failure rate > 50% in five contiguous points spaced 4K apart"
        ),
        "windows": windows,
        "high_frequency_windows": high_frequency,
        "point_99k": indexed[99 * 1024],
        "point_100k": indexed[100 * 1024],
        "failures": failures,
    }
    write_csv(output_dir / "boundary_points.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(rows, summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
