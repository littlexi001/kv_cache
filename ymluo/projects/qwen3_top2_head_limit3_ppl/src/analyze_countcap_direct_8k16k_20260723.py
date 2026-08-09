from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    "full_kv",
    "countcap_fullprompt_keypca",
    "countcap_fullprompt_keypca_direct",
)
SPARSE_METHODS = METHODS[1:]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return float(statistics.median(values))


def summarize_case(case_dir: Path) -> dict[str, Any]:
    repeats: list[dict[str, dict[str, str]]] = []
    for repeat_dir in sorted(
        case_dir.glob("repeat*"),
        key=lambda path: int(path.name.removeprefix("repeat")),
    ):
        rows = read_rows(repeat_dir / "sample_results.csv")
        by_method = {row["method"]: row for row in rows}
        if set(by_method) != set(METHODS):
            raise RuntimeError(
                f"unexpected methods in {repeat_dir}: {sorted(by_method)}"
            )
        repeats.append(by_method)
    if not repeats:
        raise RuntimeError(f"no repeats found in {case_dir}")

    full_rows = [repeat["full_kv"] for repeat in repeats]
    result: dict[str, Any] = {
        "repeats": len(repeats),
        "prompt_tokens": int(full_rows[0]["prompt_tokens"]),
        "configured_generation_tokens": int(
            case_dir.name.removeprefix("g")
        ),
        "methods": {},
    }
    full_online = [float(row["online_seconds"]) for row in full_rows]
    full_decode = [float(row["decode_seconds"]) for row in full_rows]
    full_total = [float(row["total_seconds"]) for row in full_rows]
    for method in METHODS:
        rows = [repeat[method] for repeat in repeats]
        online = [float(row["online_seconds"]) for row in rows]
        decode = [float(row["decode_seconds"]) for row in rows]
        total = [float(row["total_seconds"]) for row in rows]
        method_result = {
            "score": median([float(row["score"]) for row in rows]),
            "generated_tokens": median(
                [float(row["generated_tokens"]) for row in rows]
            ),
            "configured_attention_fraction": median(
                [float(row["configured_attention_fraction"]) for row in rows]
            ),
            "configured_attention_tokens": median(
                [float(row["configured_attention_tokens"]) for row in rows]
            ),
            "median_online_seconds": median(online),
            "median_decode_seconds": median(decode),
            "median_total_seconds": median(total),
            "paired_online_speedup": median(
                [
                    full_value / method_value
                    for full_value, method_value in zip(full_online, online)
                ]
            ),
            "paired_decode_speedup": median(
                [
                    full_value / method_value
                    for full_value, method_value in zip(full_decode, decode)
                ]
            ),
            "paired_total_speedup": median(
                [
                    full_value / method_value
                    for full_value, method_value in zip(full_total, total)
                ]
            ),
        }
        result["methods"][method] = method_result
    return result


def summarize(run_root: Path) -> dict[str, Any]:
    cases = []
    for length_dir in sorted(
        run_root.glob("length*"),
        key=lambda path: int(path.name.removeprefix("length")),
    ):
        for case_dir in sorted(
            length_dir.glob("g*"),
            key=lambda path: int(path.name.removeprefix("g")),
        ):
            case = summarize_case(case_dir)
            case["configured_prompt_tokens"] = int(
                length_dir.name.removeprefix("length")
            )
            cases.append(case)
    return {
        "protocol": (
            "same GovReport sample, three paired repetitions; Full versus "
            "Key-PCA exact top-2% rerank versus sampled-threshold direct attention"
        ),
        "methods": list(METHODS),
        "cases": cases,
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    labels = {
        "full_kv": "Full KV",
        "countcap_fullprompt_keypca": "Key-PCA 2% exact rerank",
        "countcap_fullprompt_keypca_direct": "Key-PCA 3%-6% direct",
    }
    lines = [
        "# CountCap 取消 2% 精确重排：8K/16K 速度实验",
        "",
        "同一个 GovReport 长样本，每个点重复三次。Direct 路径使用 "
        "256 点 sampled-quantile 得到的候选集合直接做精确 QK softmax/V attention，"
        "不再精确重排并压回 2%。",
        "",
        "| Prompt | 输出上限 | 方法 | Attention 比例 | Online | Decode | Total | Online speed |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        for method in METHODS:
            row = case["methods"][method]
            lines.append(
                "| {prompt} | {generation} | {method} | {fraction:.2f}% | "
                "{online:.3f}s | {decode:.3f}s | {total:.3f}s | {speed:.3f}x |".format(
                    prompt=case["prompt_tokens"],
                    generation=case["configured_generation_tokens"],
                    method=labels[method],
                    fraction=100.0 * row["configured_attention_fraction"],
                    online=row["median_online_seconds"],
                    decode=row["median_decode_seconds"],
                    total=row["median_total_seconds"],
                    speed=row["paired_online_speedup"],
                )
            )
    lines.extend(
        [
            "",
            "所有速度都包含 dense suffix、第一次 Key-PCA/INT4 建表、阈值检索、"
            "真实 QK logits、稀疏 value attention 和生成循环。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_root)
    (args.run_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.run_root / "summary_zh.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
