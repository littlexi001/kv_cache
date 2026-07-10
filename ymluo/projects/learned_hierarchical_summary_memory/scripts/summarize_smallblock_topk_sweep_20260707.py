from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_BLOCKS = (32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_GROUPS = ("longbench", "ruler4k", "ruler8k", "ruler16k")
METHOD_RE = re.compile(r"recent_plus_span_top(\d+)_b0_a0")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def top_k(method: str) -> int:
    if method == "full_raw":
        return 0
    match = METHOD_RE.fullmatch(method)
    if not match:
        raise ValueError(method)
    return int(match.group(1))


def split_dir(block: int, group: str) -> str:
    return f"qwen8b_block{block}_topk_sweep_{group}_m3_20260707"


def group_benchmark(group: str) -> str:
    return {
        "longbench": "longbench",
        "ruler4k": "ruler_4096",
        "ruler8k": "ruler_8192",
        "ruler16k": "ruler_16384",
    }[group]


def fnum(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def load_rows(root: Path, blocks: tuple[int, ...], groups: tuple[str, ...]) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    for block in blocks:
        for group in groups:
            path = root / "outputs" / split_dir(block, group) / "summary.csv"
            if not path.exists():
                continue
            for row in read_csv(path):
                row = dict(row)
                row["block"] = str(block)
                row["group"] = group
                rows_out.append(row)
    return rows_out


def full_score_for(rows: list[dict[str, str]], group: str, task: str = "__overall__") -> float:
    candidates = [
        row for row in rows
        if row["group"] == group and row["task"] == task and row["method"] == "full_raw"
    ]
    if not candidates:
        return 0.0
    # Full score should be block-invariant; use the median-like middle value after sorting to avoid one bad duplicate.
    scores = sorted(fnum(row, "avg_score") for row in candidates)
    return scores[len(scores) // 2]


def best_row(candidates: list[dict[str, str]]) -> dict[str, str] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda row: (fnum(row, "avg_score"), -fnum(row, "token_ratio_vs_full_raw")))


def min_combo(candidates: list[dict[str, str]], threshold: float) -> str:
    ok = [row for row in candidates if fnum(row, "avg_score") + 1e-12 >= threshold]
    if not ok:
        return ""
    row = min(
        ok,
        key=lambda item: (
            fnum(item, "token_ratio_vs_full_raw", 1.0),
            int(item["block"]),
            top_k(item["method"]),
        ),
    )
    return "b{}_top{}@{:.3f}".format(row["block"], top_k(row["method"]), fnum(row, "token_ratio_vs_full_raw"))


def emit_frontier(rows: list[dict[str, str]], groups: tuple[str, ...], blocks: tuple[int, ...]) -> list[str]:
    lines = ["OVERALL_FRONTIER", "group,block,method,samples,score,token,e2e_speed,attn_upper"]
    for group in groups:
        for block in blocks:
            group_rows = [
                row for row in rows
                if row["group"] == group and row["block"] == str(block) and row["task"] == "__overall__"
            ]
            for row in sorted(group_rows, key=lambda item: (top_k(item["method"]), item["method"])):
                token = fnum(row, "token_ratio_vs_full_raw")
                method = "full_raw" if row["method"] == "full_raw" else f"top{top_k(row['method'])}"
                lines.append(
                    ",".join(
                        [
                            group,
                            str(block),
                            method,
                            row["samples"],
                            f"{fnum(row, 'avg_score'):.4f}",
                            f"{token:.4f}",
                            f"{fnum(row, 'speedup_vs_full_raw'):.3f}",
                            f"{(1.0 / token if token else 0.0):.1f}",
                        ]
                    )
                )
    return lines


def emit_group_recommendations(rows: list[dict[str, str]], groups: tuple[str, ...]) -> list[str]:
    lines = [
        "",
        "OVERALL_RECOMMENDATIONS",
        "group,full_score,best_score,best_block,best_k,best_token,best_e2e,best_attn,min_ge_full,min_ge_98full,min_ge_95full",
    ]
    for group in groups:
        group_rows = [row for row in rows if row["group"] == group and row["task"] == "__overall__"]
        candidates = [row for row in group_rows if row["method"] != "full_raw"]
        best = best_row(candidates)
        if best is None:
            continue
        full_score = full_score_for(rows, group)
        token = fnum(best, "token_ratio_vs_full_raw")
        lines.append(
            ",".join(
                [
                    group,
                    f"{full_score:.4f}",
                    f"{fnum(best, 'avg_score'):.4f}",
                    best["block"],
                    str(top_k(best["method"])),
                    f"{token:.3f}",
                    f"{fnum(best, 'speedup_vs_full_raw'):.2f}",
                    f"{(1.0 / token if token else 0.0):.1f}",
                    min_combo(candidates, full_score),
                    min_combo(candidates, 0.98 * full_score),
                    min_combo(candidates, 0.95 * full_score),
                ]
            )
        )
    return lines


def emit_task_recommendations(rows: list[dict[str, str]], groups: tuple[str, ...]) -> list[str]:
    lines = [
        "",
        "PER_TASK_RECOMMENDATIONS",
        "benchmark,task,full,best_score,best_block,best_k,best_token,best_e2e,best_attn,min_ge_full,min_ge_98full,min_ge_95full",
    ]
    for group in groups:
        benchmark = group_benchmark(group)
        tasks = sorted({row["task"] for row in rows if row["benchmark"] == benchmark and row["task"] != "__overall__"})
        for task in tasks:
            task_rows = [row for row in rows if row["benchmark"] == benchmark and row["task"] == task]
            candidates = [row for row in task_rows if row["method"] != "full_raw"]
            best = best_row(candidates)
            if best is None:
                continue
            full_score = full_score_for(rows, group, task)
            token = fnum(best, "token_ratio_vs_full_raw")
            lines.append(
                ",".join(
                    [
                        benchmark,
                        task,
                        f"{full_score:.4f}",
                        f"{fnum(best, 'avg_score'):.4f}",
                        best["block"],
                        str(top_k(best["method"])),
                        f"{token:.3f}",
                        f"{fnum(best, 'speedup_vs_full_raw'):.2f}",
                        f"{(1.0 / token if token else 0.0):.1f}",
                        min_combo(candidates, full_score),
                        min_combo(candidates, 0.98 * full_score),
                        min_combo(candidates, 0.95 * full_score),
                    ]
                )
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory"))
    parser.add_argument("--blocks", default=",".join(str(block) for block in DEFAULT_BLOCKS))
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/smallblock_topk_sweep_m3_summary_20260707.txt"),
    )
    args = parser.parse_args()
    blocks = parse_ints(args.blocks)
    groups = parse_strings(args.groups)
    rows = load_rows(args.root, blocks, groups)
    if not rows:
        raise SystemExit("no summary rows found")
    lines = []
    lines.extend(emit_frontier(rows, groups, blocks))
    lines.extend(emit_group_recommendations(rows, groups))
    lines.extend(emit_task_recommendations(rows, groups))
    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
