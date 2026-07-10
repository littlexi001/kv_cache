from __future__ import annotations

import csv
import re
from pathlib import Path


BLOCKS = [256, 512, 1024, 2048]
GROUPS = ["longbench", "ruler4k", "ruler8k", "ruler16k"]
METHODS = [
    "recent_plus_span_top1_b0_a0",
    "recent_plus_span_top2_b0_a0",
    "recent_plus_span_top3_b0_a0",
    "recent_plus_span_top4_b0_a0",
    "recent_plus_span_top6_b0_a0",
    "recent_plus_span_top8_b0_a0",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def top_k(method: str) -> int:
    if method == "full_raw":
        return 0
    match = re.search(r"top(\d+)", method)
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


def main() -> None:
    root = Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory")
    all_rows: list[dict[str, str]] = []
    for block in BLOCKS:
        for group in GROUPS:
            path = root / "outputs" / split_dir(block, group) / "summary.csv"
            rows = read_csv(path)
            for row in rows:
                row = dict(row)
                row["block"] = str(block)
                row["group"] = group
                all_rows.append(row)

    print("OVERALL_FRONTIER")
    print("group,block,method,samples,score,token,e2e_speed,attn_upper")
    for group in GROUPS:
        for block in BLOCKS:
            rows = [
                row
                for row in all_rows
                if row["group"] == group and row["block"] == str(block) and row["task"] == "__overall__"
            ]
            for row in rows:
                token = float(row["token_ratio_vs_full_raw"])
                print(
                    group,
                    block,
                    "full_raw" if row["method"] == "full_raw" else f"top{top_k(row['method'])}",
                    row["samples"],
                    f"{float(row['avg_score']):.4f}",
                    f"{token:.4f}",
                    f"{float(row['speedup_vs_full_raw']):.3f}",
                    f"{(1.0 / token if token else 0.0):.1f}",
                    sep=",",
                )

    print("\nOVERALL_RECOMMENDATIONS")
    print("group,full_score,best_score,best_block,best_k,best_token,best_e2e,best_attn,min_ge_full,min_ge_98full,min_ge_95full")
    for group in GROUPS:
        rows = [row for row in all_rows if row["group"] == group and row["task"] == "__overall__"]
        full_score = float(next(row for row in rows if row["method"] == "full_raw" and row["block"] == "512")["avg_score"])
        candidates = [row for row in rows if row["method"] in METHODS]
        best = max(candidates, key=lambda row: (float(row["avg_score"]), -float(row["token_ratio_vs_full_raw"])))

        def min_combo(threshold: float) -> str:
            ok = [row for row in candidates if float(row["avg_score"]) + 1e-12 >= threshold]
            if not ok:
                return ""
            row = min(ok, key=lambda item: (float(item["token_ratio_vs_full_raw"]), int(item["block"]), top_k(item["method"])))
            return "b{}_top{}@{:.3f}".format(row["block"], top_k(row["method"]), float(row["token_ratio_vs_full_raw"]))

        token = float(best["token_ratio_vs_full_raw"])
        print(
            group,
            f"{full_score:.4f}",
            f"{float(best['avg_score']):.4f}",
            best["block"],
            top_k(best["method"]),
            f"{token:.3f}",
            f"{float(best['speedup_vs_full_raw']):.2f}",
            f"{(1.0 / token if token else 0.0):.1f}",
            min_combo(full_score),
            min_combo(0.98 * full_score),
            min_combo(0.95 * full_score),
            sep=",",
        )

    print("\nPER_TASK_RECOMMENDATIONS")
    print("benchmark,task,full,best_score,best_block,best_k,best_token,best_e2e,best_attn,min_ge_full,min_ge_98full,min_ge_95full")
    for group in GROUPS:
        benchmark = group_benchmark(group)
        tasks = sorted({row["task"] for row in all_rows if row["benchmark"] == benchmark and row["task"] != "__overall__"})
        for task in tasks:
            rows = [row for row in all_rows if row["benchmark"] == benchmark and row["task"] == task]
            full_score = float(next(row for row in rows if row["method"] == "full_raw" and row["block"] == "512")["avg_score"])
            candidates = [row for row in rows if row["method"] in METHODS]
            best = max(candidates, key=lambda row: (float(row["avg_score"]), -float(row["token_ratio_vs_full_raw"])))

            def min_combo(threshold: float) -> str:
                ok = [row for row in candidates if float(row["avg_score"]) + 1e-12 >= threshold]
                if not ok:
                    return ""
                row = min(ok, key=lambda item: (float(item["token_ratio_vs_full_raw"]), int(item["block"]), top_k(item["method"])))
                return "b{}_top{}@{:.3f}".format(row["block"], top_k(row["method"]), float(row["token_ratio_vs_full_raw"]))

            token = float(best["token_ratio_vs_full_raw"])
            print(
                benchmark,
                task,
                f"{full_score:.4f}",
                f"{float(best['avg_score']):.4f}",
                best["block"],
                top_k(best["method"]),
                f"{token:.3f}",
                f"{float(best['speedup_vs_full_raw']):.2f}",
                f"{(1.0 / token if token else 0.0):.1f}",
                min_combo(full_score),
                min_combo(0.98 * full_score),
                min_combo(0.95 * full_score),
                sep=",",
            )


if __name__ == "__main__":
    main()
