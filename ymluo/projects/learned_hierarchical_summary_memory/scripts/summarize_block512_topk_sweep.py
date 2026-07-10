from __future__ import annotations

import csv
import re
from pathlib import Path


SPLITS = [
    "qwen8b_block512_topk_sweep_longbench_m3_20260707",
    "qwen8b_block512_topk_sweep_ruler4k_m3_20260707",
    "qwen8b_block512_topk_sweep_ruler8k_m3_20260707",
    "qwen8b_block512_topk_sweep_ruler16k_m3_20260707",
]
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
    match = re.search(r"top(\d+)", method)
    if not match:
        raise ValueError(method)
    return int(match.group(1))


def method_label(method: str) -> str:
    if method == "full_raw":
        return "full_raw"
    return f"top{top_k(method)}"


def main() -> None:
    root = Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory")
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        rows.extend(read_csv(root / "outputs" / split / "summary.csv"))

    print("OVERALL")
    print("group,method,samples,score,token,e2e_speed,attn_upper")
    for split in SPLITS:
        split_rows = read_csv(root / "outputs" / split / "summary.csv")
        group = split
        prefix = "qwen8b_block512_topk_sweep_"
        suffix = "_m3_20260707"
        if group.startswith(prefix):
            group = group[len(prefix) :]
        if group.endswith(suffix):
            group = group[: -len(suffix)]
        for row in split_rows:
            if row["task"] != "__overall__":
                continue
            token = float(row["token_ratio_vs_full_raw"])
            attn_upper = 1.0 / token if token else 0.0
            print(
                group,
                method_label(row["method"]),
                row["samples"],
                f"{float(row['avg_score']):.4f}",
                f"{token:.4f}",
                f"{float(row['speedup_vs_full_raw']):.3f}",
                f"{attn_upper:.1f}",
                sep=",",
            )

    print("\nPER_TASK")
    print(
        "benchmark,task,full,best_score,best_k,best_token,best_e2e_speed,"
        "best_attn_upper,min_k_ge_full,min_k_ge_95full,min_k_ge_98full"
    )
    for benchmark in ["longbench", "ruler_4096", "ruler_8192", "ruler_16384"]:
        tasks = sorted({row["task"] for row in rows if row["benchmark"] == benchmark and row["task"] != "__overall__"})
        for task in tasks:
            task_rows = [row for row in rows if row["benchmark"] == benchmark and row["task"] == task]
            full_score = float(next(row for row in task_rows if row["method"] == "full_raw")["avg_score"])
            candidates = [row for row in task_rows if row["method"] in METHODS]
            best = max(candidates, key=lambda row: (float(row["avg_score"]), -float(row["token_ratio_vs_full_raw"])))

            def min_k(threshold: float) -> str:
                ok = [row for row in candidates if float(row["avg_score"]) + 1e-12 >= threshold]
                if not ok:
                    return ""
                row = min(ok, key=lambda item: (top_k(item["method"]), float(item["token_ratio_vs_full_raw"])))
                return f"top{top_k(row['method'])}@{float(row['token_ratio_vs_full_raw']):.3f}"

            token = float(best["token_ratio_vs_full_raw"])
            print(
                benchmark,
                task,
                f"{full_score:.4f}",
                f"{float(best['avg_score']):.4f}",
                top_k(best["method"]),
                f"{token:.3f}",
                f"{float(best['speedup_vs_full_raw']):.2f}",
                f"{(1.0 / token if token else 0.0):.1f}",
                min_k(full_score),
                min_k(0.95 * full_score),
                min_k(0.98 * full_score),
                sep=",",
            )


if __name__ == "__main__":
    main()
