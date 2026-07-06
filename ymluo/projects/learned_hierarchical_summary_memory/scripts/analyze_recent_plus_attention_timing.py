from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(
    "/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/"
    "recent_plus_attention_subsystem_qwen8b_multilen_warm_20260706"
)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    df = pd.read_csv(BASE / "recent_plus_attention_subsystem_timing.csv")

    print("== 1024-step page_once ==")
    keep = df[
        (df["steps"] == 1024)
        & (
            df["method"].isin(
                [
                    "full_attention",
                    "recent_plus_k2_once",
                    "recent_plus_k3_once",
                    "recent_plus_k4_once",
                ]
            )
        )
    ].copy()
    keep["active_ratio_pct"] = keep["active_ratio"].map(pct)
    print(
        keep[
            [
                "method",
                "full_len",
                "active_kv_start",
                "active_ratio_pct",
                "overhead_ms",
                "attention_ms",
                "total_ms",
                "speedup_vs_full_total",
            ]
        ].to_string(index=False)
    )

    print("\n== 1024-step k2 once vs interval128 ==")
    keep = df[
        (df["steps"] == 1024)
        & (df["method"].isin(["recent_plus_k2_once", "recent_plus_k2_interval128"]))
    ].copy()
    keep["active_ratio_pct"] = keep["active_ratio"].map(pct)
    print(
        keep[
            [
                "method",
                "full_len",
                "reroutes",
                "active_ratio_pct",
                "overhead_ms",
                "total_ms",
                "speedup_vs_full_total",
                "overhead_share",
            ]
        ].to_string(index=False)
    )

    print("\n== 1-step new-query overhead impact ==")
    keep = df[
        (df["steps"] == 1)
        & (
            df["method"].isin(
                ["full_attention", "recent_plus_k2_once", "recent_plus_k3_once", "recent_plus_k4_once"]
            )
        )
    ].copy()
    print(
        keep[
            [
                "method",
                "full_len",
                "overhead_ms",
                "attention_ms",
                "total_ms",
                "speedup_vs_full_total",
                "overhead_share",
            ]
        ].to_string(index=False)
    )

    print("\n== Overhead components, page_once averaged by k ==")
    keep = df[(df["steps"] == 1024) & (df["method"].str.endswith("_once"))].copy()
    grouped = (
        keep.groupby("method")[
            ["router_scoring_topk_ms", "gather_compact_ms", "overhead_ms", "overhead_share"]
        ]
        .mean()
        .reset_index()
    )
    print(grouped.to_string(index=False))

    print("\n== Best speedup by length, 1024-step page_once ==")
    keep = df[(df["steps"] == 1024) & (df["method"].str.endswith("_once"))].copy()
    idx = keep.groupby("full_len")["speedup_vs_full_total"].idxmax()
    print(
        keep.loc[idx][
            ["full_len", "method", "active_kv_start", "active_ratio", "total_ms", "speedup_vs_full_total"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
