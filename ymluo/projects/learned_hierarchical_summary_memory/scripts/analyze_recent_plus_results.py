from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(
    "/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/"
    "qwen8b_recent_plus_bench_m4_parallel_20260706/merged"
)


def main() -> None:
    summary_path = BASE / "summary.csv"
    trials_path = BASE / "trials.csv"
    summary = pd.read_csv(summary_path)
    trials = pd.read_csv(trials_path)

    overall = summary[summary["benchmark"] == "__overall__"].copy()
    full_score = float(overall[overall["method"] == "full_raw"].iloc[0]["avg_score"])
    overall["relative_score_vs_full_raw"] = overall["avg_score"] / full_score
    cols = [
        "method",
        "avg_score",
        "relative_score_vs_full_raw",
        "token_ratio_vs_full_raw",
        "speedup_vs_full_raw",
    ]
    print("== Overall methods ==")
    print(
        overall[cols]
        .sort_values("relative_score_vs_full_raw", ascending=False)
        .to_string(index=False)
    )

    print("\n== Score by benchmark ==")
    by_bench = trials.groupby(["benchmark", "method"])["score"].mean().unstack().round(4)
    print(by_bench.to_string())

    print("\n== Token ratio by benchmark ==")
    by_bench_token = (
        trials.groupby(["benchmark", "method"])["token_ratio_vs_full_raw"]
        .mean()
        .unstack()
        .round(4)
    )
    print(by_bench_token.to_string())

    actions = [m for m in overall["method"].tolist() if m != "full_raw"]
    case_keys = [
        "benchmark",
        "task",
        "case_id",
    ]
    rows = []
    for _, group in trials.groupby(case_keys, dropna=False):
        full = group[group["method"] == "full_raw"]
        if full.empty:
            continue
        full_score = float(full.iloc[0]["score"])
        candidates = group[group["method"].isin(actions)].copy()
        candidates = candidates.sort_values(
            ["token_ratio_vs_full_raw", "score"], ascending=[True, False]
        )
        match = candidates[candidates["score"] >= full_score]
        if match.empty:
            match = candidates.sort_values(
                ["score", "token_ratio_vs_full_raw"], ascending=[False, True]
            ).head(1)
        else:
            match = match.head(1)
        best = candidates.sort_values(
            ["score", "token_ratio_vs_full_raw"], ascending=[False, True]
        ).head(1)
        rows.append(
            {
                "full_score": full_score,
                "match_method": match.iloc[0]["method"],
                "match_score": float(match.iloc[0]["score"]),
                "match_token_ratio": float(match.iloc[0]["token_ratio_vs_full_raw"]),
                "best_method": best.iloc[0]["method"],
                "best_score": float(best.iloc[0]["score"]),
                "best_token_ratio": float(best.iloc[0]["token_ratio_vs_full_raw"]),
            }
        )

    oracle = pd.DataFrame(rows)
    print("\n== Oracle ==")
    full_mean = oracle["full_score"].mean()
    for prefix in ("match", "best"):
        score = oracle[f"{prefix}_score"].mean()
        token = oracle[f"{prefix}_token_ratio"].mean()
        rel = score / full_mean if full_mean else 0.0
        print(f"{prefix}: score={score:.4f}, relative={rel*100:.2f}%, token={token*100:.2f}%")
        print(oracle[f"{prefix}_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
