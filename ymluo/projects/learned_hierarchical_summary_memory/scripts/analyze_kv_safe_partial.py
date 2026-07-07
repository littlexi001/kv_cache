from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(
    "/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/"
    "qwen8b_kv_safe_actions_small_20260706"
)


def main() -> None:
    path = BASE / "trials.partial.csv"
    df = pd.read_csv(path)
    case_cols = ["benchmark", "task", "case_id"]
    print(f"trials={len(df)} cases={df[case_cols].drop_duplicates().shape[0]}")
    print("\n== method counts ==")
    print(df.groupby("method").size().to_string())

    methods = sorted(df["method"].unique())
    complete = df.groupby(case_cols).filter(lambda rows: set(rows["method"]) == set(methods))
    print(f"\ncomplete_trials={len(complete)} complete_cases={complete[case_cols].drop_duplicates().shape[0]}")
    if complete.empty:
        return

    full = complete[complete["method"] == "full_raw"][
        case_cols + ["score", "prompt_tokens", "seconds"]
    ].rename(
        columns={
            "score": "full_score",
            "prompt_tokens": "full_tokens",
            "seconds": "full_seconds",
        }
    )
    merged = complete.merge(full, on=case_cols)
    merged["token_ratio"] = merged["prompt_tokens"] / merged["full_tokens"]
    summary = (
        merged.groupby("method")
        .agg(
            samples=("score", "size"),
            score=("score", "mean"),
            full=("full_score", "mean"),
            token=("token_ratio", "mean"),
            seconds=("seconds", "mean"),
        )
        .reset_index()
    )
    summary["relative"] = summary["score"] / summary["full"]
    print("\n== completed-case summary ==")
    print(summary.sort_values("relative", ascending=False).to_string(index=False))

    print("\n== by benchmark ==")
    by_bench = merged.groupby(["benchmark", "method"])["score"].mean().unstack().round(4)
    print(by_bench.to_string())


if __name__ == "__main__":
    main()
