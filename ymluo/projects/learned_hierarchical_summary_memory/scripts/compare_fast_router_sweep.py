from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_ROOT = Path("/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs")
BASE_NAMES = [
    "fast_recent_plus_router_sweep_20260706",
    "fast_recent_plus_router_sweep_v2_20260706",
]


def main() -> None:
    rows = []
    for base_name in BASE_NAMES:
        base = OUTPUT_ROOT / base_name
        if not base.exists():
            continue
        for policy in ("budget", "balanced", "conservative"):
            if not (base / policy / "heldout_benchmark_summary.csv").exists():
                continue
            fast = pd.read_csv(base / policy / "heldout_benchmark_summary.csv")
            for _, row in fast.iterrows():
                if row["group"] == "__overall__":
                    rows.append(
                        {
                            "run": base_name,
                            "source": "fast_raw",
                            "policy": policy,
                            "eval_policy": "raw_router",
                            "group": row["group"],
                            "score": row["avg_score"],
                            "relative": row["relative_to_full"],
                            "token": row["avg_token_ratio_vs_full_raw"],
                        }
                    )
            policy_eval = base / f"{policy}_policy_eval" / "policy_summary.csv"
            if not policy_eval.exists():
                continue
            summary = pd.read_csv(policy_eval)
            keep = summary[
                (summary["group"] == "__overall__")
                & (
                    summary["policy"].isin(
                        [
                            "learned_router",
                            "learned_router_conservative",
                            "learned_router_safety_only",
                            "oracle_match_full",
                        ]
                    )
                )
            ]
            for _, row in keep.iterrows():
                rows.append(
                    {
                        "run": base_name,
                        "source": "policy_eval",
                        "policy": policy,
                        "eval_policy": row["policy"],
                        "group": row["group"],
                        "score": row["avg_score"],
                        "relative": row["relative_to_full"],
                        "token": row["avg_token_ratio_vs_full_raw"],
                    }
                )
    out = pd.DataFrame(rows)
    print(out.sort_values(["run", "policy", "source", "eval_policy"]).to_string(index=False))

    print("\n== by group for v2 balanced learned_router ==")
    summary = pd.read_csv(OUTPUT_ROOT / BASE_NAMES[-1] / "balanced_policy_eval" / "policy_summary.csv")
    rows = summary[summary["policy"].isin(["learned_router", "learned_router_conservative"])]
    print(rows[["policy", "group", "samples", "avg_score", "relative_to_full", "avg_token_ratio_vs_full_raw"]].to_string(index=False))


if __name__ == "__main__":
    main()
