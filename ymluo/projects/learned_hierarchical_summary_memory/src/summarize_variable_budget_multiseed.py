from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


INTERESTING_POLICIES = {
    "fixed_k1_compact",
    "fixed_k2_compact",
    "fixed_k8_compact",
    "learned_planner",
    "oracle_best",
    "oracle_min_safe",
    "risk_best_score_then_low_kv",
    "risk_min_kv_at_full_score",
    "risk_min_kv_within_one_point_of_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize variable-budget planner multi-seed runs.")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--print_all", action="store_true")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_overall_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        row for row in summary["prediction_summary"]
        if row["split"] == "test" and row["group"] == "__overall__"
    ]
    return {row["policy"]: row for row in rows}


def add_policy_row(
    rows: list[dict[str, Any]],
    target: str,
    seed: str,
    policy: str,
    row: dict[str, Any],
    full_score: float,
    tau: float | None = None,
) -> None:
    rows.append(
        {
            "target": target,
            "seed": seed,
            "policy": policy,
            "tau": tau,
            "score": float(row["avg_score"]),
            "kv": float(row["avg_active_kv_ratio_vs_full"]),
            "label_acc": float(row["label_accuracy"]),
            "full_score": full_score,
            "score_delta_vs_full": float(row["avg_score"]) - full_score,
        }
    )


def parse_run_name(run_dir: Path, summary: dict[str, Any]) -> tuple[str, str] | None:
    name = run_dir.name
    if "_seed_" in name:
        return name.split("_seed_", 1)
    if name.startswith("holdout_"):
        label_target = summary.get("config", {}).get("label_target", "unknown")
        return f"{label_target}_leave_task_out", name.removeprefix("holdout_")
    return None


def collect_rows(base_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parsed = parse_run_name(run_dir, summary)
        if parsed is None:
            continue
        target, seed = parsed
        by_policy = test_overall_rows(summary)
        if "fixed_full" not in by_policy:
            continue
        full_score = float(by_policy["fixed_full"]["avg_score"])
        for policy, row in sorted(by_policy.items()):
            if policy == "fixed_full":
                continue
            add_policy_row(rows, target, seed, policy, row, full_score)

        risk_summary = summary.get("risk_threshold_summary") or {}
        for key in [
            "best_score_then_low_kv",
            "min_kv_at_full_score",
            "min_kv_within_one_point_of_full",
        ]:
            row = risk_summary.get(key)
            if row:
                add_policy_row(rows, target, seed, f"risk_{key}", row, full_score, row.get("tau"))
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["target"], row["policy"]) for row in rows})
    for target, policy in keys:
        items = [row for row in rows if row["target"] == target and row["policy"] == policy]
        scores = [float(row["score"]) for row in items]
        kvs = [float(row["kv"]) for row in items]
        deltas = [float(row["score_delta_vs_full"]) for row in items]
        labels = [float(row["label_acc"]) for row in items]
        taus = [float(row["tau"]) for row in items if row.get("tau") not in (None, "")]
        out.append(
            {
                "target": target,
                "policy": policy,
                "runs": len(items),
                "score_mean": mean(scores),
                "score_sd": stdev(scores),
                "kv_mean": mean(kvs),
                "kv_sd": stdev(kvs),
                "delta_vs_full_mean": mean(deltas),
                "delta_vs_full_sd": stdev(deltas),
                "label_acc_mean": mean(labels),
                "full_level_runs": sum(1 for value in deltas if value >= -1e-12),
                "tau_mean": mean(taus) if taus else "",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    rows = collect_rows(base_dir)
    if not rows:
        raise SystemExit(f"no multiseed rows found under {base_dir}")
    summary = summarize(rows)
    write_csv(base_dir / "aggregate_policy_rows.csv", rows)
    write_csv(base_dir / "aggregate_policy_summary.csv", summary)
    print("target,policy,runs,score_mean,score_sd,kv_mean,kv_sd,delta_mean,full_level_runs,tau_mean")
    for row in summary:
        if args.print_all or row["policy"] in INTERESTING_POLICIES:
            tau = row["tau_mean"]
            tau_text = f"{tau:.4f}" if isinstance(tau, float) else ""
            print(
                f"{row['target']},{row['policy']},{row['runs']},"
                f"{row['score_mean']:.4f},{row['score_sd']:.4f},"
                f"{row['kv_mean']:.4f},{row['kv_sd']:.4f},"
                f"{row['delta_vs_full_mean']:.4f},{row['full_level_runs']},{tau_text}"
            )
    print(f"saved {base_dir / 'aggregate_policy_summary.csv'}")


if __name__ == "__main__":
    main()
