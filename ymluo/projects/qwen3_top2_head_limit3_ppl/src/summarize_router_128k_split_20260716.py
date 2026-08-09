from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from summarize_128k_multitopic_windows_20260716 import bootstrap_ci


TOPICS = ("computer", "sports", "medicine", "space", "politics", "religion")
FOLDS = {
    "fold0": {
        "train": ("space", "politics", "religion"),
        "calibration": ("medicine",),
        "test": ("computer", "sports"),
    },
    "fold1": {
        "train": ("computer", "sports", "religion"),
        "calibration": ("politics",),
        "test": ("medicine", "space"),
    },
    "fold2": {
        "train": ("sports", "medicine", "space"),
        "calibration": ("computer",),
        "test": ("politics", "religion"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router_dir", required=True, type=Path)
    parser.add_argument("--paired_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    topic_to_fold = {
        topic: fold
        for fold, split in FOLDS.items()
        for topic in split["test"]
    }
    if set(topic_to_fold) != set(TOPICS):
        raise AssertionError("each topic must appear in exactly one test fold")
    for fold, split in FOLDS.items():
        train_topics = set(split["train"])
        calibration_topics = set(split["calibration"])
        test_topics = set(split["test"])
        if (train_topics | calibration_topics) & test_topics:
            raise AssertionError(f"topic leakage in {fold}")
    for topic in TOPICS:
        fold = topic_to_fold[topic]
        split = FOLDS[fold]
        full = load(args.paired_dir / f"{topic}_w2_full.json")
        fixed = load(args.paired_dir / f"{topic}_w2_sparse.json")
        full_online = float(full["synchronized_model_forward_seconds"])
        full_total = float(full["prefill_seconds"]) + full_online
        for refresh, suffix in ((1, ""), (2, "_refresh2")):
            router = load(
                args.router_dir / f"test_{fold}_{topic}_w2{suffix}.json"
            )
            if not (
                full["target_token_ids"]
                == fixed["target_token_ids"]
                == router["target_token_ids"]
            ):
                raise ValueError(f"target mismatch for {topic} window 2")
            if int(router.get("candidate_refresh_interval", 1)) != refresh:
                raise ValueError(f"refresh metadata mismatch for {topic}")
            router_online = float(router["online_seconds"])
            router_total = (
                float(router["prefill_seconds"])
                + float(router["cache_conversion_seconds"])
                + router_online
            )
            fractions = list(map(float, router["action_fractions"]))
            rows.append(
                {
                    "topic": topic,
                    "fold": fold,
                    "train_topics": ",".join(split["train"]),
                    "calibration_topics": ",".join(split["calibration"]),
                    "window": 2,
                    "candidate_refresh_interval": refresh,
                    "target_tokens": len(full["target_token_ids"]),
                    "full_ppl": float(full["ppl"]),
                    "fixed_1p5_ppl": float(fixed["ppl"]),
                    "router_ppl": float(router["ppl"]),
                    "fixed_quality_retention": math.exp(
                        -(float(fixed["nll"]) - float(full["nll"]))
                    ),
                    "router_quality_retention": math.exp(
                        -(float(router["nll"]) - float(full["nll"]))
                    ),
                    "router_mean_fraction": sum(fractions) / len(fractions),
                    "router_low_rate": fractions.count(0.01) / len(fractions),
                    "router_mid_rate": fractions.count(0.015) / len(fractions),
                    "router_high_rate": fractions.count(0.02) / len(fractions),
                    "router_kv_ratio": float(
                        router["hierarchical_over_final_length_full_kv"]
                    ),
                    "router_decode_speedup": full_online / router_online,
                    "router_protocol_speedup": full_total / router_total,
                    "router_overhead_seconds": float(router["router_seconds"]),
                }
            )

    rng = random.Random(args.seed)
    baseline_rows = [
        row for row in rows if row["candidate_refresh_interval"] == 1
    ]
    metrics = {
        "cases": len(baseline_rows),
        "ablation_cases": len(rows),
        "protocol": (
            "three-fold strict topic holdout; train w0 / calibration w1 / test w2"
        ),
        "folds": FOLDS,
    }
    for key in (
        "fixed_quality_retention",
        "router_quality_retention",
        "router_mean_fraction",
        "router_kv_ratio",
        "router_decode_speedup",
        "router_protocol_speedup",
    ):
        metrics[key] = bootstrap_ci(
            [float(row[key]) for row in baseline_rows], args.bootstrap_samples, rng
        )
    metrics["worst_router_quality_retention"] = min(
        float(row["router_quality_retention"]) for row in baseline_rows
    )
    metrics["worst_router_case"] = min(
        baseline_rows, key=lambda row: row["router_quality_retention"]
    )
    metrics["by_candidate_refresh_interval"] = {}
    for refresh in (1, 2):
        refresh_rows = [
            row for row in rows if row["candidate_refresh_interval"] == refresh
        ]
        metrics["by_candidate_refresh_interval"][str(refresh)] = {
            "cases": len(refresh_rows),
            "quality_retention": bootstrap_ci(
                [float(row["router_quality_retention"]) for row in refresh_rows],
                args.bootstrap_samples,
                rng,
            ),
            "decode_speedup": bootstrap_ci(
                [float(row["router_decode_speedup"]) for row in refresh_rows],
                args.bootstrap_samples,
                rng,
            ),
            "protocol_speedup": bootstrap_ci(
                [float(row["router_protocol_speedup"]) for row in refresh_rows],
                args.bootstrap_samples,
                rng,
            ),
            "worst_quality_retention": min(
                float(row["router_quality_retention"]) for row in refresh_rows
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "test_cases.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
