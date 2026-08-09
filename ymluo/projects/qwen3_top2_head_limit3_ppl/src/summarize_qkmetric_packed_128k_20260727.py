from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TOPICS = ("mixed_a", "mixed_b")
DEV_WINDOWS = (0, 1)
HOLDOUT_WINDOWS = (2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize 128K QK-balanced packed-index validation."
    )
    parser.add_argument("--old_root", type=Path, required=True)
    parser.add_argument("--keypca1_root", type=Path, required=True)
    parser.add_argument("--diagnosis_root", type=Path, required=True)
    parser.add_argument("--qkmetric_dev_root", type=Path, required=True)
    parser.add_argument("--holdout_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected a list in {path}")
    return [dict(row) for row in payload]


def select(
    payload: list[dict[str, Any]],
    *,
    topic: str,
    window: int,
    method: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in payload
        if str(row.get("topic")) == topic
        and int(row.get("window", -1)) == window
        and str(row.get("method")) == method
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {method} row for {topic}/w{window}; "
            f"found {len(matches)}"
        )
    return matches[0]


def single_case(
    root: Path,
    topic: str,
    window: int,
    method: str = "direct_countcap",
) -> dict[str, Any]:
    return select(
        records(root / f"{topic}_w{window}" / "case_summary.json"),
        topic=topic,
        window=window,
        method=method,
    )


def weighted_ppl(rows: list[dict[str, Any]]) -> float:
    token_count = sum(int(row["tokens"]) for row in rows)
    mean_nll = sum(
        int(row["tokens"]) * float(row["nll"]) for row in rows
    ) / token_count
    return math.exp(min(20.0, mean_nll))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def aggregate(
    split: str,
    full_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    full_ppl = weighted_ppl(full_rows)
    method_ppl = weighted_ppl(method_rows)
    per_case_retention = [
        math.exp(float(full["nll"]) - float(method["nll"]))
        for full, method in zip(full_rows, method_rows, strict=True)
    ]
    result = {
        "split": split,
        "cases": len(method_rows),
        "tokens": sum(int(row["tokens"]) for row in method_rows),
        "full_ppl": full_ppl,
        "method_ppl": method_ppl,
        "quality_retention": full_ppl / method_ppl,
        "worst_case_retention": min(per_case_retention),
        "actual_attention_tokens_mean": mean(
            method_rows, "actual_attention_tokens_mean"
        ),
        "actual_attention_ratio_mean": mean(
            [
                {
                    "ratio": float(row["actual_attention_tokens_mean"])
                    / float(row["history_tokens"])
                }
                for row in method_rows
            ],
            "ratio",
        ),
        "index_ratio_of_full_kv_mean": mean(
            method_rows, "packed_index_ratio_of_full_kv"
        ),
        "fixed_overhead_seconds_mean": mean(
            method_rows, "fixed_sparse_overhead_seconds"
        ),
        "steady_speedup_mean": sum(
            float(full["sparse_seconds_per_step"])
            / float(method["steady_sparse_seconds_per_step"])
            for full, method in zip(full_rows, method_rows, strict=True)
        )
        / len(method_rows),
        "measured_speedup_mean": sum(
            float(full["sparse_decode_seconds"])
            / float(method["sparse_decode_seconds"])
            for full, method in zip(full_rows, method_rows, strict=True)
        )
        / len(method_rows),
        "break_even_generated_steps_mean": sum(
            float(method["fixed_sparse_overhead_seconds"])
            / (
                float(full["sparse_seconds_per_step"])
                - float(method["steady_sparse_seconds_per_step"])
            )
            for full, method in zip(full_rows, method_rows, strict=True)
        )
        / len(method_rows),
        "projected_1024_speedup_mean": sum(
            1024.0 * float(full["sparse_seconds_per_step"])
            / (
                float(method["fixed_sparse_overhead_seconds"])
                + 1024.0
                * float(method["steady_sparse_seconds_per_step"])
            )
            for full, method in zip(full_rows, method_rows, strict=True)
        )
        / len(method_rows),
        "overflow_rate_mean": mean(
            method_rows, "candidate_overflow_rate_mean"
        ),
    }
    stability = [
        row for row in method_rows if "top1_agreement" in row
    ]
    if stability:
        for key in (
            "top1_agreement",
            "margin_flip_rate",
            "margin_certificate_rate",
            "kl_full_to_sparse_mean",
            "js_divergence_mean",
            "target_nll_delta_mean",
        ):
            result[key] = mean(stability, key)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    per_case = []
    dev_full = []
    dev_qk = []
    for topic in TOPICS:
        old_payload = records(
            args.old_root
            / f"length128000_{topic}"
            / "case_summary.json"
        )
        for window in DEV_WINDOWS:
            full = select(
                old_payload,
                topic=topic,
                window=window,
                method="full_attention",
            )
            old = select(
                old_payload,
                topic=topic,
                window=window,
                method="direct_countcap",
            )
            key1 = single_case(args.keypca1_root, topic, window)
            exact1 = single_case(
                args.diagnosis_root / "exact_top1",
                topic,
                window,
                method="exact_top_fraction",
            )
            key2 = single_case(
                args.diagnosis_root / "packed_qmse_top2",
                topic,
                window,
            )
            qk = single_case(args.qkmetric_dev_root, topic, window)
            dev_full.append(full)
            dev_qk.append(qk)
            per_case.append(
                {
                    "split": "development",
                    "topic": topic,
                    "window": window,
                    "full_ppl": float(full["ppl"]),
                    "old_countcap_retention": math.exp(
                        float(full["nll"]) - float(old["nll"])
                    ),
                    "keypca_1pct_retention": math.exp(
                        float(full["nll"]) - float(key1["nll"])
                    ),
                    "exact_1pct_retention": math.exp(
                        float(full["nll"]) - float(exact1["nll"])
                    ),
                    "keypca_2pct_retention": math.exp(
                        float(full["nll"]) - float(key2["nll"])
                    ),
                    "qkmetric_1pct_ppl": float(qk["ppl"]),
                    "qkmetric_1pct_retention": math.exp(
                        float(full["nll"]) - float(qk["nll"])
                    ),
                    "qkmetric_actual_attention_ratio": float(
                        qk["actual_attention_tokens_mean"]
                    )
                    / float(qk["history_tokens"]),
                    "qkmetric_index_ratio_of_full_kv": float(
                        qk["packed_index_ratio_of_full_kv"]
                    ),
                    "qkmetric_steady_speedup": float(
                        full["sparse_seconds_per_step"]
                    )
                    / float(qk["steady_sparse_seconds_per_step"]),
                    "qkmetric_measured_speedup": float(
                        full["sparse_decode_seconds"]
                    )
                    / float(qk["sparse_decode_seconds"]),
                }
            )

    holdout_full = []
    holdout_qk = []
    for topic in TOPICS:
        for window in HOLDOUT_WINDOWS:
            payload = records(
                args.holdout_root
                / f"{topic}_w{window}"
                / "case_summary.json"
            )
            full = select(
                payload,
                topic=topic,
                window=window,
                method="full_attention",
            )
            qk = select(
                payload,
                topic=topic,
                window=window,
                method="direct_countcap",
            )
            holdout_full.append(full)
            holdout_qk.append(qk)
            per_case.append(
                {
                    "split": "holdout",
                    "topic": topic,
                    "window": window,
                    "full_ppl": float(full["ppl"]),
                    "qkmetric_1pct_ppl": float(qk["ppl"]),
                    "qkmetric_1pct_retention": math.exp(
                        float(full["nll"]) - float(qk["nll"])
                    ),
                    "qkmetric_actual_attention_ratio": float(
                        qk["actual_attention_tokens_mean"]
                    )
                    / float(qk["history_tokens"]),
                    "qkmetric_index_ratio_of_full_kv": float(
                        qk["packed_index_ratio_of_full_kv"]
                    ),
                    "qkmetric_steady_speedup": float(
                        full["sparse_seconds_per_step"]
                    )
                    / float(qk["steady_sparse_seconds_per_step"]),
                    "qkmetric_measured_speedup": float(
                        full["sparse_decode_seconds"]
                    )
                    / float(qk["sparse_decode_seconds"]),
                    "top1_agreement": float(qk["top1_agreement"]),
                    "kl_full_to_sparse_mean": float(
                        qk["kl_full_to_sparse_mean"]
                    ),
                    "margin_certificate_rate": float(
                        qk["margin_certificate_rate"]
                    ),
                }
            )

    aggregates = [
        aggregate("development", dev_full, dev_qk),
        aggregate("holdout", holdout_full, holdout_qk),
        aggregate(
            "all_eight_windows",
            dev_full + holdout_full,
            dev_qk + holdout_qk,
        ),
    ]
    output = {
        "per_case": per_case,
        "aggregates": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", per_case)
    write_csv(args.output_dir / "aggregates.csv", aggregates)
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
