#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "score": sum(fnum(row, "score") for row in rows) / max(1, len(rows)),
        "valid_format_rate": sum(bool(row.get("longbench_v2_pred", "")) for row in rows) / max(1, len(rows)),
        "mean_kv_ratio": sum(fnum(row, "keep_fraction") for row in rows) / max(1, len(rows)),
        "online_seconds": sum(fnum(row, "online_seconds") for row in rows),
        "total_seconds": sum(fnum(row, "total_seconds") for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [row for path in args.results for row in read_csv(Path(path))]
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    if "full_kv" not in by_method or "ours_page_gather" not in by_method:
        raise RuntimeError(f"Expected full_kv and ours_page_gather, got {sorted(by_method)}")

    full_index = {row["sample_id"]: row for row in by_method["full_kv"]}
    ours_index = {row["sample_id"]: row for row in by_method["ours_page_gather"]}
    common_ids = [sample_id for sample_id in full_index if sample_id in ours_index]
    full = [full_index[sample_id] for sample_id in common_ids]
    ours = [ours_index[sample_id] for sample_id in common_ids]
    full_summary = summarize(full)
    ours_summary = summarize(ours)
    agreements = sum(
        full_index[sample_id].get("longbench_v2_pred", "")
        == ours_index[sample_id].get("longbench_v2_pred", "")
        for sample_id in common_ids
    )
    payload = {
        "matched_samples": len(common_ids),
        "full": full_summary,
        "ours": ours_summary,
        "score_over_full": ours_summary["score"] / full_summary["score"] if full_summary["score"] > 0 else None,
        "prediction_agreement": agreements / max(1, len(common_ids)),
        "online_speed": full_summary["online_seconds"] / max(1e-12, ours_summary["online_seconds"]),
        "total_speed": full_summary["total_seconds"] / max(1e-12, ours_summary["total_seconds"]),
    }

    domain_rows: list[dict[str, Any]] = []
    domains = sorted({row.get("domain", "") for row in full})
    for domain in domains:
        domain_ids = [sample_id for sample_id in common_ids if full_index[sample_id].get("domain", "") == domain]
        domain_full = [full_index[sample_id] for sample_id in domain_ids]
        domain_ours = [ours_index[sample_id] for sample_id in domain_ids]
        full_score = summarize(domain_full)["score"]
        ours_score = summarize(domain_ours)["score"]
        domain_rows.append(
            {
                "domain": domain,
                "samples": len(domain_ids),
                "full_score": full_score,
                "ours_score": ours_score,
                "score_over_full": ours_score / full_score if full_score > 0 else "",
                "prediction_agreement": sum(
                    full_index[sample_id].get("longbench_v2_pred", "")
                    == ours_index[sample_id].get("longbench_v2_pred", "")
                    for sample_id in domain_ids
                )
                / max(1, len(domain_ids)),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_rows[0]) if domain_rows else ["domain"])
        writer.writeheader()
        writer.writerows(domain_rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
