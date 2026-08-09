from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("bm25", "e5", "bm25_e5_rrf")
DISTANCE_BINS = (
    (0, 256, "0-256"),
    (256, 1024, "256-1K"),
    (1024, 4096, "1K-4K"),
    (4096, 16384, "4K-16K"),
    (16384, float("inf"), ">16K"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate repository scope from local boundary distance in code utility."
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--manifest_path")
    parser.add_argument("--scope_ids_path")
    parser.add_argument("--metadata_scope_key", default="repo_index")
    parser.add_argument("--manifest_scope_key", default="repo_index")
    parser.add_argument("--scope_name", default="repository")
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query = defaultdict(list)
    for row in rows:
        by_query[int(row["query_id"])].append(row)
    return {
        "candidate_windows": len(rows),
        "queries": len(by_query),
        "future_event_rate": mean([float(row["future_event"]) for row in rows]),
        "query_macro_future_event_rate": mean(
            [
                mean([float(row["future_event"]) for row in group])
                for group in by_query.values()
            ]
        ),
        "positive_future_utility_rate": mean(
            [float(row["delta_nll_b"] > 0) for row in rows]
        ),
        "mean_delta_nll_a": mean([float(row["delta_nll_a"]) for row in rows]),
        "mean_delta_nll_b": mean([float(row["delta_nll_b"]) for row in rows]),
        "query_macro_mean_delta_nll_b": mean(
            [
                mean([float(row["delta_nll_b"]) for row in group])
                for group in by_query.values()
            ]
        ),
    }


def distance_bin(distance: float) -> str:
    for low, high, label in DISTANCE_BINS:
        if low <= distance < high:
            return label
    raise AssertionError(distance)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    rows = read_jsonl(args.rows)
    metadata = {
        int(row["query_id"]): row for row in read_jsonl(data_dir / "metadata.jsonl")
    }
    manifest_path = Path(args.manifest_path) if args.manifest_path else data_dir / "segment_manifest.jsonl"
    scope_ids_path = Path(args.scope_ids_path) if args.scope_ids_path else data_dir / "base_block_scope_ids.npy"
    manifest = read_jsonl(manifest_path)
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    block_scope_ids = np.load(scope_ids_path, mmap_mode="r")
    base_count = len(block_scope_ids)
    block_tokens = int(summary["block_tokens"])
    reserved_end = (
        int(summary["query_offset_tokens"])
        + int(summary["source_tokens"])
        + int(summary["query_tokens"])
        + int(summary["target_tokens"])
    )

    starts = [int(row["start_token"]) for row in manifest]

    def locate_block(block_id: int) -> dict[str, Any] | None:
        center = block_id * block_tokens + block_tokens / 2
        index = bisect.bisect_right(starts, center) - 1
        if index < 0:
            return None
        segment = manifest[index]
        if center >= int(segment["end_token"]):
            return None
        original_base = 0 if int(segment["part"]) == 0 else reserved_end
        return {
            "scope_id": int(segment[args.manifest_scope_key]),
            "part": int(segment["part"]),
            "original_center": original_base + center - int(segment["start_token"]),
        }

    by_query = defaultdict(list)
    for row in rows:
        by_query[int(row["query_id"])].append(row)
    random95 = {
        query_id: float(
            np.quantile(
                [
                    float(row["delta_nll_b"])
                    for row in group
                    if "random" in row["origins"]
                ],
                0.95,
            )
        )
        for query_id, group in by_query.items()
    }

    enriched = []
    for row in rows:
        if not any(method in row["origins"] for method in METHODS):
            continue
        query_id = int(row["query_id"])
        query_scope = int(metadata[query_id][args.metadata_scope_key])
        block_ids = [int(block_id) for block_id in row["block_ids"]]
        item = dict(row)
        item["future_event"] = float(row["delta_nll_b"]) > random95[query_id]
        if any(block_id >= base_count for block_id in block_ids):
            item.update({"region": "heldout_source", "distance_tokens": None})
            enriched.append(item)
            continue
        locations = [locate_block(block_id) for block_id in block_ids]
        locations = [location for location in locations if location is not None]
        if not locations:
            item.update({"region": "unmapped_separator", "distance_tokens": None})
        elif len({(loc["scope_id"], loc["part"]) for loc in locations}) != 1:
            item.update({"region": "mixed_segment", "distance_tokens": None})
        elif locations[0]["scope_id"] != query_scope:
            item.update({"region": f"other_{args.scope_name}", "distance_tokens": None})
        else:
            part = locations[0]["part"]
            center = float(np.mean([loc["original_center"] for loc in locations]))
            if part == 0:
                distance = max(0.0, float(summary["query_offset_tokens"]) - center)
                region = f"same_{args.scope_name}_before_hole"
            else:
                distance = max(0.0, center - reserved_end)
                region = f"same_{args.scope_name}_after_hole"
            item.update(
                {
                    "region": region,
                    "distance_tokens": distance,
                    "distance_bin": distance_bin(distance),
                }
            )
        enriched.append(item)

    region_groups = defaultdict(list)
    distance_groups = defaultdict(list)
    for row in enriched:
        region_groups[row["region"]].append(row)
        if "distance_bin" in row:
            distance_groups[(row["region"], row["distance_bin"])].append(row)

    output = {
        "source": f"continuation utility split by {args.scope_name}, side, and token distance",
        "protocol": {
            "future_B_is_never_used_for_retrieval": True,
            "future_B_is_used_only_for_posthoc_utility_measurement": True,
            "base_contains_text_before_and_after_the_removed_source_query_target_hole": True,
            "same_scope_after_hole_is_later_text_relative_to_the_target": True,
            "scope_name": args.scope_name,
            "distance_is_approximate_window_center_distance_to_removed_hole_boundary": True,
        },
        "overall_retrieval_candidates": summarize(enriched),
        "by_region": {
            region: summarize(group) for region, group in sorted(region_groups.items())
        },
        "same_repo_by_side_and_distance": [
            {
                "region": region,
                "distance_bin": bin_name,
                **summarize(group),
            }
            for (region, bin_name), group in sorted(distance_groups.items())
        ],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
