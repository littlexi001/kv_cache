#!/usr/bin/env python
"""Aggregate matched native-128K Value-sketch topic runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Named run in NAME=PATH or NAME=PATH::VARIANT form; may be repeated. "
            "The variant selector is required when a root contains multiple sparse rows."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path, str | None]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    path_text, selector_separator, selector = path.partition("::")
    return name, Path(path_text), selector if selector_separator else None


def sparse_row(
    document: dict[str, Any], selector: str | None
) -> dict[str, Any]:
    rows = document["rows"]
    candidates = [row for row in rows if row["method"] != "full_attention"]
    if selector is not None:
        candidates = [
            row
            for row in candidates
            if row.get("variant") == selector or row.get("method") == selector
        ]
    if len(candidates) != 1:
        raise ValueError(f"expected one sparse row, got {len(candidates)}")
    return candidates[0]


def summarize_root(root: Path, selector: str | None) -> dict[str, Any]:
    paths = sorted(root.glob("*/summary.json"))
    if not paths:
        raise FileNotFoundError(f"no topic summaries under {root}")

    topics: dict[str, Any] = {}
    full_nll_numerator = 0.0
    sparse_nll_numerator = 0.0
    token_count = 0
    full_step_numerator = 0.0
    sparse_step_numerator = 0.0
    total_index_numerator = 0.0
    key_index_numerator = 0.0
    value_index_numerator = 0.0
    refinement_numerator = 0.0
    refinement_head_count = 0

    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not (path.parent / "ALL_COMPLETE").exists():
            raise RuntimeError(f"incomplete topic run: {path.parent}")
        full = next(row for row in document["rows"] if row["method"] == "full_attention")
        sparse = sparse_row(document, selector)
        variant = str(sparse.get("variant", ""))
        tokens = int(sparse["tokens"])
        if int(full["tokens"]) != tokens:
            raise ValueError(f"token mismatch in {path}")
        topic = str(document["topic"])
        retention = math.exp(float(full["nll"]) - float(sparse["nll"]))
        speedup = float(full["steady_sparse_seconds_per_step"]) / float(
            sparse["steady_sparse_seconds_per_step"]
        )
        refinement_count = int(sparse.get("value_refinement_count", 0) or 0)
        head_count = int(sparse.get("value_refinement_head_count", 0) or 0)
        refinement_rate = (
            refinement_count / head_count if head_count else 0.0
        )
        value_ratio = float(
            sparse.get("packed_value_sketch_ratio_of_full_kv", 0.0) or 0.0
        )
        # The first corrected fixed-rank runs predate the explicit Value-rate
        # summary field. Recover the registered packed-format rate from the
        # variant name: INT4 codes plus one FP16 block scale per 128 values.
        if value_ratio == 0.0 and "valuesketch" in variant:
            for rank in (8, 16, 32):
                if f"valuesketch{rank}i4" in variant:
                    value_ratio = rank * (0.5 + 2.0 / 128.0) / 512.0
                    break
        key_ratio = float(
            sparse.get("packed_index_ratio_of_full_kv", 0.0) or 0.0
        )
        total_ratio = float(
            sparse.get("packed_total_auxiliary_ratio_of_full_kv", 0.0) or 0.0
        )
        if total_ratio == 0.0:
            total_ratio = key_ratio + value_ratio
        topics[topic] = {
            "tokens": tokens,
            "full_nll": float(full["nll"]),
            "sparse_nll": float(sparse["nll"]),
            "quality_retention": retention,
            "steady_whole_model_speedup": speedup,
            "fixed_index_overhead_seconds": float(
                sparse.get("fixed_sparse_overhead_seconds", 0.0) or 0.0
            ),
            "key_index_ratio_of_full_kv": key_ratio,
            "value_index_ratio_of_full_kv": value_ratio,
            "total_auxiliary_ratio_of_full_kv": total_ratio,
            "value_refinement_rate": refinement_rate,
        }
        full_nll_numerator += float(full["nll"]) * tokens
        sparse_nll_numerator += float(sparse["nll"]) * tokens
        full_step_numerator += float(full["steady_sparse_seconds_per_step"]) * tokens
        sparse_step_numerator += float(sparse["steady_sparse_seconds_per_step"]) * tokens
        key_index_numerator += topics[topic]["key_index_ratio_of_full_kv"] * tokens
        value_index_numerator += topics[topic]["value_index_ratio_of_full_kv"] * tokens
        total_index_numerator += topics[topic]["total_auxiliary_ratio_of_full_kv"] * tokens
        refinement_numerator += refinement_count
        refinement_head_count += head_count
        token_count += tokens

    full_nll = full_nll_numerator / token_count
    sparse_nll = sparse_nll_numerator / token_count
    return {
        "root": str(root),
        "topics": len(topics),
        "tokens": token_count,
        "full_nll": full_nll,
        "sparse_nll": sparse_nll,
        "full_ppl": math.exp(full_nll),
        "sparse_ppl": math.exp(sparse_nll),
        "quality_retention": math.exp(full_nll - sparse_nll),
        "worst_topic_quality_retention": min(
            item["quality_retention"] for item in topics.values()
        ),
        "steady_whole_model_speedup": full_step_numerator / sparse_step_numerator,
        "key_index_ratio_of_full_kv": key_index_numerator / token_count,
        "value_index_ratio_of_full_kv": value_index_numerator / token_count,
        "total_auxiliary_ratio_of_full_kv": total_index_numerator / token_count,
        "value_refinement_rate": (
            refinement_numerator / refinement_head_count
            if refinement_head_count
            else 0.0
        ),
        "per_topic": topics,
    }


def main() -> None:
    args = parse_args()
    report = {
        "schema": "qksieve_valuesketch_native128k_comparison_v1",
        "runs": {
            name: summarize_root(path, selector)
            for name, path, selector in map(parse_run, args.run)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
