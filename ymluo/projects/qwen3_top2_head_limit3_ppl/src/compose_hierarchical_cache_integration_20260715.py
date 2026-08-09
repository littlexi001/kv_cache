from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_FIELDS = (
    "history_tokens",
    "remote_tokens",
    "query_tokens",
    "eval_tokens",
    "projection_dim",
    "candidate_fraction",
    "exact_cache_fraction",
)


def compose_hierarchical_cache_integration(
    rows: list[dict[str, Any]],
    full_ppl_by_topic: dict[str, float],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one integrated result is required")
    topics = [str(row["topic"]) for row in rows]
    if len(set(topics)) != len(topics):
        raise ValueError("integrated result topics must be unique")
    missing = sorted(set(topics) - set(full_ppl_by_topic))
    if missing:
        raise ValueError(f"missing full PPL references for: {missing}")

    protocol = {field: rows[0][field] for field in PROTOCOL_FIELDS}
    directory_backends = {
        str(row.get("directory_backend", "sorted")) for row in rows
    }
    if len(directory_backends) != 1:
        raise ValueError(
            f"directory backend mismatch: {sorted(directory_backends)}"
        )
    directory_backend = directory_backends.pop()
    for row in rows[1:]:
        for field, expected in protocol.items():
            if row[field] != expected:
                raise ValueError(
                    f"protocol mismatch for {field}: {row[field]} != {expected}"
                )

    topic_results: dict[str, dict[str, float]] = {}
    full_delta_nll = []
    implementation_delta_nll = []
    for row in rows:
        topic = str(row["topic"])
        physical_ppl = float(row["ppl"])
        algorithm_ppl = float(row["known_reference_ppl"])
        full_ppl = float(full_ppl_by_topic[topic])
        if min(physical_ppl, algorithm_ppl, full_ppl) <= 0.0:
            raise ValueError("PPL references must be positive")
        physical_over_algorithm = physical_ppl / algorithm_ppl
        physical_over_full = physical_ppl / full_ppl
        implementation_delta_nll.append(math.log(physical_over_algorithm))
        full_delta_nll.append(math.log(physical_over_full))
        topic_results[topic] = {
            "full_ppl": full_ppl,
            "algorithm_reference_ppl": algorithm_ppl,
            "physical_ppl": physical_ppl,
            "physical_over_algorithm": physical_over_algorithm,
            "physical_over_full": physical_over_full,
            "quality_retention_vs_full": 1.0 / physical_over_full,
            "cache_hit_rate": float(row["mean_cache_hit_rate"]),
        }

    final_storage = [
        float(row["hierarchical_over_final_length_full_kv"]) for row in rows
    ]
    capacity_storage = [
        float(row["hierarchical_over_capacity_equivalent_full_kv"])
        for row in rows
    ]
    combined_full_ratio = math.exp(sum(full_delta_nll) / len(full_delta_nll))
    combined_algorithm_ratio = math.exp(
        sum(implementation_delta_nll) / len(implementation_delta_nll)
    )
    return {
        "status": (
            "fused_integrated_quality_and_storage_validated"
            if directory_backend == "fused"
            else "integrated_quality_and_storage_validated_kernel_fusion_pending"
        ),
        "protocol": protocol | {"directory_backend": directory_backend},
        "topics": topic_results,
        "combined": {
            "physical_over_full": combined_full_ratio,
            "quality_retention_vs_full": 1.0 / combined_full_ratio,
            "physical_over_algorithm_reference": combined_algorithm_ratio,
            "max_physical_over_algorithm_reference": max(
                result["physical_over_algorithm"]
                for result in topic_results.values()
            ),
        },
        "physical_storage": {
            "persistent_gpu_bytes": int(rows[0]["hierarchical_persistent_gpu_bytes"]),
            "pinned_host_bytes": int(rows[0]["pinned_host_bytes"]),
            "final_length_full_kv_fraction_mean": sum(final_storage)
            / len(final_storage),
            "final_length_full_kv_fraction_max": max(final_storage),
            "capacity_equivalent_full_kv_fraction_mean": sum(capacity_storage)
            / len(capacity_storage),
        },
        "checks": {
            "implementation_drift_below_0p5_percent": max(
                result["physical_over_algorithm"]
                for result in topic_results.values()
            )
            <= 1.005,
            "quality_retention_at_least_95_percent": 1.0
            / combined_full_ratio
            >= 0.95,
            "persistent_gpu_kv_below_10_percent": max(final_storage) < 0.10,
        },
        "caveat": (
            "Sports and medicine were run concurrently and are used only for "
            "quality validation; use the isolated religion E2E result for speed."
            if directory_backend == "fused"
            else "The integrated cache currently uses the clarity-first sorted "
            "Python directory. Report component-composed CUDA speed separately "
            "until the fused hash-LRU path is wired into this lifecycle."
        ),
    }


def parse_full_references(values: list[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for value in values:
        topic, separator, ppl = value.partition("=")
        if not separator or not topic:
            raise ValueError(f"expected TOPIC=PPL, received {value!r}")
        output[topic] = float(ppl)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--full_reference", action="append", required=True, metavar="TOPIC=PPL"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    summary = compose_hierarchical_cache_integration(
        rows, parse_full_references(args.full_reference)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
