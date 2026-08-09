from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compose_physical_attention_path(
    profile: dict[str, Any],
    directory: dict[str, Any],
    data_path: dict[str, Any],
    *,
    full_attention_ms: float,
) -> dict[str, Any]:
    histories = {
        int(profile["history_count"]),
        int(directory["history_count"]),
        int(data_path["history_count"]),
    }
    if len(histories) != 1:
        raise ValueError(f"history count mismatch: {sorted(histories)}")
    if profile.get("candidate_mode") != "shared_mean":
        raise ValueError("physical path summary requires shared_mean retrieval")
    if float(profile.get("budget_fraction", 0.0)) != float(
        data_path.get("attention_fraction", -1.0)
    ):
        raise ValueError("retrieval and attention budget mismatch")

    online_ms = float(profile["online_composed_ms"])
    reference_final_ms = float(profile["final_sparse_attention_ms"])
    retrieval_prefix_ms = online_ms - reference_final_ms
    if retrieval_prefix_ms <= 0.0:
        raise ValueError("online retrieval prefix must be positive")
    directory_ms = float(directory["fused_directory_total_ms_per_layer"])
    physical_data_ms = float(
        data_path["mapped_host_fill_cache_attention_ms_per_layer"]
    )
    total_ms = retrieval_prefix_ms + directory_ms + physical_data_ms
    if total_ms <= 0.0 or full_attention_ms <= 0.0:
        raise ValueError("latencies must be positive")

    required_zero_error_fields = (
        "max_abs_error",
        "resident_cache_attention_max_abs_error",
    )
    numerical_errors = {
        field: float(data_path[field]) for field in required_zero_error_fields
    }
    return {
        "status": "component_validated_not_yet_hf_integrated",
        "history_count": histories.pop(),
        "candidate_mode": profile["candidate_mode"],
        "budget_fraction": float(profile["budget_fraction"]),
        "cache_fraction": float(directory["exact_cache_fraction"]),
        "cache_hit_rate": float(data_path["cache_hit_rate"]),
        "miss_read_fraction": float(data_path["selected_fraction"]),
        "resident_storage_fraction": float(directory["total_resident_fraction"]),
        "latency_ms_per_layer": {
            "full_attention": float(full_attention_ms),
            "online_retrieval_prefix": retrieval_prefix_ms,
            "fused_cache_directory": directory_ms,
            "mapped_miss_fill_and_cache_slot_attention": physical_data_ms,
            "ours_total": total_ms,
        },
        "attention_path_speedup": float(full_attention_ms) / total_ms,
        "numerical_errors": numerical_errors,
        "is_numerically_exact": all(
            value == 0.0 for value in numerical_errors.values()
        ),
        "is_below_ten_percent_resident": float(
            directory["total_resident_fraction"]
        )
        < 0.10,
        "is_above_2p5x_attention_path": float(full_attention_ms) / total_ms
        >= 2.5,
    }


def read_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected one profile row in {path}, found {len(value)}")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--full_attention_ms", type=float, default=2.628)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = compose_physical_attention_path(
        read_profile(args.profile),
        read_profile(args.directory),
        read_profile(args.data_path),
        full_attention_ms=args.full_attention_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
