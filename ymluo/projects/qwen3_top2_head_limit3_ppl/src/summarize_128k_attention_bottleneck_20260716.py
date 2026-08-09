from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "index_order",
    "attention_fraction",
    "cache_hit_rate",
    "attention_count_per_kv_head",
    "mapped_host_fill_cache_ms_per_layer",
    "packed_final_attention_ms_per_layer",
    "mapped_host_fill_cache_attention_ms_per_layer",
    "online_address_sort_metadata_ms_per_layer",
    "online_address_sort_fill_cache_attention_ms_per_layer",
    "online_miss_compact_sort_fill_cache_attention_ms_per_layer",
    "online_address_sort_max_abs_error",
    "online_miss_sort_max_abs_error",
    "hybrid_mapped_attention_cache_update_ms_per_layer",
    "gqa_hybrid_mapped_attention_cache_update_ms_per_layer",
    "hybrid_pack_ms_per_layer",
    "hybrid_pack_cache_update_ms_per_layer",
    "hybrid_pack_attention_ms_per_layer",
    "hybrid_pack_cache_update_attention_ms_per_layer",
    "gather_resident_cache_ms_per_layer",
    "selected_sdpa_attention_ms_per_layer",
    "gather_resident_cache_sdpa_ms_per_layer",
    "resident_cache_attention_max_abs_error",
    "selected_sdpa_attention_max_abs_error",
    "hybrid_max_abs_error",
    "gqa_hybrid_max_abs_error",
    "hybrid_pack_max_abs_error",
)

ERROR_THRESHOLDS = {
    "resident_cache_attention_max_abs_error": 0.02,
    "selected_sdpa_attention_max_abs_error": 0.125,
    "hybrid_max_abs_error": 0.02,
    "gqa_hybrid_max_abs_error": 0.02,
    "hybrid_pack_max_abs_error": 0.02,
    "online_address_sort_max_abs_error": 0.02,
    "online_miss_sort_max_abs_error": 0.02,
}


def positive_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("latencies must be positive")
    return numerator / denominator


def summarize_payload(
    case: str, payload: dict[str, float | str]
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"case": case}
    row.update({field: payload.get(field) for field in FIELDS})
    missing = [field for field in FIELDS if row[field] is None]
    if missing:
        raise ValueError(f"{case}: missing benchmark fields: {missing}")
    for field, threshold in ERROR_THRESHOLDS.items():
        if float(row[field]) > threshold:
            raise ValueError(
                f"{case}: {field}={row[field]} exceeds {threshold}"
            )

    miss_fill = float(row["mapped_host_fill_cache_attention_ms_per_layer"])
    row["direct_gqa_over_miss_fill_latency"] = positive_ratio(
        float(row["gqa_hybrid_mapped_attention_cache_update_ms_per_layer"]),
        miss_fill,
    )
    row["direct_query_head_over_miss_fill_latency"] = positive_ratio(
        float(row["hybrid_mapped_attention_cache_update_ms_per_layer"]),
        miss_fill,
    )
    row["pack_then_attention_over_miss_fill_latency"] = positive_ratio(
        float(row["hybrid_pack_cache_update_attention_ms_per_layer"]),
        miss_fill,
    )
    row["online_address_sort_over_current_order_latency"] = positive_ratio(
        float(row["online_address_sort_fill_cache_attention_ms_per_layer"]),
        miss_fill,
    )
    row["online_miss_sort_over_current_order_latency"] = positive_ratio(
        float(row["online_miss_compact_sort_fill_cache_attention_ms_per_layer"]),
        miss_fill,
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(summarize_payload(path.stem, payload))
    if not rows:
        raise RuntimeError(f"no benchmark JSON files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
