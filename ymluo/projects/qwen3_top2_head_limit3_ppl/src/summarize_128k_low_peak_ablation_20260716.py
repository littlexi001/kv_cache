from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched 128K peak-memory paths.")
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--dynamic", required=True, type=Path)
    parser.add_argument("--offloaded", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_matched(
    full: dict[str, Any], dynamic: dict[str, Any], offloaded: dict[str, Any]
) -> None:
    common = ("topic", "window_index", "history_tokens", "query_tokens", "eval_tokens")
    for key in common:
        values = {full[key], dynamic[key], offloaded[key]}
        if len(values) != 1:
            raise ValueError(f"unmatched {key}: {values}")
    if full["target_token_ids"] != dynamic["target_token_ids"]:
        raise ValueError("Full and dynamic target token ids differ")
    if full["target_token_ids"] != offloaded["target_token_ids"]:
        raise ValueError("Full and offloaded target token ids differ")
    sparse_fields = (
        "projection_dim",
        "index_bits",
        "candidate_fraction",
        "attention_fraction",
        "candidate_selection_mode",
        "exact_cache_fraction",
        "stream_group_size",
        "directory_backend",
    )
    for key in sparse_fields:
        if dynamic[key] != offloaded[key]:
            raise ValueError(f"dynamic/offloaded mismatch for {key}")


def summarize(
    full: dict[str, Any], dynamic: dict[str, Any], offloaded: dict[str, Any]
) -> dict[str, Any]:
    validate_matched(full, dynamic, offloaded)
    full_peak = int(full["process_peak_gpu_allocated_during_prefill_decode"])
    full_total = float(full["prefill_seconds"]) + float(
        full["synchronized_model_forward_seconds"]
    )
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "method": "full_kv",
            "ppl": full["ppl"],
            "quality_retention": 1.0,
            "prefill_seconds": full["prefill_seconds"],
            "conversion_seconds": 0.0,
            "decode_seconds": full["synchronized_model_forward_seconds"],
            "protocol_seconds": full_total,
            "protocol_speedup": 1.0,
            "persistent_gpu_bytes_ratio": 1.0,
            "peak_gpu_bytes": full_peak,
            "peak_gpu_bytes_ratio": 1.0,
            "pinned_host_over_full_kv": 0.0,
        }
    )
    for name, payload in (("dynamic", dynamic), ("offloaded_exact", offloaded)):
        protocol = (
            float(payload["prefill_seconds"])
            + float(payload["cache_conversion_seconds"])
            + float(payload["online_seconds"])
        )
        rows.append(
            {
                "method": name,
                "ppl": payload["ppl"],
                "quality_retention": float(full["ppl"]) / float(payload["ppl"]),
                "prefill_seconds": payload["prefill_seconds"],
                "conversion_seconds": payload["cache_conversion_seconds"],
                "decode_seconds": payload["online_seconds"],
                "protocol_seconds": protocol,
                "protocol_speedup": full_total / protocol,
                "persistent_gpu_bytes_ratio": payload[
                    "hierarchical_over_final_length_full_kv"
                ],
                "peak_gpu_bytes": payload[
                    "process_peak_gpu_allocated_during_prefill_conversion"
                ],
                "peak_gpu_bytes_ratio": int(
                    payload["process_peak_gpu_allocated_during_prefill_conversion"]
                )
                / full_peak,
                "pinned_host_over_full_kv": int(payload["pinned_host_bytes"])
                / int(payload["original_remote_full_gpu_kv_bytes"]),
            }
        )
    dynamic_peak = int(dynamic["process_peak_gpu_allocated_during_prefill_conversion"])
    offloaded_peak = int(offloaded["process_peak_gpu_allocated_during_prefill_conversion"])
    token_nll_delta = [
        float(right) - float(left)
        for left, right in zip(dynamic["token_nll"], offloaded["token_nll"])
    ]
    return {
        "protocol": "matched_128k_religion_w0_256_targets",
        "rows": rows,
        "offloaded_peak_reduction_vs_dynamic": 1.0 - offloaded_peak / dynamic_peak,
        "offloaded_protocol_time_ratio_vs_dynamic": rows[2]["protocol_seconds"]
        / rows[1]["protocol_seconds"],
        "offloaded_vs_dynamic_mean_token_nll_delta": sum(token_nll_delta)
        / len(token_nll_delta),
        "offloaded_vs_dynamic_max_abs_token_nll_delta": max(
            abs(value) for value in token_nll_delta
        ),
    }


def main() -> None:
    args = parse_args()
    summary = summarize(load(args.full), load(args.dynamic), load(args.offloaded))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
