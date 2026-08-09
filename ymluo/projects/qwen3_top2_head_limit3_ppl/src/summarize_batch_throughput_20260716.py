from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_cases", type=int, default=6)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    payloads = [load(path) for path in sorted(args.input_dir.glob("l*_b*.json"))]
    if len(payloads) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} batch cases, found {len(payloads)}"
        )
    baseline = {
        int(row["history_tokens"]): row
        for row in payloads
        if int(row["batch_size"]) == 1
    }
    lengths = {int(row["history_tokens"]) for row in payloads}
    if set(baseline) != lengths:
        raise ValueError("each context length requires a batch=1 baseline")

    rows = []
    for payload in sorted(
        payloads, key=lambda row: (int(row["history_tokens"]), int(row["batch_size"]))
    ):
        length = int(payload["history_tokens"])
        batch = int(payload["batch_size"])
        reference = baseline[length]
        evaluated_tokens = batch * int(payload["decode_steps_per_sequence"])
        full_protocol_seconds = float(payload["full_prefill_seconds"]) + float(
            payload["full_forward_seconds"]
        )
        sparse_protocol_seconds = (
            float(payload["sparse_prefill_seconds"])
            + float(payload["sparse_conversion_seconds"])
            + float(payload["sparse_forward_seconds"])
        )
        full_tps = float(payload["full_tokens_per_second"])
        sparse_tps = float(payload["sparse_tokens_per_second"])
        rows.append(
            {
                "history_tokens": length,
                "batch_size": batch,
                "quality_retention": float(payload["full_mean_ppl"])
                / float(payload["sparse_mean_ppl"]),
                "full_decode_tokens_per_second": full_tps,
                "sparse_decode_tokens_per_second": sparse_tps,
                "decode_throughput_speedup": float(payload["throughput_speedup"]),
                "full_scaling_vs_b1": full_tps
                / float(reference["full_tokens_per_second"]),
                "sparse_scaling_vs_b1": sparse_tps
                / float(reference["sparse_tokens_per_second"]),
                "full_batch_efficiency": full_tps
                / float(reference["full_tokens_per_second"])
                / batch,
                "sparse_batch_efficiency": sparse_tps
                / float(reference["sparse_tokens_per_second"])
                / batch,
                "full_protocol_tokens_per_second": evaluated_tokens
                / full_protocol_seconds,
                "sparse_protocol_tokens_per_second": evaluated_tokens
                / sparse_protocol_seconds,
                "protocol_speedup": full_protocol_seconds / sparse_protocol_seconds,
                "persistent_gpu_tensor_bytes_ratio": float(
                    payload["sparse_over_final_full_kv"]
                ),
                "peak_gpu_bytes_ratio": float(payload["sparse_peak_gpu_allocated_bytes"])
                / float(payload["full_peak_gpu_allocated_bytes"]),
                "cache_hit_rate": float(payload["mean_cache_hit_rate"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
