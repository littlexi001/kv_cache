from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched 32K ablations.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_methods", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    reference = load_json(args.input_dir / "full_kv.json")
    method_paths = sorted(
        path
        for path in args.input_dir.glob("*.json")
        if path.name != "full_kv.json"
    )
    if args.expected_methods and len(method_paths) != args.expected_methods:
        raise ValueError(
            f"expected {args.expected_methods} sparse methods, found {len(method_paths)}"
        )
    full_prefill = float(reference["prefill_seconds"])
    full_decode = float(reference["synchronized_model_forward_seconds"])
    rows: list[dict[str, Any]] = []
    for path in method_paths:
        method = load_json(path)
        sparse_prefill = float(method["prefill_seconds"])
        conversion = float(method["cache_conversion_seconds"])
        sparse_decode = float(method["online_seconds"])
        rows.append(
            {
                "method": path.stem,
                "projection_dim": method["projection_dim"],
                "index_bits": method["index_bits"],
                "candidate_fraction": method["candidate_fraction"],
                "attention_fraction": method["attention_fraction"],
                "candidate_selection_mode": method["candidate_selection_mode"],
                "rerank_selection_mode": method["rerank_selection_mode"],
                "stream_group_size": method["stream_group_size"],
                "exact_cache_fraction": method["exact_cache_fraction"],
                "directory_backend": method["directory_backend"],
                "ppl": method["ppl"],
                "full_ppl": reference["ppl"],
                "quality_retention": float(reference["ppl"]) / float(method["ppl"]),
                "logical_attention_ratio": method["attention_fraction"],
                "reference_full_gpu_kv_bytes": method[
                    "original_remote_full_gpu_kv_bytes"
                ],
                "physical_gpu_tensor_bytes": method[
                    "hierarchical_persistent_gpu_bytes"
                ],
                "physical_gpu_kv_ratio": method["hierarchical_over_final_length_full_kv"],
                "physical_minus_logical_ratio": float(
                    method["hierarchical_over_final_length_full_kv"]
                )
                - float(method["attention_fraction"]),
                "physical_over_logical_ratio": float(
                    method["hierarchical_over_final_length_full_kv"]
                )
                / float(method["attention_fraction"]),
                "pinned_host_bytes": method["pinned_host_bytes"],
                "pinned_host_over_remote_full_kv": int(method["pinned_host_bytes"])
                / int(method["original_remote_full_gpu_kv_bytes"]),
                "cache_hit_rate": method["mean_cache_hit_rate"],
                "conversion_seconds": conversion,
                "decode_speedup": full_decode / sparse_decode,
                "protocol_total_speedup": (full_prefill + full_decode)
                / (sparse_prefill + conversion + sparse_decode),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"reference": reference, "rows": rows}
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in rows:
        print(
            f"{row['method']}: retention={row['quality_retention']:.4f} "
            f"gpu_kv={row['physical_gpu_kv_ratio']:.4f} "
            f"decode={row['decode_speedup']:.3f}x"
        )


if __name__ == "__main__":
    main()
