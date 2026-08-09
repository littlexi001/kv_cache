from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 128K speed Pareto sweep.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reference_for(input_dir: Path, topic: str) -> dict[str, Any]:
    path = input_dir / f"full_{topic}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_json(path)


def summarize_one(method_path: Path, input_dir: Path) -> dict[str, Any]:
    method = load_json(method_path)
    reference = reference_for(input_dir, str(method["topic"]))
    full_prefill = float(reference["prefill_seconds"])
    full_decode = float(reference["synchronized_model_forward_seconds"])
    sparse_prefill = float(method["prefill_seconds"])
    sparse_conversion = float(method["cache_conversion_seconds"])
    sparse_decode = float(method["online_seconds"])
    return {
        "method": method_path.stem,
        "topic": method["topic"],
        "projection_dim": method["projection_dim"],
        "candidate_fraction": method["candidate_fraction"],
        "exact_cache_fraction": method["exact_cache_fraction"],
        "stream_group_size": method["stream_group_size"],
        "candidate_refresh_interval": method["candidate_refresh_interval"],
        "host_append_mode": method.get("host_append_mode", "unrecorded"),
        "conversion_mode": method.get("conversion_mode", "unrecorded"),
        "ppl": method["ppl"],
        "full_ppl": reference["ppl"],
        "quality_retention": float(reference["ppl"]) / float(method["ppl"]),
        "physical_gpu_kv_ratio": method["hierarchical_over_final_length_full_kv"],
        "cache_hit_rate": method["mean_cache_hit_rate"],
        "prefill_seconds": sparse_prefill,
        "conversion_seconds": sparse_conversion,
        "decode_seconds": sparse_decode,
        "full_decode_seconds": full_decode,
        "decode_speedup": full_decode / sparse_decode,
        "protocol_total_speedup": (full_prefill + full_decode)
        / (sparse_prefill + sparse_conversion + sparse_decode),
    }


def main() -> None:
    args = parse_args()
    method_paths = sorted(
        path for path in args.input_dir.glob("*.json") if not path.stem.startswith("full_")
    )
    if not method_paths:
        raise RuntimeError(f"no method JSON files found in {args.input_dir}")
    rows = [summarize_one(path, args.input_dir) for path in method_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2, ensure_ascii=False)
    for row in rows:
        print(
            f"{row['method']}: retention={row['quality_retention']:.4f} "
            f"gpu_kv={row['physical_gpu_kv_ratio']:.4f} "
            f"decode={row['decode_speedup']:.3f}x "
            f"protocol={row['protocol_total_speedup']:.3f}x"
        )


if __name__ == "__main__":
    main()
