from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 128K long decode pair.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    full = load_json(args.input_dir / "full_kv.json")
    sparse = load_json(args.input_dir / "pca64_int4_top1p5_stream2_cache3p2.json")
    full_prefill = float(full["prefill_seconds"])
    full_decode = float(full["synchronized_model_forward_seconds"])
    sparse_prefill = float(sparse["prefill_seconds"])
    conversion = float(sparse["cache_conversion_seconds"])
    sparse_decode = float(sparse["online_seconds"])
    payload = {
        "protocol": "128K history + 256 query + 2048 target tokens, religion window3",
        "full_ppl": full["ppl"],
        "sparse_ppl": sparse["ppl"],
        "quality_retention": float(full["ppl"]) / float(sparse["ppl"]),
        "physical_gpu_kv_ratio": sparse["hierarchical_over_final_length_full_kv"],
        "full_prefill_seconds": full_prefill,
        "sparse_prefill_seconds": sparse_prefill,
        "conversion_seconds": conversion,
        "full_decode_seconds": full_decode,
        "sparse_decode_seconds": sparse_decode,
        "decode_speedup": full_decode / sparse_decode,
        "protocol_total_speedup": (full_prefill + full_decode)
        / (sparse_prefill + conversion + sparse_decode),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
