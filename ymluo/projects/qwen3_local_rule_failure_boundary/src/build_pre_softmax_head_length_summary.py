from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


def encode_f16(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f2").tobytes(order="C")).decode("ascii")


def encode_f32(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes(order="C")).decode("ascii")


def encode_u32(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<u4").tobytes(order="C")).decode("ascii")


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def build(data_dir: Path, output: Path) -> dict[str, Any]:
    paths = list(data_dir.glob("length_*.json")) + list(data_dir.glob("length_*.json.gz"))
    if not paths:
        raise FileNotFoundError(f"no length JSON files under {data_dir}")
    rows = [read_json(path) for path in paths]
    rows.sort(key=lambda row: int(row["target_context_tokens"]))
    first_attention = rows[0]["attention"]
    roles = list(first_attention["role_order"])
    layers = len(first_attention["head_role_logit_mean"])
    heads = len(first_attention["head_role_logit_mean"][0])
    shape = (len(rows), len(roles), layers, heads)
    logits = np.empty(shape, dtype=np.float32)
    masses = np.empty(shape, dtype=np.float32)
    best_ranks = np.empty(shape, dtype=np.uint32)
    logsumexp = np.empty((len(rows), layers, heads), dtype=np.float32)
    max_logits = np.empty_like(logsumexp)
    lengths: list[int] = []
    key_lengths: list[int] = []

    for row_index, row in enumerate(rows):
        attention = row["attention"]
        if list(attention["role_order"]) != roles:
            raise ValueError(f"role order changed at row {row_index}")
        # Source layout is [layer, head, role]; browser layout is
        # [length, role, layer, head] so one selected role is contiguous.
        logits[row_index] = np.asarray(attention["head_role_logit_mean"], dtype=np.float32).transpose(2, 0, 1)
        masses[row_index] = np.asarray(attention["head_role_mass"], dtype=np.float32).transpose(2, 0, 1)
        best_ranks[row_index] = np.asarray(attention["head_role_best_rank"], dtype=np.uint32).transpose(2, 0, 1)
        logsumexp[row_index] = np.asarray(attention["head_logsumexp"], dtype=np.float32)
        max_logits[row_index] = np.asarray(attention["head_max_logit"], dtype=np.float32)
        lengths.append(int(row["target_context_tokens"]))
        key_lengths.append(int(attention["key_length"]))

    payload = {
        "schema_version": 1,
        "model": rows[0]["model"],
        "code_mode": rows[0]["code_mode"],
        "gold_codes": rows[0]["gold_codes"],
        "lengths": lengths,
        "key_lengths": key_lengths,
        "role_order": roles,
        "num_layers": layers,
        "num_attention_heads": heads,
        "shape": list(shape),
        "storage_dtype": "float16 logits / float32 probability mass, little-endian base64",
        "aggregation": {
            "role_logit": "mean raw QK/sqrt(d) over positions in the marked role, per query head",
            "role_mass": "sum of exact post-softmax probability over positions in the marked role, per query head",
        },
        "role_logits_f16_b64": encode_f16(logits),
        # Individual heads can assign less than 2^-24 probability to a role at
        # long lengths.  Float16 would silently turn those valid values into 0.
        "role_mass_f32_b64": encode_f32(masses),
        "role_best_rank_u32_b64": encode_u32(best_ranks),
        "head_logsumexp_f16_b64": encode_f16(logsumexp),
        "head_max_logit_f16_b64": encode_f16(max_logits),
    }
    write_gzip_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact per-head evidence diagnostics across all lengths.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args.data_dir, args.output)
    print(json.dumps({"output": str(args.output), "shape": payload["shape"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
