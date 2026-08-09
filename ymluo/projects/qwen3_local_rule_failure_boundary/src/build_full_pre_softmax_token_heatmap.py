from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def encode_f16(values: np.ndarray) -> str:
    raw = np.asarray(values, dtype="<f2").tobytes(order="C")
    return base64.b64encode(raw).decode("ascii")


def encode_u32(values: np.ndarray) -> str:
    raw = np.asarray(values, dtype="<u4").tobytes(order="C")
    return base64.b64encode(raw).decode("ascii")


def write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def build(root: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tokens = read_gzip_json(root / manifest["files"]["tokens"])
    token_ids = np.frombuffer(
        base64.b64decode(tokens["token_ids_u32_b64"]), dtype="<u4"
    )
    unique_ids, inverse, counts = np.unique(
        token_ids, return_inverse=True, return_counts=True
    )
    layers = int(manifest["num_layers"])
    heads = int(manifest["num_attention_heads"])
    mean_logits = np.empty((len(unique_ids), layers, heads), dtype=np.float32)
    probability_mass = np.empty_like(mean_logits)

    for layer in range(layers):
        for head in range(heads):
            relative = manifest["files"]["head_pattern"].format(
                layer=layer, head=head
            )
            shard = read_gzip_json(root / relative)
            logits = np.frombuffer(
                base64.b64decode(shard["logits_f16_b64"]), dtype="<f2"
            ).astype(np.float32)
            if len(logits) != len(token_ids):
                raise ValueError(
                    f"length mismatch in {relative}: {len(logits)} != {len(token_ids)}"
                )
            probabilities = np.exp(logits - float(shard["logsumexp"])).astype(
                np.float32
            )
            logit_sums = np.bincount(
                inverse, weights=logits, minlength=len(unique_ids)
            )
            probability_sums = np.bincount(
                inverse, weights=probabilities, minlength=len(unique_ids)
            )
            mean_logits[:, layer, head] = logit_sums / counts
            probability_mass[:, layer, head] = probability_sums

    payload = {
        "schema_version": 1,
        "model": manifest["model"],
        "target_context_tokens": manifest["target_context_tokens"],
        "key_length": manifest["key_length"],
        "num_layers": layers,
        "num_attention_heads": heads,
        "shape": [int(len(unique_ids)), layers, heads],
        "aggregation": {
            "raw_logit": "mean over every prompt occurrence with the selected token id",
            "share": "sum of exact per-head softmax probability over every prompt occurrence with the selected token id",
        },
        "storage_dtype": "float16_le_base64",
        "token_ids_u32_b64": encode_u32(unique_ids),
        "token_counts_u32_b64": encode_u32(counts),
        "token_text": {
            str(int(token_id)): tokens["token_text"][str(int(token_id))]
            for token_id in unique_ids
        },
        "mean_logits_f16_b64": encode_f16(mean_logits),
        "probability_mass_f16_b64": encode_f16(probability_mass),
    }
    write_gzip_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact token-type x layer x head heatmap from 128K QK shards."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args.root, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": payload["shape"],
                "key_length": payload["key_length"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
