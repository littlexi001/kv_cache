from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


LENGTH_PATTERN = re.compile(r"length_(\d+)\.json\.gz$")


def decode_array(value: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    array = np.frombuffer(base64.b64decode(value), dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(f"decoded {array.size} values, expected {expected} for {shape}")
    return array.reshape(shape)


def encode_array(array: np.ndarray, dtype: np.dtype[Any]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return base64.b64encode(contiguous.tobytes()).decode("ascii")


def canonical_token(text: str) -> str:
    return text.strip().lower()


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def discover_inputs(input_dir: Path) -> list[tuple[int, Path]]:
    discovered: list[tuple[int, Path]] = []
    for path in input_dir.glob("length_*.json.gz"):
        match = LENGTH_PATTERN.search(path.name)
        if match:
            discovered.append((int(match.group(1)), path))
    discovered.sort()
    if not discovered:
        raise FileNotFoundError(f"no length_*.json.gz inputs in {input_dir}")
    return discovered


def build_index(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    inputs = discover_inputs(input_dir)
    first = read_gzip_json(inputs[0][1])
    num_layers = int(first["num_layers"])
    num_heads = int(first["num_attention_heads"])
    lengths = [length for length, _ in inputs]

    token_metadata: dict[str, dict[str, Any]] = {}
    for length, path in inputs:
        payload = read_gzip_json(path)
        if int(payload["num_layers"]) != num_layers:
            raise ValueError(f"layer mismatch in {path}")
        if int(payload["num_attention_heads"]) != num_heads:
            raise ValueError(f"head mismatch in {path}")
        token_ids = decode_array(
            payload["token_ids_u32_b64"], np.dtype("<u4"), (int(payload["shape"][0]),)
        )
        for token_id in token_ids:
            raw_text = str(payload["token_text"][str(int(token_id))])
            canonical = canonical_token(raw_text)
            if not canonical:
                continue
            metadata = token_metadata.setdefault(
                canonical,
                {"display": raw_text.strip() or raw_text, "token_ids": set()},
            )
            metadata["token_ids"].add(int(token_id))

    token_names = sorted(token_metadata)
    token_to_index = {token: index for index, token in enumerate(token_names)}
    shape = (len(token_names), len(inputs), num_layers, num_heads)
    mean_logits = np.full(shape, np.nan, dtype=np.float16)
    probability_mass = np.zeros(shape, dtype=np.float32)
    occurrence_counts = np.zeros((len(token_names), len(inputs)), dtype=np.uint32)
    key_lengths = np.zeros(len(inputs), dtype=np.uint32)

    for length_index, (length, path) in enumerate(inputs):
        payload = read_gzip_json(path)
        source_shape = tuple(int(value) for value in payload["shape"])
        if source_shape[1:] != (num_layers, num_heads):
            raise ValueError(f"unexpected matrix shape {source_shape} in {path}")
        token_ids = decode_array(payload["token_ids_u32_b64"], np.dtype("<u4"), (source_shape[0],))
        token_counts = decode_array(
            payload["token_counts_u32_b64"], np.dtype("<u4"), (source_shape[0],)
        )
        source_logits = decode_array(
            payload["mean_logits_f16_b64"], np.dtype("<f2"), source_shape
        ).astype(np.float32)
        source_mass = decode_array(
            payload["probability_mass_f32_b64"], np.dtype("<f4"), source_shape
        )
        key_lengths[length_index] = int(payload["key_length"])

        grouped: dict[int, list[int]] = {}
        for source_index, token_id in enumerate(token_ids):
            raw_text = str(payload["token_text"][str(int(token_id))])
            canonical = canonical_token(raw_text)
            if canonical:
                grouped.setdefault(token_to_index[canonical], []).append(source_index)

        for target_index, source_indices in grouped.items():
            indices = np.asarray(source_indices, dtype=np.int64)
            counts = token_counts[indices].astype(np.float32)
            total_count = int(counts.sum())
            weighted_logits = (
                source_logits[indices] * counts[:, None, None]
            ).sum(axis=0) / float(total_count)
            mean_logits[target_index, length_index] = weighted_logits.astype(np.float16)
            probability_mass[target_index, length_index] = source_mass[indices].sum(axis=0)
            occurrence_counts[target_index, length_index] = total_count

        print(
            f"[{length_index + 1}/{len(inputs)}] length={length} "
            f"tokens={source_shape[0]} key_length={int(key_lengths[length_index])}",
            flush=True,
        )

    token_manifest: dict[str, Any] = {}
    tokens_dir = output_dir / "tokens"
    for token_index, token in enumerate(token_names):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:20]
        relative_path = f"tokens/{digest}.json.gz"
        counts = occurrence_counts[token_index]
        present = np.flatnonzero(counts)
        metadata = token_metadata[token]
        write_gzip_json(
            output_dir / relative_path,
            {
                "schema_version": 1,
                "token": token,
                "display": metadata["display"],
                "token_ids": sorted(metadata["token_ids"]),
                "lengths": lengths,
                "shape": [len(lengths), num_layers, num_heads],
                "occurrence_counts_u32_b64": encode_array(counts, np.dtype("<u4")),
                "mean_logits_f16_b64": encode_array(
                    mean_logits[token_index], np.dtype("<f2")
                ),
                "probability_mass_f32_b64": encode_array(
                    probability_mass[token_index], np.dtype("<f4")
                ),
            },
        )
        token_manifest[token] = {
            "display": metadata["display"],
            "file": relative_path,
            "token_ids": sorted(metadata["token_ids"]),
            "present_length_count": int(present.size),
            "first_length": int(lengths[int(present[0])]) if present.size else None,
            "last_length": int(lengths[int(present[-1])]) if present.size else None,
        }

    manifest = {
        "schema_version": 1,
        "model": "Qwen3-8B",
        "experiment": "english_single_token_clean_two_hop_full2",
        "normalization": "trim whitespace, then lowercase",
        "raw_logit_definition": (
            "mean QK/sqrt(d) over all positions whose tokenizer token normalizes to the selected token"
        ),
        "share_definition": (
            "sum of exact post-softmax attention probability over all positions whose tokenizer token "
            "normalizes to the selected token"
        ),
        "lengths": lengths,
        "key_lengths": [int(value) for value in key_lengths],
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "shape_per_token": [len(lengths), num_layers, num_heads],
        "token_count": len(token_names),
        "tokens": token_manifest,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "complete.txt").write_text("ok\n", encoding="utf-8")
    print(f"wrote {len(token_names)} token series to {tokens_dir}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invert per-length token-type attention exports into browser-lazy token series."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_length_count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.expected_length_count:
        inputs = discover_inputs(input_dir)
        if len(inputs) != args.expected_length_count:
            raise RuntimeError(
                f"expected {args.expected_length_count} length files, found {len(inputs)}"
            )
    build_index(input_dir, output_dir)


if __name__ == "__main__":
    main()
