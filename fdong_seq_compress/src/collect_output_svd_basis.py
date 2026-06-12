from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch

from model_loader import load_model_and_tokenizer


DEFAULT_GLOBS = (
    "fdong_seq_compress/data/synthetic_texts/*.txt",
)


def discover_texts(patterns: str) -> List[Path]:
    paths: List[Path] = []
    for pattern in [item.strip() for item in patterns.split(",") if item.strip()]:
        paths.extend(Path(".").glob(pattern))
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if not unique:
        raise ValueError(f"No text files matched: {patterns}")
    return unique


def sample_positions(length: int, count: int, seed: int) -> torch.Tensor:
    candidates = torch.arange(1, length, dtype=torch.long)
    if candidates.numel() <= count:
        return candidates
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return candidates[torch.randperm(candidates.numel(), generator=generator)[:count]].sort().values


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final hidden states and cache an uncentered SVD basis.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-globs", default=",".join(DEFAULT_GLOBS))
    parser.add_argument("--output-dir", default="fdong_seq_compress/artifacts/output_svd_qwen3_0p6b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--chunk-stride", type=int, default=768)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--samples-per-chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_paths = discover_texts(args.text_globs)
    chunks_per_file = max(1, math.ceil(args.max_chunks / len(text_paths)))
    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path, device=args.device, dtype=args.dtype, attn_implementation="eager"
    )

    states: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    rows: List[Dict] = []
    chunk_id = 0
    started = time.perf_counter()
    for file_idx, path in enumerate(text_paths):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        starts = list(range(0, max(1, ids.numel() - 1), args.chunk_stride))
        for start in starts[:chunks_per_file]:
            if chunk_id >= args.max_chunks:
                break
            chunk = ids[start : start + args.chunk_tokens]
            if chunk.numel() < 32:
                continue
            positions = sample_positions(
                int(chunk.numel()), args.samples_per_chunk, args.seed + file_idx * 10000 + start
            )
            with torch.no_grad():
                hidden = model.model(input_ids=chunk[None, :].to(device), use_cache=False).last_hidden_state[0]
            prediction_positions = positions - 1
            selected = hidden[prediction_positions.to(device)].detach().cpu().float()
            target = chunk[positions].cpu().long()
            states.append(selected)
            labels.append(target)
            for local_idx, position in enumerate(positions.tolist()):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "file": str(path),
                        "chunk_id": chunk_id,
                        "chunk_start": start,
                        "prediction_position_in_chunk": position - 1,
                        "target_position_in_chunk": position,
                        "absolute_prediction_position": start + position - 1,
                        "absolute_target_position": start + position,
                        "target_token_id": int(target[local_idx]),
                    }
                )
            chunk_id += 1
            print(
                f"chunk={chunk_id}/{args.max_chunks} file={path.name} tokens={chunk.numel()} "
                f"samples={len(rows)} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
        if chunk_id >= args.max_chunks:
            break

    if not states:
        raise RuntimeError("No hidden states were collected.")
    x = torch.cat(states, dim=0).contiguous()
    y = torch.cat(labels, dim=0).contiguous()
    # For X = U S V^T, eigenvectors of X^T X are V and eigenvalues are S^2.
    gram = x.T @ x
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    singular_values = eigenvalues[order].clamp_min(0).sqrt()
    basis = eigenvectors[:, order].contiguous()
    energy = singular_values.square()
    explained = energy / energy.sum().clamp_min(1e-12)

    torch.save(
        {"hidden_states": x, "target_ids": y, "rows": rows},
        output_dir / "sampled_final_hidden.pt",
    )
    torch.save(
        {
            "basis": basis,
            "singular_values": singular_values,
            "explained_energy": explained,
            "mean": x.mean(dim=0),
            "uncentered": True,
        },
        output_dir / "uncentered_svd_basis.pt",
    )
    cumulative = explained.cumsum(dim=0)
    with (output_dir / "singular_value_energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["direction_index", "singular_value", "explained_energy", "cumulative_energy"],
        )
        writer.writeheader()
        for idx in range(singular_values.numel()):
            writer.writerow(
                {
                    "direction_index": idx,
                    "singular_value": float(singular_values[idx].item()),
                    "explained_energy": float(explained[idx].item()),
                    "cumulative_energy": float(cumulative[idx].item()),
                }
            )
    metadata = {
        "model_path": args.model_path,
        "device": str(device),
        "hidden_size": int(x.shape[1]),
        "sample_count": int(x.shape[0]),
        "rank_upper_bound": int(min(x.shape)),
        "chunk_count": chunk_id,
        "text_file_count": len({row["file"] for row in rows}),
        "text_globs": args.text_globs,
        "chunk_tokens": args.chunk_tokens,
        "chunk_stride": args.chunk_stride,
        "samples_per_chunk": args.samples_per_chunk,
        "elapsed_seconds": time.perf_counter() - started,
        "basis_definition": "Uncentered right singular vectors of final-norm hidden states.",
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote reusable SVD artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
