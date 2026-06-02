from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geometry_metrics import (  # noqa: E402
    block_geometry_metrics,
    cosine_stats,
    extract_cache_tensor,
    novelty_against_previous,
    parse_int_list,
    select_indices,
    subspace_basis,
    subspace_overlap,
    svd_stats,
    temporal_stats,
)
from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import decode_tokens, load_tokenized_text  # noqa: E402


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_forward(model, input_ids: torch.Tensor, device: torch.device):
    with torch.no_grad():
        return model(
            input_ids=input_ids[None, :].to(device),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )


def analyze_cache(
    past_key_values,
    prefix_len: int,
    layer_indices: List[int],
    head_selector: str,
    kinds: List[str],
    energy_thresholds: List[float],
    block_sizes: List[int],
    subspace_rank: int,
    prev_state: Dict[Tuple[str, int, int], Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[Tuple[str, int, int], Dict]]:
    metric_rows: List[Dict] = []
    singular_rows: List[Dict] = []
    block_rows: List[Dict] = []
    curr_state: Dict[Tuple[str, int, int], Dict] = {}

    for layer_idx in layer_indices:
        for kind in kinds:
            cache = extract_cache_tensor(past_key_values, layer_idx, kind)
            # cache shape: [kv_heads, seq_len, head_dim]
            head_indices = select_indices(cache.shape[0], head_selector)
            for head_idx in head_indices:
                x = cache[head_idx, :prefix_len, :]
                base = {
                    "prefix_len": prefix_len,
                    "layer": layer_idx,
                    "head": head_idx,
                    "kind": kind,
                }
                svd_row, singular_values, _ = svd_stats(x, energy_thresholds)
                basis = subspace_basis(x, subspace_rank)
                key = (kind, layer_idx, head_idx)
                previous = prev_state.get(key)
                previous_basis = previous["basis"] if previous is not None else None
                previous_prefix_len = previous["prefix_len"] if previous is not None else None

                row = dict(base)
                row.update(svd_row)
                row.update(cosine_stats(x))
                row.update(temporal_stats(x))
                row.update(subspace_overlap(previous_basis, basis))
                row.update(novelty_against_previous(x, previous_prefix_len, previous_basis))
                metric_rows.append(row)

                for idx, value in enumerate(singular_values.tolist()):
                    singular_rows.append(
                        {
                            **base,
                            "singular_index": idx,
                            "singular_value": value,
                        }
                    )

                for block_row in block_geometry_metrics(x, block_sizes):
                    block_rows.append({**base, **block_row})

                curr_state[key] = {
                    "basis": basis,
                    "prefix_len": prefix_len,
                }

    return metric_rows, singular_rows, block_rows, curr_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prefix-growth geometry diagnostics on Qwen KV cache.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/long_english_article_01.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prefix-lengths", default="128,256,512,1024")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--kinds", default="K,V")
    parser.add_argument("--energy-thresholds", default="0.90,0.95,0.99")
    parser.add_argument("--block-sizes", default="4,8,16,32,64")
    parser.add_argument("--subspace-rank", type=int, default=16)
    parser.add_argument("--allow-longer-than-model-max", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/prefix_geometry_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, args.max_tokens)
    max_available = int(input_ids.numel())
    model_max_position_embeddings = getattr(model.config, "max_position_embeddings", None)
    if model_max_position_embeddings is not None and max_available > int(model_max_position_embeddings):
        message = (
            f"Tokenized sequence length ({max_available}) exceeds model.config.max_position_embeddings "
            f"({model_max_position_embeddings}). This can make KV geometry unreliable due to position/RoPE handling."
        )
        if not args.allow_longer_than_model_max:
            raise ValueError(message + " Use a shorter --max-tokens or pass --allow-longer-than-model-max intentionally.")
        print(f"WARNING: {message}", flush=True)
    prefix_lengths = [p for p in parse_int_list(args.prefix_lengths) if p <= max_available]
    if not prefix_lengths:
        raise ValueError(f"No prefix lengths <= available token count ({max_available}).")

    num_layers = int(getattr(model.config, "num_hidden_layers"))
    layer_indices = select_indices(num_layers, args.layers)
    kinds = [part.strip() for part in args.kinds.split(",") if part.strip()]
    energy_thresholds = [float(x) for x in args.energy_thresholds.split(",") if x.strip()]
    block_sizes = parse_int_list(args.block_sizes)

    write_csv(output_dir / "tokens.csv", decode_tokens(tokenizer, input_ids))

    all_metric_rows: List[Dict] = []
    all_singular_rows: List[Dict] = []
    all_block_rows: List[Dict] = []
    prev_state: Dict[Tuple[str, int, int], Dict] = {}
    timings = []

    for prefix_len in prefix_lengths:
        start = time.time()
        outputs = run_forward(model, input_ids[:prefix_len], device)
        forward_s = time.time() - start
        metric_rows, singular_rows, block_rows, prev_state = analyze_cache(
            outputs.past_key_values,
            prefix_len,
            layer_indices,
            args.heads,
            kinds,
            energy_thresholds,
            block_sizes,
            args.subspace_rank,
            prev_state,
        )
        analysis_s = time.time() - start - forward_s
        all_metric_rows.extend(metric_rows)
        all_singular_rows.extend(singular_rows)
        all_block_rows.extend(block_rows)
        timings.append(
            {
                "prefix_len": prefix_len,
                "forward_seconds": forward_s,
                "analysis_seconds": analysis_s,
            }
        )
        print(
            f"prefix={prefix_len} forward={forward_s:.2f}s analysis={analysis_s:.2f}s "
            f"metric_rows={len(metric_rows)}",
            flush=True,
        )

    write_csv(output_dir / "metrics_by_prefix_layer_head.csv", all_metric_rows)
    write_csv(output_dir / "singular_values.csv", all_singular_rows)
    write_csv(output_dir / "block_metrics.csv", all_block_rows)
    write_csv(output_dir / "timings.csv", timings)

    summary = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "output_dir": str(output_dir),
        "device": str(device),
        "dtype": args.dtype,
        "max_available_tokens": max_available,
        "model_max_position_embeddings": model_max_position_embeddings,
        "seq_len_within_model_max_position_embeddings": (
            None if model_max_position_embeddings is None else max_available <= int(model_max_position_embeddings)
        ),
        "prefix_lengths": prefix_lengths,
        "layers": layer_indices,
        "heads": args.heads,
        "kinds": kinds,
        "energy_thresholds": energy_thresholds,
        "block_sizes": block_sizes,
        "subspace_rank": args.subspace_rank,
        "num_metric_rows": len(all_metric_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
