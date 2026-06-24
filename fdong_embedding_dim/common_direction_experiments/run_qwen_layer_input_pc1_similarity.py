#!/usr/bin/env python3
"""Layer-input hidden-state PC1 similarity for local Qwen3-0.6B.

This diagnostic captures each transformer block's input hidden states on a
natural-language local markdown corpus, computes the centered top right
singular direction for each layer-input activation matrix, and reports:

  - adjacent-layer absolute cosine similarity;
  - similarity to layer 0 PC1;
  - top-1 energy of each layer-input activation matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_qwen_activation_parameter_alignment import (
    collect_natural_markdown_text,
    make_chunks,
    normalized,
    top_right_power,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="fdong/Qwen3-0.6B")
    parser.add_argument(
        "--output_dir",
        default="fdong_embedding_dim/outputs/qwen_layer_input_pc1_similarity",
    )
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_chunks", type=int, default=8)
    parser.add_argument("--max_chars", type=int, default=220_000)
    parser.add_argument("--power_iters", type=int, default=40)
    return parser.parse_args()


def abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((normalized(a.float()) @ normalized(b.float())).abs().item())


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    text, used_files = collect_natural_markdown_text(args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    input_ids = make_chunks(tokenizer, text, args.seq_len, args.max_chunks)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()
    num_layers = len(model.model.layers)

    raw_captures: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}
    normed_captures: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_layers)}
    handles = []
    for layer_id, layer in enumerate(model.model.layers):
        def hook(mod, inputs, output, layer_id=layer_id):
            x = inputs[0].detach().cpu().float().reshape(-1, inputs[0].shape[-1])
            raw_captures[layer_id].append(x)

        handles.append(layer.register_forward_hook(hook))

        def norm_hook(mod, inputs, output, layer_id=layer_id):
            y = output.detach().cpu().float().reshape(-1, output.shape[-1])
            normed_captures[layer_id].append(y)

        handles.append(layer.input_layernorm.register_forward_hook(norm_hook))

    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            _ = model(input_ids=input_ids[i : i + 1])

    for h in handles:
        h.remove()

    def build_rows(captures: Dict[int, List[torch.Tensor]], representation: str) -> List[Dict[str, float | int | str]]:
        pcs: Dict[int, torch.Tensor] = {}
        rows: List[Dict[str, float | int | str]] = []
        for layer_id in range(num_layers):
            x = torch.cat(captures[layer_id], dim=0)
            pc1, sigma1, top1_energy = top_right_power(x, args.power_iters, center=True)
            pcs[layer_id] = pc1
            rows.append(
                {
                    "representation": representation,
                    "layer": layer_id,
                    "num_rows": int(x.shape[0]),
                    "hidden_dim": int(x.shape[1]),
                    "top1_energy": top1_energy,
                    "sigma1": sigma1,
                    "abs_cos_to_layer0_pc1": abs_cos(pc1, pcs[0]) if layer_id > 0 else 1.0,
                    "sqcos_to_layer0_pc1": abs_cos(pc1, pcs[0]) ** 2 if layer_id > 0 else 1.0,
                    "abs_cos_to_prev_layer_pc1": abs_cos(pc1, pcs[layer_id - 1]) if layer_id > 0 else 1.0,
                    "sqcos_to_prev_layer_pc1": abs_cos(pc1, pcs[layer_id - 1]) ** 2 if layer_id > 0 else 1.0,
                }
            )
        return rows

    rows = build_rows(raw_captures, "raw_layer_input") + build_rows(normed_captures, "attention_input_after_input_layernorm")

    csv_path = outdir / "qwen_layer_input_pc1_similarity.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    turns_by_representation = {
        rep: sorted([r for r in rows if r["representation"] == rep and int(r["layer"]) > 0], key=lambda r: float(r["abs_cos_to_prev_layer_pc1"]))[:8]
        for rep in sorted(set(str(r["representation"]) for r in rows))
    }
    summary = {
        "model_dir": args.model_dir,
        "seq_len": args.seq_len,
        "num_chunks": int(input_ids.shape[0]),
        "num_tokens": int(input_ids.numel()),
        "num_layers": num_layers,
        "text_source": "local markdown natural-language corpus with fenced code blocks removed",
        "num_source_files": len(used_files),
        "source_files_preview": used_files[:20],
        "csv_path": str(csv_path),
        "lowest_adjacent_similarity_layers_by_representation": turns_by_representation,
        "rows": rows,
    }
    summary_path = outdir / "qwen_layer_input_pc1_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
