#!/usr/bin/env python3
"""Activation-to-parameter alignment in local Qwen3-0.6B on natural text.

This diagnostic answers:

    For a real language input stream, do a layer's parameter matrices align
    with the top singular direction of the hidden states that they actually
    receive as input?

We collect forward-hook inputs to selected Linear modules:

    self_attn.q_proj, k_proj, v_proj, o_proj
    mlp.gate_proj, up_proj, down_proj

For each module, we compare:

    top right singular vector of centered activation input X
    vs
    top input-side singular vector of weight W

For weight W with shape [out_dim, in_dim], the input-side singular vector is
the top right singular vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODULE_SUFFIXES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="fdong/Qwen3-0.6B")
    parser.add_argument(
        "--output_dir",
        default="fdong_embedding_dim/outputs/qwen_activation_parameter_alignment",
    )
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_chunks", type=int, default=8)
    parser.add_argument("--max_chars", type=int, default=220_000)
    parser.add_argument("--power_iters", type=int, default=40)
    return parser.parse_args()


def parse_layers(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def strip_markdown_noise(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("|") or s.startswith("!") or s.startswith("::"):
            continue
        if len(re.sub(r"[\W_]+", "", s)) < max(8, len(s) * 0.25):
            continue
        lines.append(s)
    return "\n".join(lines)


def collect_natural_markdown_text(max_chars: int) -> Tuple[str, List[str]]:
    roots = [
        Path("common_doc"),
        Path("main_specialization"),
        Path("main_inverse_kv"),
        Path("references"),
        Path("fdong"),
        Path("fdong_embedding_dim"),
        Path("fdong_seq_compress"),
    ]
    pieces: List[str] = []
    used: List[str] = []
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            if "outputs" in path.parts or "curated_results" in path.parts:
                continue
            try:
                raw = path.read_text(errors="ignore")
            except Exception:
                continue
            text = strip_markdown_noise(raw)
            if len(text) < 200:
                continue
            take = text[: max(0, max_chars - total)]
            if not take:
                break
            pieces.append(take)
            used.append(str(path))
            total += len(take)
            if total >= max_chars:
                break
        if total >= max_chars:
            break
    if not pieces:
        raise RuntimeError("No markdown natural-language text was found.")
    return "\n\n".join(pieces), used


def make_chunks(tokenizer, text: str, seq_len: int, max_chunks: int) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    needed = seq_len * max_chunks
    ids = ids[:needed]
    n = ids.numel() // seq_len
    if n == 0:
        raise RuntimeError(f"Not enough tokens for one chunk of length {seq_len}.")
    ids = ids[: n * seq_len].reshape(n, seq_len)
    return ids


def normalized(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / v.norm().clamp_min(eps)


def top_right_power(matrix: torch.Tensor, iters: int, center: bool) -> Tuple[torch.Tensor, float, float]:
    x = matrix.detach().float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    in_dim = x.shape[1]
    gen = torch.Generator().manual_seed(1234 + in_dim + x.shape[0])
    v = normalized(torch.randn(in_dim, generator=gen))
    for _ in range(iters):
        u = normalized(x @ v)
        v = normalized(x.T @ u)
    sigma = float((x @ v).norm().item())
    total = float(x.square().sum().item())
    energy = (sigma * sigma / total) if total > 1e-12 else 0.0
    return v, sigma, energy


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((normalized(a.float()) @ normalized(b.float())).square().item())


def get_module(model, name: str):
    cur = model
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    input_captures: Dict[str, List[torch.Tensor]] = {}
    output_captures: Dict[str, List[torch.Tensor]] = {}
    handles = []
    module_names: List[str] = []
    for layer in layers:
        for suffix in MODULE_SUFFIXES:
            name = f"model.layers.{layer}.{suffix}"
            module_names.append(name)
            input_captures[name] = []
            output_captures[name] = []
            module = get_module(model, name)

            def hook(mod, inputs, output, name=name):
                x = inputs[0].detach().cpu().float().reshape(-1, inputs[0].shape[-1])
                y = output.detach().cpu().float().reshape(-1, output.shape[-1])
                input_captures[name].append(x)
                output_captures[name].append(y)

            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        for i in range(input_ids.shape[0]):
            _ = model(input_ids=input_ids[i : i + 1])

    for h in handles:
        h.remove()

    rows = []
    for name in module_names:
        layer = int(name.split(".")[2])
        module_suffix = ".".join(name.split(".")[3:])
        module = get_module(model, name)
        weight = module.weight.detach().cpu().float()
        input_act = torch.cat(input_captures[name], dim=0)
        output_act = torch.cat(output_captures[name], dim=0)
        input_act_v, input_act_sigma, input_act_top1_energy = top_right_power(
            input_act, args.power_iters, center=True
        )
        output_act_v, output_act_sigma, output_act_top1_energy = top_right_power(
            output_act, args.power_iters, center=True
        )
        weight_input_v, weight_sigma, weight_input_top1_energy = top_right_power(
            weight, args.power_iters, center=False
        )
        weight_output_v, _, weight_output_top1_energy = top_right_power(
            weight.T, args.power_iters, center=False
        )
        input_dim = input_act.shape[1]
        output_dim = output_act.shape[1]
        input_align = sqcos(input_act_v, weight_input_v)
        output_align = sqcos(output_act_v, weight_output_v)
        rows.append(
            {
                "layer": layer,
                "module": module_suffix,
                "num_activation_rows": int(input_act.shape[0]),
                "input_dim": int(input_dim),
                "output_dim": int(output_dim),
                "input_random_sqcos_expectation": 1.0 / float(input_dim),
                "output_random_sqcos_expectation": 1.0 / float(output_dim),
                "input_alignment_sqcos": input_align,
                "input_alignment_over_random": input_align / (1.0 / float(input_dim)),
                "output_alignment_sqcos": output_align,
                "output_alignment_over_random": output_align / (1.0 / float(output_dim)),
                "input_activation_top1_energy": input_act_top1_energy,
                "input_activation_sigma1": input_act_sigma,
                "output_activation_top1_energy": output_act_top1_energy,
                "output_activation_sigma1": output_act_sigma,
                "weight_input_top1_energy": weight_input_top1_energy,
                "weight_output_top1_energy": weight_output_top1_energy,
                "weight_sigma1": weight_sigma,
            }
        )

    csv_path = output_dir / "qwen_activation_parameter_alignment.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_dir": args.model_dir,
        "layers": layers,
        "seq_len": args.seq_len,
        "num_chunks": int(input_ids.shape[0]),
        "num_tokens": int(input_ids.numel()),
        "text_source": "local markdown natural-language corpus with fenced code blocks removed",
        "num_source_files": len(used_files),
        "source_files_preview": used_files[:20],
        "csv_path": str(csv_path),
        "top_input_alignment_rows": sorted(rows, key=lambda r: float(r["input_alignment_sqcos"]), reverse=True)[:15],
        "top_output_alignment_rows": sorted(rows, key=lambda r: float(r["output_alignment_sqcos"]), reverse=True)[:15],
        "rows": rows,
    }
    summary_path = output_dir / "qwen_activation_parameter_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
