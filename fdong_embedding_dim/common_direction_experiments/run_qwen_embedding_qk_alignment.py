#!/usr/bin/env python3
"""Inspect embedding-to-attention input-direction alignment in local Qwen weights.

This is a read-only diagnostic for the local official Qwen3-0.6B checkpoint.
It measures whether the input-side singular directions of Q/K/V projection
weights align with:

1. the uncentered embedding mean direction;
2. the top centered embedding PCA direction;
3. the top-k centered embedding PCA subspace.

The key metric is squared cosine. For a random direction in hidden dimension d,
the expected top-1 squared cosine is about 1/d.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors.torch import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="fdong/Qwen3-0.6B")
    parser.add_argument(
        "--output_dir",
        default="fdong_embedding_dim/outputs/qwen_embedding_qk_alignment",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--layers", default="all")
    return parser.parse_args()


def requested_layers(spec: str, num_layers: int) -> List[int]:
    if spec == "all":
        return list(range(num_layers))
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(x for x in out if 0 <= x < num_layers))


def top_eigenvectors_symmetric(mat: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return descending top-k eigenvalues and eigenvectors for a symmetric matrix."""
    eigvals, eigvecs = torch.linalg.eigh(mat)
    vals = eigvals[-k:].flip(0).contiguous()
    vecs = eigvecs[:, -k:].flip(1).contiguous()
    return vals, vecs


def normalized(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vec / vec.norm().clamp_min(eps)


def squared_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = normalized(a.float())
    b = normalized(b.float())
    return float((a @ b).pow(2).item())


def subspace_mass(direction: torch.Tensor, basis: torch.Tensor) -> float:
    direction = normalized(direction.float())
    basis = basis.float()
    return float(torch.sum((basis.T @ direction).pow(2)).item())


def mean_pairwise_subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Average squared cosine between two same-rank orthonormal bases."""
    rank = min(a.shape[1], b.shape[1])
    m = a[:, :rank].T.float() @ b[:, :rank].float()
    return float((m.pow(2).sum() / rank).item())


def projection_input_basis(weight: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Input-side right singular vectors of a linear weight [out_dim, in_dim]."""
    w = weight.float()
    gram = w.T @ w
    vals, vecs = top_eigenvectors_symmetric(gram, topk)
    total = float(torch.trace(gram).item())
    top1_energy = float(vals[0].item() / total) if total > 0 else 0.0
    return vals, vecs, top1_energy


def qk_power_top_singular(
    q_head_weight: torch.Tensor,
    k_head_weight: torch.Tensor,
    init: torch.Tensor,
    num_iters: int = 40,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Top singular vectors of B = Wq_h.T @ Wk_h without forming B.

    Returns query-input direction u, key-input direction v, and singular value.
    The score is x_q.T @ B @ x_k, so u lives in query input space and v lives
    in key input space. In this model both spaces are the residual hidden size.
    """
    q = q_head_weight.float()
    k = k_head_weight.float()
    v = normalized(init.float())
    u = normalized(q.T @ (k @ v))
    for _ in range(num_iters):
        v = normalized(k.T @ (q @ u))
        u = normalized(q.T @ (k @ v))
    bv = q.T @ (k @ v)
    sigma = float(bv.norm().item())
    return u, v, sigma


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_hidden_layers"])
    layers = requested_layers(args.layers, num_layers)
    topk = min(args.topk, hidden_size)

    rows: List[Dict[str, float | int | str]] = []
    model_path = model_dir / "model.safetensors"

    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        embed = f.get_tensor("model.embed_tokens.weight").float()
        lm_head = f.get_tensor("lm_head.weight").float()

        embedding_mean = normalized(embed.mean(dim=0))
        centered = embed - embed.mean(dim=0, keepdim=True)
        cov = centered.T @ centered
        embed_vals, embed_basis = top_eigenvectors_symmetric(cov, topk)
        embed_pc1 = embed_basis[:, 0]
        embed_total = float(torch.trace(cov).item())
        embed_top_energy = [float(v.item() / embed_total) for v in embed_vals]
        tied_max_abs_diff = float((embed - lm_head).abs().max().item())

        for layer in layers:
            for proj in ["q_proj", "k_proj", "v_proj"]:
                key = f"model.layers.{layer}.self_attn.{proj}.weight"
                weight = f.get_tensor(key)
                vals, basis, top1_energy = projection_input_basis(weight, topk)
                total = float(torch.sum(weight.float().pow(2)).item())
                rows.append(
                    {
                        "layer": layer,
                        "projection": proj,
                        "weight_shape": "x".join(map(str, weight.shape)),
                        "input_top1_energy": top1_energy,
                        "align_mean_to_input_top1_sqcos": squared_cosine(embedding_mean, basis[:, 0]),
                        "align_centered_pc1_to_input_top1_sqcos": squared_cosine(embed_pc1, basis[:, 0]),
                        f"mean_mass_in_input_top{topk}": subspace_mass(embedding_mean, basis),
                        f"centered_pc1_mass_in_input_top{topk}": subspace_mass(embed_pc1, basis),
                        f"embed_top{topk}_to_input_top{topk}_mean_overlap": mean_pairwise_subspace_overlap(
                            embed_basis, basis
                        ),
                        "projection_fro_norm": total**0.5,
                        "top1_singular_value": float(vals[0].sqrt().item()),
                    }
                )

        qk_rows: List[Dict[str, float | int | str]] = []
        num_attention_heads = int(config["num_attention_heads"])
        num_key_value_heads = int(config["num_key_value_heads"])
        head_dim = hidden_size // num_attention_heads
        kv_group = num_attention_heads // num_key_value_heads
        for layer in layers:
            q_weight = f.get_tensor(f"model.layers.{layer}.self_attn.q_proj.weight").float()
            k_weight = f.get_tensor(f"model.layers.{layer}.self_attn.k_proj.weight").float()
            for q_head in range(num_attention_heads):
                kv_head = q_head // kv_group
                qh = q_weight[q_head * head_dim : (q_head + 1) * head_dim, :]
                kh = k_weight[kv_head * head_dim : (kv_head + 1) * head_dim, :]
                uq, vk, sigma = qk_power_top_singular(qh, kh, embed_pc1)
                qk_rows.append(
                    {
                        "layer": layer,
                        "q_head": q_head,
                        "kv_head": kv_head,
                        "top_singular_value": sigma,
                        "query_input_align_mean_sqcos": squared_cosine(embedding_mean, uq),
                        "key_input_align_mean_sqcos": squared_cosine(embedding_mean, vk),
                        "query_input_align_centered_pc1_sqcos": squared_cosine(embed_pc1, uq),
                        "key_input_align_centered_pc1_sqcos": squared_cosine(embed_pc1, vk),
                    }
                )

    csv_path = output_dir / "qwen_embedding_attention_input_alignment.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    qk_csv_path = output_dir / "qwen_embedding_qk_bilinear_head_alignment.csv"
    with open(qk_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qk_rows)

    summary = {
        "model_dir": str(model_dir),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "layers": layers,
        "topk": topk,
        "random_top1_sqcos_expectation": 1.0 / hidden_size,
        "random_topk_mass_expectation": topk / hidden_size,
        "embedding_top_energy": embed_top_energy,
        "embedding_lm_head_max_abs_diff": tied_max_abs_diff,
        "csv_path": str(csv_path),
        "qk_csv_path": str(qk_csv_path),
        "layer0_1_rows": [r for r in rows if int(r["layer"]) in (0, 1)],
        "qk_layer0_1_max_rows": [
            max(
                (r for r in qk_rows if int(r["layer"]) == layer),
                key=lambda r: max(
                    float(r["query_input_align_centered_pc1_sqcos"]),
                    float(r["key_input_align_centered_pc1_sqcos"]),
                ),
            )
            for layer in [0, 1]
            if any(int(r["layer"]) == layer for r in qk_rows)
        ],
        "max_centered_pc1_alignment_by_projection": {
            proj: max(
                (float(r["align_centered_pc1_to_input_top1_sqcos"]) for r in rows if r["projection"] == proj),
                default=0.0,
            )
            for proj in ["q_proj", "k_proj", "v_proj"]
        },
        "mean_centered_pc1_alignment_by_projection": {
            proj: sum(
                float(r["align_centered_pc1_to_input_top1_sqcos"]) for r in rows if r["projection"] == proj
            )
            / max(1, sum(1 for r in rows if r["projection"] == proj))
            for proj in ["q_proj", "k_proj", "v_proj"]
        },
        "qk_max_centered_pc1_alignment": {
            "query_input": max(float(r["query_input_align_centered_pc1_sqcos"]) for r in qk_rows),
            "key_input": max(float(r["key_input_align_centered_pc1_sqcos"]) for r in qk_rows),
        },
        "qk_mean_centered_pc1_alignment": {
            "query_input": sum(float(r["query_input_align_centered_pc1_sqcos"]) for r in qk_rows)
            / len(qk_rows),
            "key_input": sum(float(r["key_input_align_centered_pc1_sqcos"]) for r in qk_rows)
            / len(qk_rows),
        },
    }
    summary_path = output_dir / "qwen_embedding_attention_input_alignment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
