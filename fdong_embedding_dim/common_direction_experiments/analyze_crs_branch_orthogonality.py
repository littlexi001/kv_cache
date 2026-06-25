#!/usr/bin/env python3
"""Diagnose whether CRS branches learn on separated common/residual spaces.

This script intentionally does not save checkpoints.  It trains the same toy
single-layer residual attention model as run_crs_split_attention.py, then
measures branch-level geometry on the trained model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from run_crs_split_attention import (
    CRSLinear,
    Config,
    ToyAttention,
    build_data,
    causal_prefix_unit_mean,
    set_seed,
    split_common_residual,
    train_loss,
)


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1]).detach().float()


def pc1(x: torch.Tensor) -> torch.Tensor:
    x = flatten(x)
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[0]


def top_right(matrix: torch.Tensor) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(matrix.detach().float(), full_matrices=False)
    return vh[0]


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float(((a @ b) / denom).square())


def abs_frob_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float()
    b = b.detach().float()
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float((a * b).sum().abs() / denom)


def mean_same_token_abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = flatten(a)
    b = flatten(b)
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    mask = denom > 1e-12
    if not bool(mask.any()):
        return 0.0
    return float(((a[mask] * b[mask]).sum(dim=-1).abs() / denom[mask]).mean())


def cross_cov_norm(a: torch.Tensor, b: torch.Tensor) -> float:
    a = flatten(a)
    b = flatten(b)
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float((a.T @ b).norm() / denom)


def train_model(cfg: Config, variant: str, seed: int) -> Tuple[ToyAttention, Dict[str, object]]:
    set_seed(seed)
    data = build_data(cfg)
    if variant == "crs_alpha1":
        model_variant = "crs"
        alpha = 1.0
    elif variant == "crs_alpha05":
        model_variant = "crs"
        alpha = cfg.alpha
    else:
        raise ValueError(f"this diagnostic is for full CRS variants, got {variant}")
    model = ToyAttention(data["E0"], cfg, seed, model_variant, alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    for _ in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        loss = train_loss(model, data, reweight=False)
        loss.backward()
        opt.step()
    return model, data


def layer_inputs(model: ToyAttention, data: Dict[str, object]) -> Dict[str, torch.Tensor]:
    tokens = data["tokens"]
    x0 = model.E[tokens]
    xn = model.norm(x0)
    q = model.q(xn)
    k = model.k(xn)
    v = model.v(xn)
    scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(xn.shape[-1])
    seq_len = tokens.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask[None, :, :], -1e9)
    attn = torch.softmax(scores, dim=-1) @ v
    return {"q": xn, "k": xn, "v": xn, "o": attn}


def diagnose_one(cfg: Config, variant: str, seed: int) -> List[Dict[str, object]]:
    model, data = train_model(cfg, variant, seed)

    # One backward pass at the final model gives final-step branch gradients.
    model.zero_grad(set_to_none=True)
    loss = train_loss(model, data, reweight=False)
    loss.backward()

    inputs = layer_inputs(model, data)
    rows: List[Dict[str, object]] = []
    for name in ["q", "k", "v", "o"]:
        layer = getattr(model, name)
        assert isinstance(layer, CRSLinear)
        x = inputs[name]
        common, residual, u = split_common_residual(x)
        u_flat = flatten(u)
        nonzero = u_flat.norm(dim=-1) > 1e-8
        u_ref = u_flat[nonzero].mean(dim=0)
        u_ref = u_ref / u_ref.norm().clamp_min(1e-12)

        common_out = layer.alpha * (common @ layer.Wc.T)
        residual_out = residual @ layer.Wr.T

        eye = torch.eye(x.shape[-1])
        pc = torch.outer(u_ref, u_ref)
        pr = eye - pc
        wc_eff = layer.alpha * layer.Wc.detach().float() @ pc
        wr_eff = layer.Wr.detach().float() @ pr

        row: Dict[str, object] = {
            "variant": variant,
            "seed": seed,
            "layer": name,
            "loss": float(loss.detach()),
            "input_common_residual_same_token_abs_cos": mean_same_token_abs_cos(common, residual),
            "input_common_residual_cross_cov_norm": cross_cov_norm(common, residual),
            "output_common_residual_same_token_abs_cos": mean_same_token_abs_cos(common_out, residual_out),
            "output_common_residual_cross_cov_norm": cross_cov_norm(common_out, residual_out),
            "effective_branch_top_right_sqcos": sqcos(top_right(wc_eff), top_right(wr_eff)),
            "param_Wc_Wr_abs_frob_cos": abs_frob_cos(layer.Wc, layer.Wr),
            "grad_Wc_Wr_abs_frob_cos": abs_frob_cos(layer.Wc.grad, layer.Wr.grad),
            "grad_Wc_top_right_align_common_pc1": sqcos(top_right(layer.Wc.grad), pc1(common)),
            "grad_Wc_top_right_align_residual_pc1": sqcos(top_right(layer.Wc.grad), pc1(residual)),
            "grad_Wr_top_right_align_common_pc1": sqcos(top_right(layer.Wr.grad), pc1(common)),
            "grad_Wr_top_right_align_residual_pc1": sqcos(top_right(layer.Wr.grad), pc1(residual)),
            "common_input_energy_fraction": float((common.square().sum() / x.square().sum().clamp_min(1e-12)).detach()),
            "residual_input_energy_fraction": float((residual.square().sum() / x.square().sum().clamp_min(1e-12)).detach()),
        }
        rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {"by_variant_layer": {}, "by_variant": {}}
    numeric = [
        k
        for k, v in rows[0].items()
        if k not in {"variant", "seed", "layer"} and isinstance(v, (int, float))
    ]
    for key_fields, target in [(["variant", "layer"], "by_variant_layer"), (["variant"], "by_variant")]:
        groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
        for row in rows:
            groups.setdefault(tuple(row[k] for k in key_fields), []).append(row)
        for key, items in sorted(groups.items()):
            rec: Dict[str, object] = {"num_items": len(items)}
            for field, value in zip(key_fields, key):
                rec[field] = value
            for metric in numeric:
                vals = np.array([float(x[metric]) for x in items], dtype=np.float64)
                rec[f"{metric}_mean"] = float(vals.mean())
                rec[f"{metric}_std"] = float(vals.std(ddof=0))
            out[target]["/".join(map(str, key))] = rec
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--record_every", type=int, default=1000)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument("--init_noise", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--data_mode", choices=["cycle", "conflict"], default="cycle")
    parser.add_argument("--variants", default="crs_alpha1,crs_alpha05")
    args = parser.parse_args()
    cfg = Config(
        outdir=args.outdir,
        seeds=args.seeds,
        dim=args.dim,
        steps=args.steps,
        lr=args.lr,
        record_every=args.record_every,
        theta_deg=args.theta_deg,
        init_noise=args.init_noise,
        alpha=args.alpha,
        data_mode=args.data_mode,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for variant in [x.strip() for x in args.variants.split(",") if x.strip()]:
        for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
            print(f"diagnosing variant={variant} seed={seed}", flush=True)
            rows.extend(diagnose_one(cfg, variant, seed))
    summary = aggregate(rows)
    write_csv(outdir / "branch_orthogonality_rows.csv", rows)
    (outdir / "branch_orthogonality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "config.json").write_text(json.dumps({**asdict(cfg), "variants": args.variants}, indent=2) + "\n")
    print(json.dumps(summary["by_variant"], indent=2), flush=True)


if __name__ == "__main__":
    main()
