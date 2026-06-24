#!/usr/bin/env python3
"""Stage 8: Does normed-skip PreNorm reduce parameter singularization?

Compare standard PreNorm

    x <- x + F(RMSNorm(x))

against the aggressive normalized-skip variant proposed in the discussion:

    x <- RMSNorm(x) + F(RMSNorm(x))

on the existing low-dimensional shared-K Zipf synthetic language task.

The diagnostic records:
  - raw layer-input representation top-1 spectral energy;
  - parameter top-1 spectral energy for Q/K/V/O/MLP weights;
  - input-side parameter direction alignment to raw/normed activation PC1;
  - output-side parameter direction alignment to module output activation PC1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Config:
    outdir: str
    seeds: str = "0,1,2,3,4"
    dim: int = 4
    num_layers: int = 6
    steps: int = 1600
    lr: float = 0.02
    record_every: int = 40
    theta_deg: float = 12.0
    init_noise: float = 0.01
    mlp_mult: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rot(v: np.ndarray, degrees: float, dim: int) -> np.ndarray:
    center = np.array(v, dtype=np.float32)
    center /= np.linalg.norm(center)
    perp = np.zeros(dim, dtype=np.float32)
    perp[0 if abs(center[0]) < 0.9 else 1] = 1.0
    perp -= float(np.dot(perp, center)) * center
    perp /= np.linalg.norm(perp)
    theta = math.radians(degrees)
    return (math.cos(theta) * center + math.sin(theta) * perp).astype(np.float32)


def build_data(cfg: Config) -> Dict[str, object]:
    centers = {
        "A": np.eye(cfg.dim, dtype=np.float32)[0],
        "B": np.eye(cfg.dim, dtype=np.float32)[1],
        "C": np.eye(cfg.dim, dtype=np.float32)[2],
        "D": np.eye(cfg.dim, dtype=np.float32)[3],
    }
    e_rows: List[np.ndarray] = [np.ones(cfg.dim, dtype=np.float32) / math.sqrt(cfg.dim)]
    token_groups: List[str] = ["K"]
    group_ids: Dict[str, List[int]] = {}
    for group, center in centers.items():
        ids = []
        for off in [0.0, cfg.theta_deg, -cfg.theta_deg]:
            ids.append(len(e_rows))
            e_rows.append(rot(center, off, cfg.dim))
            token_groups.append(group)
        group_ids[group] = ids

    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    groups: List[str] = []
    families: List[str] = []
    for group in ["A", "B", "C", "D"]:
        i0, i1, i2 = group_ids[group]
        k = 0
        patterns = [
            (i0, i1, k, "to_K"),
            (i1, k, i2, "from_K_1"),
            (k, i2, i0, "from_K_2"),
            (i2, i0, i1, "internal"),
        ]
        for a, b, y, family in patterns:
            c1.append(a)
            c2.append(b)
            targets.append(y)
            groups.append(group)
            families.append(family)

    probs = {"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10}
    weights = torch.tensor([probs[g] / sum(x == g for x in groups) for g in groups], dtype=torch.float32)
    weights = weights / weights.sum()
    return {
        "E0": torch.tensor(np.stack(e_rows), dtype=torch.float32),
        "tokens": torch.tensor(np.stack([c1, c2], axis=1), dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "weights": weights,
        "groups": groups,
        "families": families,
        "token_groups": token_groups,
    }


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return x / rms * self.weight


class ToyBlock(torch.nn.Module):
    def __init__(self, dim: int, mlp_mult: int, seed: int, init_noise: float, mode: str):
        super().__init__()
        self.mode = mode
        gen = torch.Generator().manual_seed(seed)
        eye = torch.eye(dim, dtype=torch.float32)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.Wq = torch.nn.Parameter(eye * 0.1 + init_noise * torch.randn(dim, dim, generator=gen))
        self.Wk = torch.nn.Parameter(eye * 0.1 + init_noise * torch.randn(dim, dim, generator=gen))
        self.Wv = torch.nn.Parameter(eye * 0.1 + init_noise * torch.randn(dim, dim, generator=gen))
        self.Wo = torch.nn.Parameter(eye * 0.1 + init_noise * torch.randn(dim, dim, generator=gen))
        hidden = dim * mlp_mult
        self.Wup = torch.nn.Parameter(init_noise * torch.randn(hidden, dim, generator=gen))
        self.Wdown = torch.nn.Parameter(init_noise * torch.randn(dim, hidden, generator=gen))
        self.scale = math.sqrt(dim)

    def skip(self, x: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
        if self.mode == "prenorm":
            return x
        if self.mode == "normed_skip":
            return xn
        raise ValueError(self.mode)

    def forward(self, x: torch.Tensor, cache: Dict[str, List[torch.Tensor]], layer: int) -> torch.Tensor:
        cache[f"layer{layer}.raw_input"].append(x.detach())
        xn = self.norm1(x)
        cache[f"layer{layer}.attn_norm_input"].append(xn.detach())
        q = xn @ self.Wq.T
        k = xn @ self.Wk.T
        v = xn @ self.Wv.T
        cache[f"layer{layer}.q_out"].append(q.detach())
        cache[f"layer{layer}.k_out"].append(k.detach())
        cache[f"layer{layer}.v_out"].append(v.detach())
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None, :, :], -1e9)
        attn = torch.softmax(scores, dim=-1) @ v
        attn_out = attn @ self.Wo.T
        cache[f"layer{layer}.o_input"].append(attn.detach())
        cache[f"layer{layer}.o_out"].append(attn_out.detach())
        x = self.skip(x, xn) + attn_out

        xn2 = self.norm2(x)
        cache[f"layer{layer}.mlp_norm_input"].append(xn2.detach())
        up = F.gelu(xn2 @ self.Wup.T)
        down = up @ self.Wdown.T
        cache[f"layer{layer}.up_out"].append(up.detach())
        cache[f"layer{layer}.down_out"].append(down.detach())
        x = self.skip(x, xn2) + down
        return x


class ToyTransformer(torch.nn.Module):
    def __init__(self, e0: torch.Tensor, cfg: Config, seed: int, mode: str):
        super().__init__()
        self.E = torch.nn.Parameter(e0.clone())
        self.blocks = torch.nn.ModuleList(
            [ToyBlock(cfg.dim, cfg.mlp_mult, seed + 1009 * (i + 1), cfg.init_noise, mode) for i in range(cfg.num_layers)]
        )

    def forward(self, tokens: torch.Tensor, return_cache: bool = False) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        x = self.E[tokens]
        cache: Dict[str, List[torch.Tensor]] = {f"layer{i}.{name}": [] for i in range(len(self.blocks)) for name in [
            "raw_input", "attn_norm_input", "q_out", "k_out", "v_out", "o_input", "o_out",
            "mlp_norm_input", "up_out", "down_out"
        ]}
        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        logits = x[:, -1, :] @ self.E.T
        return logits, cache


def centered(x: torch.Tensor) -> torch.Tensor:
    return x.float() - x.float().mean(dim=0, keepdim=True)


def svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(matrix.detach().float(), full_matrices=False)


def spectrum(matrix: torch.Tensor) -> Dict[str, float]:
    _, s, _ = svd(matrix)
    e = s.square()
    total = float(e.sum())
    if total <= 1e-12:
        return {"top1_energy": 0.0, "sigma1": 0.0}
    return {"top1_energy": float(e[0] / total), "sigma1": float(s[0])}


def top_right(matrix: torch.Tensor) -> torch.Tensor:
    _, _, vh = svd(matrix)
    return vh[0]


def top_left(matrix: torch.Tensor) -> torch.Tensor:
    u, _, _ = svd(matrix)
    return u[:, 0]


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float(((a @ b) / denom).square())


def flatten_cache(cache: Dict[str, List[torch.Tensor]], key: str) -> torch.Tensor:
    x = torch.cat(cache[key], dim=0)
    return x.reshape(-1, x.shape[-1])


def loss_metrics(model: ToyTransformer, data: Dict[str, object]) -> Dict[str, float]:
    logits, _ = model(data["tokens"])
    targets = data["targets"]
    losses = F.cross_entropy(logits, targets, reduction="none")
    pred = logits.argmax(dim=-1)
    out = {
        "loss": float((losses * data["weights"]).sum()),
        "accuracy": float((pred == targets).float().mean()),
    }
    for name, mask in {
        "common": torch.tensor([g == "A" for g in data["groups"]], dtype=torch.bool),
        "tail": torch.tensor([g != "A" for g in data["groups"]], dtype=torch.bool),
        "k_related": torch.tensor(["K" in f for f in data["families"]], dtype=torch.bool),
        "internal": torch.tensor(["internal" in f for f in data["families"]], dtype=torch.bool),
    }.items():
        out[f"{name}_loss"] = float(losses[mask].mean()) if bool(mask.any()) else 0.0
        out[f"{name}_accuracy"] = float((pred[mask] == targets[mask]).float().mean()) if bool(mask.any()) else 0.0
    return out


def measure(model: ToyTransformer, data: Dict[str, object], cfg: Config, mode: str, seed: int, step: int) -> Dict[str, object]:
    with torch.no_grad():
        logits, cache = model(data["tokens"])
        out: Dict[str, object] = {"mode": mode, "seed": seed, "step": step, **loss_metrics(model, data)}
        E = centered(model.E.detach())
        out.update({f"E_centered_{k}": v for k, v in spectrum(E).items()})
        rE = top_right(E)

        for layer, block in enumerate(model.blocks):
            raw = centered(flatten_cache(cache, f"layer{layer}.raw_input"))
            attn_in = centered(flatten_cache(cache, f"layer{layer}.attn_norm_input"))
            mlp_in = centered(flatten_cache(cache, f"layer{layer}.mlp_norm_input"))
            out.update({f"layer{layer}_raw_input_{k}": v for k, v in spectrum(raw).items()})
            out.update({f"layer{layer}_attn_norm_input_{k}": v for k, v in spectrum(attn_in).items()})
            out.update({f"layer{layer}_mlp_norm_input_{k}": v for k, v in spectrum(mlp_in).items()})
            r_raw = top_right(raw)
            r_attn = top_right(attn_in)
            r_mlp = top_right(mlp_in)
            out[f"layer{layer}_raw_pc1_sqcos_to_E"] = sqcos(r_raw, rE)
            if layer > 0:
                prev = centered(flatten_cache(cache, f"layer{layer-1}.raw_input"))
                out[f"layer{layer}_raw_pc1_sqcos_to_prev"] = sqcos(r_raw, top_right(prev))
            else:
                out[f"layer{layer}_raw_pc1_sqcos_to_prev"] = 1.0

            matrices = {
                "Wq": block.Wq.detach(),
                "Wk": block.Wk.detach(),
                "Wv": block.Wv.detach(),
                "Wo": block.Wo.detach(),
                "Wup": block.Wup.detach(),
                "Wdown": block.Wdown.detach(),
                "Bqk": block.Wq.detach().T @ block.Wk.detach(),
            }
            out_acts = {
                "Wq": centered(flatten_cache(cache, f"layer{layer}.q_out")),
                "Wk": centered(flatten_cache(cache, f"layer{layer}.k_out")),
                "Wv": centered(flatten_cache(cache, f"layer{layer}.v_out")),
                "Wo": centered(flatten_cache(cache, f"layer{layer}.o_out")),
                "Wup": centered(flatten_cache(cache, f"layer{layer}.up_out")),
                "Wdown": centered(flatten_cache(cache, f"layer{layer}.down_out")),
            }
            input_refs = {"Wq": r_attn, "Wk": r_attn, "Wv": r_attn, "Bqk": r_attn, "Wo": top_right(centered(flatten_cache(cache, f"layer{layer}.o_input"))), "Wup": r_mlp, "Wdown": top_right(centered(flatten_cache(cache, f"layer{layer}.up_out")))}
            for name, mat in matrices.items():
                out.update({f"layer{layer}_{name}_{k}": v for k, v in spectrum(mat).items()})
                out[f"layer{layer}_{name}_input_align_sqcos"] = sqcos(input_refs[name], top_right(mat))
                if name in out_acts:
                    out[f"layer{layer}_{name}_output_align_sqcos"] = sqcos(top_right(out_acts[name]), top_left(mat))
                else:
                    out[f"layer{layer}_{name}_output_align_sqcos"] = 0.0
        return out


def train_one(cfg: Config, mode: str, seed: int) -> List[Dict[str, object]]:
    set_seed(seed)
    data = build_data(cfg)
    model = ToyTransformer(data["E0"], cfg, seed, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    rows: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, cfg, mode, seed, step))
        if step == cfg.steps:
            break
        opt.zero_grad(set_to_none=True)
        logits, _ = model(data["tokens"])
        loss = (F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()
        loss.backward()
        opt.step()
    return rows


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["mode"]), int(row["step"])), []).append(row)
    out = []
    for (mode, step), items in sorted(groups.items()):
        row: Dict[str, object] = {"mode": mode, "step": step, "num_seeds": len(items)}
        keys = sorted(k for k in set.intersection(*(set(x) for x in items)) if k not in {"mode", "seed", "step"} and all(isinstance(x[k], (int, float)) for x in items))
        for k in keys:
            vals = np.array([float(x[k]) for x in items], dtype=np.float64)
            row[f"{k}_mean"] = float(vals.mean())
            row[f"{k}_std"] = float(vals.std(ddof=0))
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def final_for(agg: List[Dict[str, object]], mode: str) -> Dict[str, object]:
    return sorted([r for r in agg if r["mode"] == mode], key=lambda r: int(r["step"]))[-1]


def summarize(cfg: Config, agg: List[Dict[str, object]]) -> Dict[str, object]:
    out = {"key_numbers": {}, "checks": {}}
    for mode in ["prenorm", "normed_skip"]:
        f = final_for(agg, mode)
        out["key_numbers"][mode] = {
            "loss": f["loss_mean"],
            "accuracy": f["accuracy_mean"],
            "final_layer_raw_top1_energy": f[f"layer{cfg.num_layers-1}_raw_input_top1_energy_mean"],
            "final_layer_raw_pc1_sqcos_to_prev": f[f"layer{cfg.num_layers-1}_raw_pc1_sqcos_to_prev_mean"],
            "mean_param_top1_energy": float(np.mean([
                f[f"layer{l}_{name}_top1_energy_mean"]
                for l in range(cfg.num_layers)
                for name in ["Wq", "Wk", "Wv", "Wo", "Wup", "Wdown", "Bqk"]
            ])),
            "mean_param_input_align": float(np.mean([
                f[f"layer{l}_{name}_input_align_sqcos_mean"]
                for l in range(cfg.num_layers)
                for name in ["Wq", "Wk", "Wv", "Wo", "Wup", "Wdown", "Bqk"]
            ])),
            "mean_param_output_align": float(np.mean([
                f[f"layer{l}_{name}_output_align_sqcos_mean"]
                for l in range(cfg.num_layers)
                for name in ["Wq", "Wk", "Wv", "Wo", "Wup", "Wdown"]
            ])),
            "max_param_top1_energy": float(max(
                f[f"layer{l}_{name}_top1_energy_mean"]
                for l in range(cfg.num_layers)
                for name in ["Wq", "Wk", "Wv", "Wo", "Wup", "Wdown", "Bqk"]
            )),
        }
    p = out["key_numbers"]["prenorm"]
    n = out["key_numbers"]["normed_skip"]
    out["checks"] = {
        "normed_skip_reduces_final_raw_rep_top1_energy": n["final_layer_raw_top1_energy"] < p["final_layer_raw_top1_energy"],
        "normed_skip_reduces_mean_param_top1_energy": n["mean_param_top1_energy"] < p["mean_param_top1_energy"],
        "normed_skip_reduces_param_input_alignment": n["mean_param_input_align"] < p["mean_param_input_align"],
        "normed_skip_reduces_param_output_alignment": n["mean_param_output_align"] < p["mean_param_output_align"],
        "normed_skip_still_trains": n["accuracy"] >= 0.95,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--record_every", type=int, default=40)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument("--init_noise", type=float, default=0.01)
    parser.add_argument("--mlp_mult", type=int, default=2)
    cfg = Config(**vars(parser.parse_args()))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for mode in ["prenorm", "normed_skip"]:
        for seed in [int(x) for x in cfg.seeds.split(",") if x.strip()]:
            print(f"running mode={mode} seed={seed}", flush=True)
            rows.extend(train_one(cfg, mode, seed))
    agg = aggregate(rows)
    summary = summarize(cfg, agg)
    write_csv(outdir / "history.csv", rows)
    write_csv(outdir / "aggregate.csv", agg)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
