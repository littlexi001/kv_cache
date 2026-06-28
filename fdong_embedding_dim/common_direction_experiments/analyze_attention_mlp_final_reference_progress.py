#!/usr/bin/env python3
"""Final-reference singular progress in a toy attention+MLP language model.

This extends analyze_final_reference_progress.py by adding a residual MLP after
the attention block:

    h_attn = attn_out + residual_alpha * h_query
    h_mlp  = h_attn + W_down GELU(W_up h_attn)
    logits = h_mlp E^T

For each matrix, the final checkpoint is used as the reference.  We compare
normalized progress of top singular directions and top singular values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_two_phase_singular_dynamics import build_init_and_data, set_seed


@dataclass
class Config:
    outdir: str
    seeds: str = "0,1,2,3,4"
    dim: int = 4
    mlp_mult: int = 4
    steps: int = 5000
    lr: float = 0.03
    record_every: int = 50
    theta_deg: float = 12.0
    init_noise: float = 0.005
    residual_alpha: float = 1.0
    mlp_residual_alpha: float = 1.0
    data_condition: str = "withK_zipf"


class AttnMLPLM(torch.nn.Module):
    def __init__(self, e0: torch.Tensor, cfg: Config, seed: int):
        super().__init__()
        self.residual_alpha = cfg.residual_alpha
        self.mlp_residual_alpha = cfg.mlp_residual_alpha
        self.dim = cfg.dim
        self.mlp_dim = cfg.dim * cfg.mlp_mult
        self.E = torch.nn.Parameter(e0.clone())
        gen = torch.Generator().manual_seed(seed + 1729)
        eye = torch.eye(cfg.dim, dtype=torch.float32) * 0.1
        self.Wq = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.Wk = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.Wv = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.Wo = torch.nn.Parameter(eye + cfg.init_noise * torch.randn(cfg.dim, cfg.dim, generator=gen))
        self.Wup = torch.nn.Parameter(0.1 * torch.randn(self.mlp_dim, cfg.dim, generator=gen) / math.sqrt(cfg.dim))
        self.Wdown = torch.nn.Parameter(0.1 * torch.randn(cfg.dim, self.mlp_dim, generator=gen) / math.sqrt(self.mlp_dim))
        self.scale = math.sqrt(cfg.dim)

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h1 = self.E[c1]
        h2 = self.E[c2]
        q = h2 @ self.Wq.T
        k1 = h1 @ self.Wk.T
        k2 = h2 @ self.Wk.T
        v1 = h1 @ self.Wv.T
        v2 = h2 @ self.Wv.T
        scores = torch.stack([(q * k1).sum(dim=-1), (q * k2).sum(dim=-1)], dim=-1) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn_pre_o = attn[:, 0:1] * v1 + attn[:, 1:2] * v2
        attn_out = attn_pre_o @ self.Wo.T
        h_attn = attn_out + self.residual_alpha * h2
        mlp_pre = h_attn @ self.Wup.T
        mlp_act = F.gelu(mlp_pre)
        mlp_out = mlp_act @ self.Wdown.T
        final_h = h_attn + self.mlp_residual_alpha * mlp_out
        logits = final_h @ self.E.T
        cache = {
            "h_query": h2.detach(),
            "h_key": torch.cat([h1, h2], dim=0).detach(),
            "h_value": torch.cat([h1, h2], dim=0).detach(),
            "q_out": q.detach(),
            "k_out": torch.cat([k1, k2], dim=0).detach(),
            "v_out": torch.cat([v1, v2], dim=0).detach(),
            "o_in": attn_pre_o.detach(),
            "o_out": attn_out.detach(),
            "h_attn": h_attn.detach(),
            "mlp_pre": mlp_pre.detach(),
            "mlp_act": mlp_act.detach(),
            "mlp_out": mlp_out.detach(),
            "final_h": final_h.detach(),
        }
        return logits, cache


def weighted_loss_metrics(model: AttnMLPLM, data: Dict[str, object]) -> Dict[str, float]:
    logits, _ = model(data["c1"], data["c2"])
    targets = data["targets"]
    weights = data["weights"]
    losses = F.cross_entropy(logits, targets, reduction="none")
    pred = logits.argmax(dim=-1)
    target_logits = logits[torch.arange(len(targets)), targets]
    tmp = logits.detach().clone()
    tmp[torch.arange(len(targets)), targets] = -float("inf")
    margins = target_logits.detach() - tmp.max(dim=-1).values
    common = torch.tensor([g == "A" for g in data["groups"]], dtype=torch.bool)
    tail = ~common
    return {
        "loss": float((losses.detach() * weights).sum()),
        "accuracy": float((pred == targets).float().mean()),
        "margin_weighted": float((margins * weights).sum()),
        "common_loss": float(losses.detach()[common].mean()),
        "tail_loss": float(losses.detach()[tail].mean()),
        "common_accuracy": float((pred[common] == targets[common]).float().mean()),
        "tail_accuracy": float((pred[tail] == targets[tail]).float().mean()),
        "common_margin": float(margins[common].mean()),
        "tail_margin": float(margins[tail].mean()),
    }


def compute_grad(model: AttnMLPLM, data: Dict[str, object]) -> Dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits, _ = model(data["c1"], data["c2"])
    loss = (F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()
    loss.backward()
    grads = {}
    for name in ["E", "Wq", "Wk", "Wv", "Wo", "Wup", "Wdown"]:
        param = getattr(model, name)
        grads[name] = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
    model.zero_grad(set_to_none=True)
    return grads


def centered(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float() - x.detach().float().mean(dim=0, keepdim=True)


def snapshot(model: AttnMLPLM) -> Dict[str, torch.Tensor]:
    return {
        "E_centered": centered(model.E),
        "Wq": model.Wq.detach().clone(),
        "Wk": model.Wk.detach().clone(),
        "Wv": model.Wv.detach().clone(),
        "Wo": model.Wo.detach().clone(),
        "Bqk": model.Wq.detach().T @ model.Wk.detach(),
        "Wup": model.Wup.detach().clone(),
        "Wdown": model.Wdown.detach().clone(),
        "Bmlp": model.Wdown.detach() @ model.Wup.detach(),
    }


def top_svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
    u, s, vh = torch.linalg.svd(matrix.detach().float(), full_matrices=False)
    return u[:, 0].detach(), float(s[0]), vh[0].detach()


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float(((a @ b) / denom).square())


def progress(numer: float, denom: float) -> float:
    if abs(denom) <= 1e-12:
        return 1.0
    return numer / denom


def train_collect(cfg: Config, seed: int) -> List[Dict[str, object]]:
    set_seed(seed)
    proxy_cfg = type(
        "ProxyConfig",
        (),
        {
            "dim": cfg.dim,
            "theta_deg": cfg.theta_deg,
        },
    )()
    data = build_init_and_data(proxy_cfg, cfg.data_condition)
    model = AttnMLPLM(data["E0"], cfg, seed)
    checkpoints: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            checkpoints.append({"step": step, "matrices": snapshot(model), "metrics": weighted_loss_metrics(model, data)})
        if step == cfg.steps:
            break
        grads = compute_grad(model, data)
        with torch.no_grad():
            for name in ["E", "Wq", "Wk", "Wv", "Wo", "Wup", "Wdown"]:
                getattr(model, name).sub_(cfg.lr * grads[name])

    modules = ["E_centered", "Wq", "Wk", "Wv", "Wo", "Bqk", "Wup", "Wdown", "Bmlp"]
    first = checkpoints[0]["matrices"]
    final = checkpoints[-1]["matrices"]
    refs: Dict[str, Dict[str, object]] = {}
    for module in modules:
        u0, sigma0, v0 = top_svd(first[module])
        uT, sigmaT, vT = top_svd(final[module])
        refs[module] = {
            "uT": uT,
            "vT": vT,
            "sigma0": sigma0,
            "sigmaT": sigmaT,
            "right_c0": sqcos(v0, vT),
            "left_c0": sqcos(u0, uT),
        }

    rows: List[Dict[str, object]] = []
    for ckpt in checkpoints:
        for module in modules:
            u, sigma, v = top_svd(ckpt["matrices"][module])
            ref = refs[module]
            right_closeness = sqcos(v, ref["vT"])
            left_closeness = sqcos(u, ref["uT"])
            sigma_prog = progress(sigma - float(ref["sigma0"]), float(ref["sigmaT"]) - float(ref["sigma0"]))
            right_prog = progress(right_closeness - float(ref["right_c0"]), 1.0 - float(ref["right_c0"]))
            left_prog = progress(left_closeness - float(ref["left_c0"]), 1.0 - float(ref["left_c0"]))
            rows.append(
                {
                    "seed": seed,
                    "step": int(ckpt["step"]),
                    "module": module,
                    "block": "attention" if module in {"Wq", "Wk", "Wv", "Wo", "Bqk"} else ("mlp" if module in {"Wup", "Wdown", "Bmlp"} else "embedding"),
                    "loss": ckpt["metrics"]["loss"],
                    "tail_loss": ckpt["metrics"]["tail_loss"],
                    "common_loss": ckpt["metrics"]["common_loss"],
                    "sigma1": sigma,
                    "sigma0": ref["sigma0"],
                    "sigmaT": ref["sigmaT"],
                    "right_closeness_to_final": right_closeness,
                    "right_initial_closeness_to_final": ref["right_c0"],
                    "left_closeness_to_final": left_closeness,
                    "left_initial_closeness_to_final": ref["left_c0"],
                    "sigma_progress": sigma_prog,
                    "right_vector_progress": right_prog,
                    "left_vector_progress": left_prog,
                    "right_minus_sigma_progress": right_prog - sigma_prog,
                    "left_minus_sigma_progress": left_prog - sigma_prog,
                    "abs_right_minus_sigma_progress": abs(right_prog - sigma_prog),
                    "abs_left_minus_sigma_progress": abs(left_prog - sigma_prog),
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["module"]), int(row["step"])), []).append(row)
    numeric = [
        k
        for k, v in rows[0].items()
        if k not in {"seed", "step", "module", "block"} and isinstance(v, (int, float))
    ]
    out: List[Dict[str, object]] = []
    for (module, step), items in sorted(grouped.items()):
        rec: Dict[str, object] = {"module": module, "block": items[0]["block"], "step": step, "num_seeds": len(items)}
        for key in numeric:
            vals = np.array([float(x[key]) for x in items], dtype=np.float64)
            rec[f"{key}_mean"] = float(vals.mean())
            rec[f"{key}_std"] = float(vals.std(ddof=0))
        out.append(rec)
    return out


def first_reach(rows: List[Dict[str, object]], key: str, threshold: float) -> int | None:
    ordered = sorted(rows, key=lambda r: int(r["step"]))
    for row in ordered:
        if float(row[key]) >= threshold:
            return int(row["step"])
    return None


def summarize(rows: List[Dict[str, object]], agg: List[Dict[str, object]]) -> Dict[str, object]:
    modules = ["Wq", "Wk", "Wv", "Wo", "Bqk", "Wup", "Wdown", "Bmlp", "E_centered"]
    summary: Dict[str, object] = {"modules": {}, "blocks": {}}
    for module in modules:
        ars = [r for r in agg if r["module"] == module]
        raw = [r for r in rows if r["module"] == module]
        by_seed: Dict[int, List[Dict[str, object]]] = {}
        for row in raw:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        rec: Dict[str, object] = {
            "block": raw[0]["block"],
            "mean_abs_right_minus_sigma_progress": float(np.mean([float(r["abs_right_minus_sigma_progress_mean"]) for r in ars])),
            "mean_abs_left_minus_sigma_progress": float(np.mean([float(r["abs_left_minus_sigma_progress_mean"]) for r in ars])),
            "max_abs_right_minus_sigma_progress": float(np.max([float(r["abs_right_minus_sigma_progress_mean"]) for r in ars])),
            "final_sigma1_mean": sorted(ars, key=lambda r: int(r["step"]))[-1]["sigma1_mean"],
            "final_loss_mean": sorted(ars, key=lambda r: int(r["step"]))[-1]["loss_mean"],
            "final_tail_loss_mean": sorted(ars, key=lambda r: int(r["step"]))[-1]["tail_loss_mean"],
        }
        for th in [0.5, 0.8, 0.9, 0.95]:
            sigma_steps = []
            right_steps = []
            left_steps = []
            for items in by_seed.values():
                s = first_reach(items, "sigma_progress", th)
                r = first_reach(items, "right_vector_progress", th)
                l = first_reach(items, "left_vector_progress", th)
                if s is not None:
                    sigma_steps.append(s)
                if r is not None:
                    right_steps.append(r)
                if l is not None:
                    left_steps.append(l)
            rec[f"t{int(th*100)}_sigma_step_mean"] = float(np.mean(sigma_steps)) if sigma_steps else None
            rec[f"t{int(th*100)}_right_step_mean"] = float(np.mean(right_steps)) if right_steps else None
            rec[f"t{int(th*100)}_left_step_mean"] = float(np.mean(left_steps)) if left_steps else None
            rec[f"t{int(th*100)}_right_minus_sigma_step_mean"] = (
                rec[f"t{int(th*100)}_right_step_mean"] - rec[f"t{int(th*100)}_sigma_step_mean"]
                if rec[f"t{int(th*100)}_right_step_mean"] is not None and rec[f"t{int(th*100)}_sigma_step_mean"] is not None
                else None
            )
        summary["modules"][module] = rec

    for block in ["attention", "mlp"]:
        mods = [m for m, r in summary["modules"].items() if r["block"] == block]
        summary["blocks"][block] = {
            "modules": mods,
            "mean_abs_right_minus_sigma_progress": float(np.mean([summary["modules"][m]["mean_abs_right_minus_sigma_progress"] for m in mods])),
            "mean_t90_right_minus_sigma_step": float(np.mean([summary["modules"][m]["t90_right_minus_sigma_step_mean"] for m in mods])),
            "mean_t95_right_minus_sigma_step": float(np.mean([summary["modules"][m]["t95_right_minus_sigma_step_mean"] for m in mods])),
        }
    return summary


def plot(path: Path, agg: List[Dict[str, object]]) -> None:
    modules = ["Wq", "Wk", "Wv", "Wo", "Bqk", "Wup", "Wdown", "Bmlp"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True, sharey=True)
    for ax, module in zip(axes.flat, modules):
        rows = sorted([r for r in agg if r["module"] == module], key=lambda r: int(r["step"]))
        xs = [int(r["step"]) for r in rows]
        ax.plot(xs, [float(r["right_vector_progress_mean"]) for r in rows], label="right vector progress")
        ax.plot(xs, [float(r["left_vector_progress_mean"]) for r in rows], label="left vector progress")
        ax.plot(xs, [float(r["sigma_progress_mean"]) for r in rows], label="sigma progress")
        ax.set_title(f"{module} ({rows[0]['block']})")
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Attention vs MLP final-reference progress")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--mlp_mult", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--record_every", type=int, default=50)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument("--init_noise", type=float, default=0.005)
    parser.add_argument("--residual_alpha", type=float, default=1.0)
    parser.add_argument("--mlp_residual_alpha", type=float, default=1.0)
    parser.add_argument("--data_condition", default="withK_zipf")
    args = parser.parse_args()
    cfg = Config(**vars(args))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for seed in [int(x) for x in cfg.seeds.split(",") if x.strip()]:
        print(f"running attention+mlp final-reference progress seed={seed}", flush=True)
        rows.extend(train_collect(cfg, seed))
    agg = aggregate(rows)
    summary = summarize(rows, agg)
    write_csv(outdir / "history.csv", rows)
    write_csv(outdir / "aggregate.csv", agg)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    plot(outdir / "attention_mlp_final_reference_progress.png", agg)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
