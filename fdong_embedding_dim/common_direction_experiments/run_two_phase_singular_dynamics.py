#!/usr/bin/env python3
"""Checkpoint-free diagnostic for two-phase singular-mode learning.

This script reuses the low-dimensional tied-embedding attention language task
from stage 6.  It records only small scalar diagnostics during plain CE
training:

  1. singular-vector drift of parameter/input/output top-1 directions;
  2. singular gain change;
  3. alignment between parameter singular directions and activation directions;
  4. gain-weighted feature effect, sigma_1^2 * alignment.

The falsifiable prediction is:

  if the two-phase story is correct, then in the shared-K Zipf condition the
  top singular direction should stabilize before the top singular gain stops
  increasing.  Therefore late training should show low subspace drift but
  still positive singular-energy growth and gain-weighted effect.
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


@dataclass
class Config:
    outdir: str
    seeds: str = "0,1,2,3,4"
    dim: int = 4
    steps: int = 2000
    lr: float = 0.03
    record_every: int = 20
    theta_deg: float = 12.0
    init_noise: float = 0.005
    residual_alpha: float = 0.0
    use_o_proj: bool = True


class AttnLM(torch.nn.Module):
    def __init__(
        self,
        e0: torch.Tensor,
        dim: int,
        seed: int,
        init_noise: float,
        residual_alpha: float,
        use_o_proj: bool,
    ):
        super().__init__()
        self.residual_alpha = residual_alpha
        self.use_o_proj = use_o_proj
        self.E = torch.nn.Parameter(e0.clone())
        eye = torch.eye(dim, dtype=torch.float32) * 0.1
        gen = torch.Generator().manual_seed(seed + 1729)
        self.Wq = torch.nn.Parameter(eye + init_noise * torch.randn(dim, dim, generator=gen))
        self.Wk = torch.nn.Parameter(eye + init_noise * torch.randn(dim, dim, generator=gen))
        self.Wv = torch.nn.Parameter(eye + init_noise * torch.randn(dim, dim, generator=gen))
        self.Wo = torch.nn.Parameter(eye + init_noise * torch.randn(dim, dim, generator=gen))
        self.scale = math.sqrt(dim)

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
        attn_out = attn_pre_o @ self.Wo.T if self.use_o_proj else attn_pre_o
        final_h = attn_out + self.residual_alpha * h2
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
            "final_h": final_h.detach(),
        }
        return logits, cache


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


def build_init_and_data(cfg: Config, condition: str) -> Dict[str, object]:
    centers = {
        "A": np.eye(cfg.dim, dtype=np.float32)[0],
        "B": np.eye(cfg.dim, dtype=np.float32)[1],
        "C": np.eye(cfg.dim, dtype=np.float32)[2],
        "D": np.eye(cfg.dim, dtype=np.float32)[3],
    }
    with_k = condition.startswith("withK")
    e_rows: List[np.ndarray] = []
    token_groups: List[str] = []
    if with_k:
        e_rows.append(np.ones(cfg.dim, dtype=np.float32) / math.sqrt(cfg.dim))
        token_groups.append("K")
    group_ids: Dict[str, List[int]] = {}
    for group, center in centers.items():
        ids = []
        for off in [0.0, cfg.theta_deg, -cfg.theta_deg]:
            ids.append(len(e_rows))
            e_rows.append(rot(center, off, cfg.dim))
            token_groups.append(group)
        group_ids[group] = ids
    e0 = torch.tensor(np.stack(e_rows), dtype=torch.float32)

    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    groups: List[str] = []
    families: List[str] = []
    for group in ["A", "B", "C", "D"]:
        i0, i1, i2 = group_ids[group]
        if with_k:
            k = 0
            patterns = [
                (i0, i1, k, "to_K"),
                (i1, k, i2, "from_K_1"),
                (k, i2, i0, "from_K_2"),
                (i2, i0, i1, "internal"),
            ]
        else:
            patterns = [
                (i0, i1, i2, "internal_1"),
                (i1, i2, i0, "internal_2"),
                (i2, i0, i1, "internal_3"),
            ]
        for a, b, y, family in patterns:
            c1.append(a)
            c2.append(b)
            targets.append(y)
            groups.append(group)
            families.append(family)

    if condition.endswith("zipf"):
        probs = {"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10}
    else:
        probs = {g: 0.25 for g in ["A", "B", "C", "D"]}
    weights = torch.tensor([probs[g] / sum(x == g for x in groups) for g in groups], dtype=torch.float32)
    weights = weights / weights.sum()
    return {
        "E0": e0,
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "weights": weights,
        "groups": groups,
        "families": families,
        "token_groups": token_groups,
    }


def weighted_centered(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.float().reshape(-1, 1)
    total = w.sum().clamp_min(1e-12)
    mu = (x.float() * w).sum(dim=0, keepdim=True) / total
    return (x.float() - mu) * torch.sqrt(w / total)


def centered(x: torch.Tensor) -> torch.Tensor:
    return x.float() - x.float().mean(dim=0, keepdim=True)


def svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(matrix.detach().float(), full_matrices=False)


def top_dirs(matrix: torch.Tensor) -> Tuple[torch.Tensor, float, float, torch.Tensor]:
    u, s, vh = svd(matrix)
    energy = s.square()
    total = float(energy.sum())
    top1_energy = float(energy[0] / energy.sum().clamp_min(1e-12)) if total > 0 else 0.0
    return u[:, 0], float(s[0]), top1_energy, vh[0]


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float(((a @ b) / denom).square())


def rank1_drift(prev: torch.Tensor | None, cur: torch.Tensor) -> float:
    if prev is None:
        return 0.0
    return math.sqrt(max(0.0, 1.0 - sqcos(prev, cur)))


def weighted_loss_metrics(model: AttnLM, data: Dict[str, object]) -> Dict[str, float]:
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
    out = {
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
    return out


def compute_grad(model: AttnLM, data: Dict[str, object]) -> Dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    logits, _ = model(data["c1"], data["c2"])
    loss = (F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()
    loss.backward()
    grads = {}
    for name in ["E", "Wq", "Wk", "Wv", "Wo"]:
        param = getattr(model, name)
        grads[name] = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
    model.zero_grad(set_to_none=True)
    return grads


def direction_pc1(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        matrix = centered(x)
    else:
        matrix = weighted_centered(x, weights)
    _, _, vh = svd(matrix)
    return vh[0]


def measure(
    model: AttnLM,
    data: Dict[str, object],
    cfg: Config,
    condition: str,
    seed: int,
    step: int,
    prev_dirs: Dict[str, torch.Tensor],
    prev_energy: Dict[str, float],
) -> Tuple[Dict[str, object], Dict[str, torch.Tensor], Dict[str, float]]:
    logits, cache = model(data["c1"], data["c2"])
    del logits
    weights = data["weights"]
    doubled_weights = torch.cat([weights, weights], dim=0) / 2.0
    feature_dirs = {
        "E_centered": direction_pc1(model.E.detach()),
        "X_query": direction_pc1(cache["h_query"], weights),
        "X_key": direction_pc1(cache["h_key"], doubled_weights),
        "X_value": direction_pc1(cache["h_value"], doubled_weights),
        "Q_out": direction_pc1(cache["q_out"], weights),
        "K_out": direction_pc1(cache["k_out"], doubled_weights),
        "V_out": direction_pc1(cache["v_out"], doubled_weights),
        "O_in": direction_pc1(cache["o_in"], weights),
        "O_out": direction_pc1(cache["o_out"], weights),
        "Final_h": direction_pc1(cache["final_h"], weights),
    }
    matrices = {
        "E_centered": centered(model.E.detach()),
        "Wq": model.Wq.detach(),
        "Wk": model.Wk.detach(),
        "Wv": model.Wv.detach(),
        "Wo": model.Wo.detach(),
        "Bqk": model.Wq.detach().T @ model.Wk.detach(),
    }
    module_features = {
        "Wq": ("X_query", "Q_out"),
        "Wk": ("X_key", "K_out"),
        "Wv": ("X_value", "V_out"),
        "Wo": ("O_in", "O_out"),
        "Bqk": ("X_query", "K_out"),
        "E_centered": ("Final_h", "E_centered"),
    }
    row: Dict[str, object] = {
        "condition": condition,
        "seed": seed,
        "step": step,
        **weighted_loss_metrics(model, data),
    }
    new_dirs: Dict[str, torch.Tensor] = {}
    new_energy: Dict[str, float] = {}
    for name, matrix in matrices.items():
        u1, sigma1, top1_energy, v1 = top_dirs(matrix)
        input_feature, output_feature = module_features[name]
        input_align = sqcos(feature_dirs[input_feature], v1)
        if name == "E_centered":
            # E is token-by-hidden.  Its right singular vector is a hidden-space
            # read/write direction, but its left singular vector lives in token
            # index space.  Do not compare that left vector to hidden activations.
            output_align = 0.0
        else:
            output_align = sqcos(feature_dirs[output_feature], u1)
        sigma_energy = sigma1 * sigma1
        new_dirs[f"{name}.v"] = v1.detach()
        new_dirs[f"{name}.u"] = u1.detach()
        new_energy[name] = sigma_energy
        row[f"{name}_sigma1"] = sigma1
        row[f"{name}_sigma1_sq"] = sigma_energy
        row[f"{name}_top1_energy"] = top1_energy
        row[f"{name}_right_drift"] = rank1_drift(prev_dirs.get(f"{name}.v"), v1)
        row[f"{name}_left_drift"] = rank1_drift(prev_dirs.get(f"{name}.u"), u1)
        row[f"{name}_sigma1_sq_delta"] = sigma_energy - prev_energy.get(name, sigma_energy)
        row[f"{name}_input_align"] = input_align
        row[f"{name}_output_align"] = output_align
        row[f"{name}_input_weighted_effect"] = sigma_energy * input_align
        row[f"{name}_output_weighted_effect"] = sigma_energy * output_align
    return row, new_dirs, new_energy


def train_one(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    set_seed(seed)
    data = build_init_and_data(cfg, condition)
    model = AttnLM(data["E0"], cfg.dim, seed, cfg.init_noise, cfg.residual_alpha, cfg.use_o_proj)
    rows: List[Dict[str, object]] = []
    prev_dirs: Dict[str, torch.Tensor] = {}
    prev_energy: Dict[str, float] = {}
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            row, prev_dirs, prev_energy = measure(model, data, cfg, condition, seed, step, prev_dirs, prev_energy)
            rows.append(row)
        if step == cfg.steps:
            break
        grads = compute_grad(model, data)
        with torch.no_grad():
            for name in ["E", "Wq", "Wk", "Wv", "Wo"]:
                getattr(model, name).sub_(cfg.lr * grads[name])
    return rows


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), int(row["step"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (condition, step), items in sorted(grouped.items()):
        agg: Dict[str, object] = {"condition": condition, "step": step, "num_seeds": len(items)}
        keys = sorted(
            k
            for k in set.intersection(*(set(x) for x in items))
            if k not in {"condition", "seed", "step"} and all(isinstance(x[k], (int, float)) for x in items)
        )
        for key in keys:
            vals = np.array([float(x[key]) for x in items], dtype=np.float64)
            agg[f"{key}_mean"] = float(vals.mean())
            agg[f"{key}_std"] = float(vals.std(ddof=0))
        out.append(agg)
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def condition_rows(agg: List[Dict[str, object]], condition: str) -> List[Dict[str, object]]:
    return sorted([x for x in agg if x["condition"] == condition], key=lambda x: int(x["step"]))


def window_mean(rows: List[Dict[str, object]], key: str, lo_frac: float, hi_frac: float) -> float:
    steps = [int(r["step"]) for r in rows]
    max_step = max(steps)
    selected = [r for r in rows if int(r["step"]) >= lo_frac * max_step and int(r["step"]) <= hi_frac * max_step]
    return float(np.mean([float(r[key]) for r in selected]))


def summarize(cfg: Config, agg: List[Dict[str, object]]) -> Dict[str, object]:
    modules = ["E_centered", "Wq", "Wk", "Wv", "Wo", "Bqk"]
    summary: Dict[str, object] = {"conditions": {}}
    for condition in ["noK_uniform", "withK_uniform", "withK_zipf"]:
        rows = condition_rows(agg, condition)
        final = rows[-1]
        cond = {
            "final_loss": final["loss_mean"],
            "final_common_loss": final["common_loss_mean"],
            "final_tail_loss": final["tail_loss_mean"],
            "modules": {},
        }
        for module in modules:
            early_drift = window_mean(rows, f"{module}_right_drift_mean", 0.05, 0.25)
            late_drift = window_mean(rows, f"{module}_right_drift_mean", 0.70, 1.00)
            early_gain = window_mean(rows, f"{module}_sigma1_sq_delta_mean", 0.05, 0.25)
            late_gain = window_mean(rows, f"{module}_sigma1_sq_delta_mean", 0.70, 1.00)
            cond["modules"][module] = {
                "final_sigma1_sq": final[f"{module}_sigma1_sq_mean"],
                "final_top1_energy": final[f"{module}_top1_energy_mean"],
                "final_input_align": final[f"{module}_input_align_mean"],
                "final_output_align": final[f"{module}_output_align_mean"],
                "final_input_weighted_effect": final[f"{module}_input_weighted_effect_mean"],
                "final_output_weighted_effect": final[f"{module}_output_weighted_effect_mean"],
                "early_right_drift": early_drift,
                "late_right_drift": late_drift,
                "early_sigma1_sq_delta": early_gain,
                "late_sigma1_sq_delta": late_gain,
                "two_phase_signature": bool(late_drift < early_drift * 0.5 and late_gain > 0.0),
            }
        summary["conditions"][condition] = cond
    zipf_modules = summary["conditions"]["withK_zipf"]["modules"]
    summary["main_checks"] = {
        "Bqk_late_drift_smaller_than_early": zipf_modules["Bqk"]["late_right_drift"] < zipf_modules["Bqk"]["early_right_drift"] * 0.5,
        "Bqk_late_gain_positive": zipf_modules["Bqk"]["late_sigma1_sq_delta"] > 0.0,
        "at_least_one_attention_module_two_phase": any(
            zipf_modules[m]["two_phase_signature"] for m in ["Wq", "Wk", "Wv", "Wo", "Bqk"]
        ),
        "sharedK_zipf_Bqk_more_singular_than_noK": zipf_modules["Bqk"]["final_top1_energy"]
        > summary["conditions"]["noK_uniform"]["modules"]["Bqk"]["final_top1_energy"],
    }
    return summary


def plot(out_path: Path, agg: List[Dict[str, object]]) -> None:
    modules = ["E_centered", "Wq", "Wk", "Wv", "Wo", "Bqk"]
    conditions = ["noK_uniform", "withK_uniform", "withK_zipf"]
    fig, axes = plt.subplots(len(modules), 4, figsize=(20, 4 * len(modules)), sharex=True)
    for r, module in enumerate(modules):
        panels = [
            (f"{module}_right_drift_mean", "right-vector drift"),
            (f"{module}_sigma1_sq_delta_mean", "singular energy delta"),
            (f"{module}_input_align_mean", "input alignment"),
            (f"{module}_input_weighted_effect_mean", "input weighted effect"),
        ]
        for c, (metric, title) in enumerate(panels):
            ax = axes[r, c]
            for condition in conditions:
                rows = condition_rows(agg, condition)
                xs = np.array([int(x["step"]) for x in rows])
                ys = np.array([float(x[metric]) for x in rows])
                ax.plot(xs, ys, label=condition, linewidth=1.6)
            ax.set_title(f"{module}: {title}")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Two-phase singular-mode diagnostic: direction drift vs singular gain", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--record_every", type=int, default=20)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument("--init_noise", type=float, default=0.005)
    parser.add_argument("--residual_alpha", type=float, default=0.0)
    parser.add_argument("--no_o_proj", action="store_true")
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
        residual_alpha=args.residual_alpha,
        use_o_proj=not args.no_o_proj,
    )
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for condition in ["noK_uniform", "withK_uniform", "withK_zipf"]:
        for seed in [int(x) for x in cfg.seeds.split(",") if x.strip()]:
            print(f"running condition={condition} seed={seed}", flush=True)
            rows.extend(train_one(cfg, condition, seed))
    agg = aggregate(rows)
    summary = summarize(cfg, agg)
    write_csv(outdir / "history.csv", rows)
    write_csv(outdir / "aggregate.csv", agg)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(outdir / "two_phase_dynamics.png", agg)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
