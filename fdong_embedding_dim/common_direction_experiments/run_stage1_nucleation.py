#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    dim: int = 8
    branches: int = 8
    steps: int = 400
    lr: float = 0.08
    record_every: int = 10
    reweight_alpha: float = 0.5


class TiedConcatLM(torch.nn.Module):
    def __init__(self, E0: torch.Tensor, M0: torch.Tensor):
        super().__init__()
        self.E = torch.nn.Parameter(E0.clone())
        self.M = torch.nn.Parameter(M0.clone())

    def hidden(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.E[c1], self.E[c2]], dim=-1)
        return x @ self.M.T

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        return self.hidden(c1, c2) @ self.E.T


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def spectrum(x: torch.Tensor) -> Dict[str, float]:
    s = torch.linalg.svdvals(x.detach())
    s1 = float(s[0]) if len(s) else 0.0
    s2 = float(s[1]) if len(s) > 1 else 0.0
    sq = s.square()
    total = float(sq.sum())
    return {
        "sigma1": s1,
        "sigma2": s2,
        "sigma1_over_sigma2": s1 / max(s2, 1e-12),
        "top1_energy": float(sq[0]) / max(total, 1e-12),
    }


def top_right_vector(x: torch.Tensor) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(x.detach(), full_matrices=False)
    return vh[0]


def make_vocab(branches: int) -> Tuple[List[str], Dict[str, int]]:
    names = ["K"]
    for prefix in ["C", "A", "T", "S"]:
        names.extend([f"{prefix}{i}" for i in range(branches)])
    return names, {name: i for i, name in enumerate(names)}


def build_data(
    condition: str, branches: int, token_to_id: Dict[str, int], alpha: float
) -> Dict[str, object]:
    shared_target = condition.startswith("shared")
    k_input = "k_input" in condition and "no_k_input" not in condition
    low_diversity = "lowdiv" in condition
    do_reweight = condition.endswith("reweight")

    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    families: List[str] = []
    k_target_mask: List[bool] = []

    for i in range(branches):
        context_i = i % 2 if low_diversity else i
        c1.append(token_to_id[f"C{context_i}"])
        c2.append(token_to_id[f"A{context_i}"])
        targets.append(token_to_id["K"] if shared_target else token_to_id[f"S{i}"])
        families.append("shared_target" if shared_target else "distributed_target")
        k_target_mask.append(shared_target)

        c1.append(token_to_id["K"] if k_input else token_to_id[f"A{i}"])
        c2.append(token_to_id[f"C{i}"])
        targets.append(token_to_id[f"T{i}"])
        families.append("k_continuation" if k_input else "ordinary_continuation")
        k_target_mask.append(False)

    target_counts: Dict[int, int] = {}
    for target in targets:
        target_counts[target] = target_counts.get(target, 0) + 1
    weights = torch.ones(len(targets), dtype=torch.float32)
    if do_reweight:
        weights = torch.tensor(
            [target_counts[t] ** (-alpha) for t in targets], dtype=torch.float32
        )
    weights = weights / weights.sum()
    return {
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "weights": weights,
        "families": families,
        "k_target_mask": torch.tensor(k_target_mask, dtype=torch.bool),
    }


def initialize(vocab_size: int, dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    E = torch.randn(vocab_size, dim)
    E = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eye = torch.eye(dim)
    M = torch.cat([eye, eye], dim=1) / math.sqrt(2.0)
    M = M + 0.01 * torch.randn_like(M)
    return E, M


def exact_gradient_parts(
    model: TiedConcatLM, data: Dict[str, object]
) -> Dict[str, torch.Tensor]:
    c1 = data["c1"]
    c2 = data["c2"]
    targets = data["targets"]
    weights = data["weights"]
    E = model.E
    M = model.M
    x = torch.cat([E[c1], E[c2]], dim=-1)
    h = x @ M.T
    logits = h @ E.T
    probs = logits.softmax(dim=-1)
    gz = probs
    gz = gz.clone()
    gz[torch.arange(len(targets)), targets] -= 1.0
    gz = gz * weights[:, None]

    g_output = gz.T @ h
    gh = gz @ E
    gx = gh @ M
    g_input = torch.zeros_like(E)
    g_input.index_add_(0, c1, gx[:, : E.shape[1]])
    g_input.index_add_(0, c2, gx[:, E.shape[1] :])
    g_m = gh.T @ x
    return {
        "g_output": g_output,
        "g_input": g_input,
        "g_total_e": g_output + g_input,
        "g_m": g_m,
        "h": h,
        "logits": logits,
    }


def family_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    families: List[str],
) -> Dict[str, float]:
    losses = F.cross_entropy(logits, targets, reduction="none")
    pred = logits.argmax(dim=-1)
    values: Dict[str, float] = {}
    for family in sorted(set(families)):
        idx = torch.tensor([x == family for x in families], dtype=torch.bool)
        rows = logits[idx].detach().clone()
        tgts = targets[idx]
        target_logits = rows[torch.arange(len(tgts)), tgts]
        rows[torch.arange(len(tgts)), tgts] = -float("inf")
        margins = target_logits - rows.max(dim=-1).values
        values[f"{family}_loss"] = float(losses[idx].mean())
        values[f"{family}_accuracy"] = float((pred[idx] == tgts).float().mean())
        values[f"{family}_margin"] = float(margins.mean())
    return values


def hidden_family_metrics(h: torch.Tensor, families: List[str]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    normalized = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    for family in sorted(set(families)):
        idx = torch.tensor([x == family for x in families], dtype=torch.bool)
        subset = h[idx]
        spec = spectrum(subset)
        values[f"{family}_hidden_top1_energy"] = spec["top1_energy"]
        n = int(idx.sum())
        if n > 1:
            norm_subset = normalized[idx]
            sim = norm_subset @ norm_subset.T
            offdiag = sim[~torch.eye(n, dtype=torch.bool)]
            values[f"{family}_hidden_pairwise_cosine"] = float(offdiag.mean())
        else:
            values[f"{family}_hidden_pairwise_cosine"] = 0.0
    return values


def measure(
    model: TiedConcatLM,
    data: Dict[str, object],
    condition: str,
    seed: int,
    step: int,
    k_id: int,
) -> Dict[str, object]:
    with torch.no_grad():
        parts = exact_gradient_parts(model, data)
        h = parts["h"]
        logits = parts["logits"]
        k_mask = data["k_target_mask"]
        if bool(k_mask.any()):
            k_context_mean = h[k_mask].mean(dim=0)
            k_context_cosine = cosine(model.E[k_id], k_context_mean)
            neg_k_output_grad = -parts["g_output"][k_id]
            k_grad_context_cosine = cosine(neg_k_output_grad, k_context_mean)
        else:
            k_context_cosine = 0.0
            k_grad_context_cosine = 0.0

        e_top = top_right_vector(model.E)
        row: Dict[str, object] = {
            "condition": condition,
            "seed": seed,
            "step": step,
            "weighted_loss": float(
                (F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()
            ),
            "k_context_cosine": k_context_cosine,
            "k_gradient_context_cosine": k_grad_context_cosine,
            "k_top_e_cosine_abs": abs(cosine(model.E[k_id], e_top)),
        }
        for name, matrix in [
            ("e", model.E),
            ("m", model.M),
            ("hidden", h),
            ("g_output", parts["g_output"]),
            ("g_input", parts["g_input"]),
            ("g_total_e", parts["g_total_e"]),
            ("g_m", parts["g_m"]),
        ]:
            for metric, value in spectrum(matrix).items():
                row[f"{name}_{metric}"] = value
        row.update(family_metrics(logits, data["targets"], data["families"]))
        row.update(hidden_family_metrics(h, data["families"]))
        return row


def run_one(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    vocab, token_to_id = make_vocab(cfg.branches)
    E0, M0 = initialize(len(vocab), cfg.dim, seed)
    model = TiedConcatLM(E0, M0)
    data = build_data(condition, cfg.branches, token_to_id, cfg.reweight_alpha)
    rows: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, condition, seed, step, token_to_id["K"]))
        if step == cfg.steps:
            break
        logits = model(data["c1"], data["c2"])
        losses = F.cross_entropy(logits, data["targets"], reduction="none")
        loss = (losses * data["weights"]).sum()
        loss.backward()
        with torch.no_grad():
            model.E -= cfg.lr * model.E.grad
            model.M -= cfg.lr * model.M.grad
            model.E.grad = None
            model.M.grad = None
    return rows


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), int(row["step"])), []).append(row)
    result: List[Dict[str, object]] = []
    for (condition, step), items in sorted(groups.items()):
        out: Dict[str, object] = {"condition": condition, "step": step, "num_seeds": len(items)}
        numeric = sorted(
            key
            for key in set.intersection(*(set(item) for item in items))
            if key not in {"seed", "step"}
            and all(isinstance(item[key], (int, float)) for item in items)
        )
        for key in numeric:
            vals = np.array([float(x[key]) for x in items], dtype=np.float64)
            out[f"{key}_mean"] = float(vals.mean())
            out[f"{key}_std"] = float(vals.std(ddof=0))
        result.append(out)
    return result


def get_agg(agg: List[Dict[str, object]], condition: str, step: int) -> Dict[str, object]:
    return next(x for x in agg if x["condition"] == condition and x["step"] == step)


def summarize(cfg: Config, agg: List[Dict[str, object]]) -> Dict[str, object]:
    distributed = get_agg(agg, "distributed_high_no_k_input", 0)
    shared = get_agg(agg, "shared_high_no_k_input", 0)
    shared_input = get_agg(agg, "shared_high_k_input", 0)
    precedence_step = min(50, cfg.steps)
    distributed_early = get_agg(agg, "distributed_high_no_k_input", precedence_step)
    shared_early = get_agg(agg, "shared_high_no_k_input", precedence_step)
    reweighted_final = get_agg(agg, "shared_high_k_input_reweight", cfg.steps)
    baseline_final = get_agg(agg, "shared_high_k_input", cfg.steps)
    checks = {
        "output_gradient_mode_stronger": shared["g_output_sigma1_over_sigma2_mean"]
        > distributed["g_output_sigma1_over_sigma2_mean"] * 1.2,
        "k_gradient_aligns_context": shared["k_gradient_context_cosine_mean"] > 0.9,
        "shared_input_creates_hidden_mode": shared_input["k_continuation_hidden_top1_energy_mean"]
        > shared["ordinary_continuation_hidden_top1_energy_mean"] * 1.1,
        "gradient_mode_precedes_parameter_divergence": shared["e_top1_energy_mean"]
        == distributed["e_top1_energy_mean"]
        and shared_early["e_top1_energy_mean"]
        > distributed_early["e_top1_energy_mean"] + 0.02,
        "reweight_reduces_final_e_concentration": reweighted_final["e_top1_energy_mean"]
        < baseline_final["e_top1_energy_mean"],
        "reweight_learns_k": reweighted_final.get("shared_target_accuracy_mean", 0.0) >= 0.99,
    }
    status = "pass" if all(checks.values()) else "partial" if any(checks.values()) else "fail"
    return {"status": status, "checks": checks, "final_rows": {
        "baseline": baseline_final,
        "reweighted": reweighted_final,
    }}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(path: Path, agg: List[Dict[str, object]]) -> None:
    conditions = sorted(set(str(x["condition"]) for x in agg))
    panels = [
        ("g_output_sigma1_over_sigma2_mean", "Output-gradient σ1/σ2"),
        ("g_input_sigma1_over_sigma2_mean", "Input-gradient σ1/σ2"),
        ("k_continuation_hidden_top1_energy_mean", "K-continuation hidden top-1 energy"),
        ("k_context_cosine_mean", "cos(E[K], mean h | target=K)"),
        ("weighted_loss_mean", "Weighted CE"),
        ("shared_target_accuracy_mean", "Shared-K target accuracy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (metric, title) in zip(axes.flat, panels):
        for condition in conditions:
            items = [x for x in agg if x["condition"] == condition and metric in x]
            if not items:
                continue
            xs = np.array([int(x["step"]) for x in items])
            ys = np.array([float(x[metric]) for x in items])
            std_key = metric.replace("_mean", "_std")
            std = np.array([float(x.get(std_key, 0.0)) for x in items])
            ax.plot(xs, ys, label=condition, linewidth=2)
            ax.fill_between(xs, ys - std, ys + std, alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 1: shared-target/input nucleation; bands show ±1 seed std", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument("--reweight_alpha", type=float, default=0.5)
    cfg = Config(**vars(parser.parse_args()))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in cfg.seeds.split(",") if x.strip()]
    conditions = [
        "distributed_high_no_k_input",
        "shared_high_no_k_input",
        "shared_high_k_input",
        "shared_high_k_input_reweight",
        "shared_lowdiv_no_k_input",
    ]
    rows: List[Dict[str, object]] = []
    for condition in conditions:
        for seed in seeds:
            print(f"running condition={condition} seed={seed}", flush=True)
            rows.extend(run_one(cfg, condition, seed))
    agg = aggregate(rows)
    summary = summarize(cfg, agg)
    write_csv(outdir / "history.csv", rows)
    write_csv(outdir / "aggregate.csv", agg)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_metrics(outdir / "metrics.png", agg)
    print(json.dumps(summary["checks"], indent=2), flush=True)
    print(f"status={summary['status']}", flush=True)


if __name__ == "__main__":
    main()
