#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    experiment_name: str
    group_probs: str
    outdir: str
    dim: int = 3
    init_layout: str = "spread"
    seed: int = 0
    steps: int = 2000
    lr: float = 3e-2
    record_every: int = 20
    theta_deg: float = 12.0
    init_noise: float = 0.0
    eps: float = 1e-12


class BigramLM(nn.Module):
    def __init__(self, e0: torch.Tensor):
        super().__init__()
        self.E = nn.Parameter(e0.clone())
        self.W = nn.Parameter(e0.clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.E[input_ids] @ self.W.T


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_group_probs(spec: str) -> List[Tuple[str, float]]:
    pairs = []
    for item in spec.split(","):
        if not item.strip():
            continue
        name, prob = item.split(":", 1)
        pairs.append((name.strip(), float(prob)))
    total = sum(prob for _, prob in pairs)
    if len(pairs) < 2 or total <= 0:
        raise ValueError("--group_probs must contain at least two positive groups")
    return [(name, prob / total) for name, prob in pairs]


def group_prefix(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return alphabet[index] if index < len(alphabet) else f"G{index}"


def build_data(group_probs: Sequence[Tuple[str, float]]):
    vocab, groups, inputs, targets, weights, group_names = [], {}, [], [], [], []
    for group_idx, (group, prob) in enumerate(group_probs):
        prefix = group_prefix(group_idx)
        toks = [f"{prefix}{i}" for i in range(3)]
        base = len(vocab)
        vocab.extend(toks)
        groups[group] = toks
        for i in range(3):
            inputs.append(base + i)
            targets.append(base + ((i + 1) % 3))
            weights.append(prob / 3.0)
            group_names.append(group)
    return (
        vocab,
        groups,
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
        group_names,
    )


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        out = np.zeros_like(v)
        out[0] = 1.0
        return out
    return v / norm


def rotate_in_plane(center: np.ndarray, angle_deg: float) -> np.ndarray:
    dim = center.shape[0]
    u = normalize(center)
    candidate = np.zeros(dim, dtype=np.float32)
    if dim == 2:
        candidate[:] = np.array([-u[1], u[0]], dtype=np.float32)
    else:
        candidate[1 if abs(u[1]) < 0.9 else 0] = 1.0
        candidate = candidate - float(np.dot(candidate, u)) * u
    v = normalize(candidate)
    th = math.radians(angle_deg)
    return math.cos(th) * u + math.sin(th) * v


def default_tail_centers(num_tail: int, dim: int, layout: str) -> List[np.ndarray]:
    if layout == "packed_common":
        return [np.eye(dim, dtype=np.float32)[0] for _ in range(num_tail)]
    if layout == "packed_negative_common":
        v = -np.eye(dim, dtype=np.float32)[0]
        return [v.copy() for _ in range(num_tail)]
    if layout != "spread":
        raise ValueError(f"Unknown init_layout: {layout}")
    if dim == 2:
        base = [
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
        ]
    elif dim == 3:
        base = [
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]
    else:
        raise ValueError("This experiment intentionally supports only dim=2 or dim=3")
    if num_tail > len(base):
        raise ValueError(f"dim={dim} default spread supports at most {len(base)} tail groups")
    return base[:num_tail]


def build_initial_geometry(vocab, groups, group_probs, cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    centers = {group_probs[0][0]: np.eye(cfg.dim, dtype=np.float32)[0]}
    tail_centers = default_tail_centers(len(group_probs) - 1, cfg.dim, cfg.init_layout)
    for (group, _), center in zip(group_probs[1:], tail_centers):
        centers[group] = normalize(center)

    tok_to_vec = {}
    for group, toks in groups.items():
        center = centers[group]
        for tok, angle in zip(toks, [0.0, cfg.theta_deg, -cfg.theta_deg]):
            vec = rotate_in_plane(center, angle)
            if cfg.init_noise > 0:
                vec = vec + rng.normal(0.0, cfg.init_noise, size=cfg.dim).astype(np.float32)
            tok_to_vec[tok] = vec.astype(np.float32)
    e0 = torch.tensor(np.stack([tok_to_vec[tok] for tok in vocab]), dtype=torch.float32)
    return e0, {k: v.tolist() for k, v in centers.items()}


def compute_metrics(model, inputs, targets, weights, group_names, active_groups):
    logits = model(inputs)
    losses = F.cross_entropy(logits, targets, reduction="none")
    loss = (losses * weights).sum() / weights.sum()
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float()
    top2 = torch.topk(logits, k=2, dim=-1)
    target_logits = logits.gather(1, targets[:, None]).squeeze(1)
    best = top2.values[:, 0]
    second = top2.values[:, 1]
    best_id = top2.indices[:, 0]
    margin = torch.where(best_id == targets, target_logits - second, target_logits - best)

    metrics = {
        "loss": float(loss.item()),
        "acc_all_weighted": float((correct * weights).sum().item() / weights.sum().item()),
        "margin_all_weighted": float((margin * weights).sum().item() / weights.sum().item()),
    }
    for group in active_groups:
        idx = torch.tensor([i for i, g in enumerate(group_names) if g == group], dtype=torch.long)
        metrics[f"loss_{group}"] = float(losses[idx].mean().item())
        metrics[f"acc_{group}"] = float(correct[idx].mean().item())
        metrics[f"margin_{group}"] = float(margin[idx].mean().item())
    return loss, metrics


def flatten_grads(model: nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(p.grad.detach().reshape(-1).clone())
    return torch.cat(parts)


def effective_rank_from_vectors(vectors: Sequence[np.ndarray], weights: Sequence[float], eps: float) -> float:
    if not vectors:
        return 0.0
    mat = np.stack([w * v for w, v in zip(weights, vectors)], axis=0)
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    eig = s ** 2
    total = float(eig.sum())
    if total <= eps:
        return 0.0
    return float((total ** 2) / max(float((eig ** 2).sum()), eps))


def pairwise_cosine(vectors: Dict[str, np.ndarray], eps: float):
    names = list(vectors)
    out = {}
    for a in names:
        out[a] = {}
        for b in names:
            va, vb = vectors[a], vectors[b]
            out[a][b] = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), eps))
    return out


def group_centroids(E: np.ndarray, vocab, groups):
    return {group: E[[vocab.index(tok) for tok in toks]].mean(axis=0) for group, toks in groups.items()}


def residual_tail_effective_rank(E: np.ndarray, vocab, groups, active_groups, eps: float) -> float:
    centroids = group_centroids(E, vocab, groups)
    common = centroids[active_groups[0]]
    common_norm = np.linalg.norm(common)
    if common_norm <= eps:
        residuals = [centroids[g] for g in active_groups[1:]]
    else:
        u = common / common_norm
        residuals = [centroids[g] - np.dot(centroids[g], u) * u for g in active_groups[1:]]
    if len(residuals) < 2:
        return 0.0
    mat = np.stack(residuals, axis=0)
    mat = mat - mat.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    eig = s ** 2
    total = float(eig.sum())
    if total <= eps:
        return 0.0
    return float((total ** 2) / max(float((eig ** 2).sum()), eps))


def analyze_group_gradients(model, inputs, targets, group_names, active_groups, probs, eps: float):
    q: Dict[str, np.ndarray] = {}
    norms = {}
    for group in active_groups:
        idx = torch.tensor([i for i, g in enumerate(group_names) if g == group], dtype=torch.long)
        model.zero_grad(set_to_none=True)
        logits = model(inputs[idx])
        loss = F.cross_entropy(logits, targets[idx], reduction="mean")
        loss.backward()
        vec = flatten_grads(model).cpu().numpy().astype(np.float64)
        q[group] = vec
        norms[group] = float(np.linalg.norm(vec))
    model.zero_grad(set_to_none=True)

    cos = pairwise_cosine(q, eps)
    all_grad_eff_rank = effective_rank_from_vectors([q[g] for g in active_groups], [probs[g] for g in active_groups], eps)
    tail_groups = active_groups[1:]
    tail_grad_eff_rank = effective_rank_from_vectors([q[g] for g in tail_groups], [probs[g] for g in tail_groups], eps)

    tail_sir = {}
    tail_interference = {}
    for group in tail_groups:
        unit = q[group] / max(np.linalg.norm(q[group]), eps)
        signal = probs[group] * norms[group]
        common_interf = abs(probs[active_groups[0]] * float(np.dot(q[active_groups[0]], unit)))
        other_tail_interf = 0.0
        signed_other_tail = 0.0
        for other in tail_groups:
            if other == group:
                continue
            val = probs[other] * float(np.dot(q[other], unit))
            signed_other_tail += val
            other_tail_interf += abs(val)
        total_interf = common_interf + other_tail_interf
        tail_sir[group] = float(signal / (total_interf + eps))
        tail_interference[group] = {
            "signal": float(signal),
            "common_abs": float(common_interf),
            "other_tail_abs": float(other_tail_interf),
            "other_tail_signed": float(signed_other_tail),
            "total_abs": float(total_interf),
        }

    tail_cos_vals = []
    for i, a in enumerate(tail_groups):
        for b in tail_groups[i + 1:]:
            tail_cos_vals.append(cos[a][b])
    common_tail_cos_vals = [cos[active_groups[0]][g] for g in tail_groups]
    return {
        "grad_norms": norms,
        "grad_cosine": cos,
        "all_grad_eff_rank": all_grad_eff_rank,
        "tail_grad_eff_rank": tail_grad_eff_rank,
        "tail_grad_cosine_mean": float(np.mean(tail_cos_vals)) if tail_cos_vals else 0.0,
        "common_tail_grad_cosine_mean": float(np.mean(common_tail_cos_vals)) if common_tail_cos_vals else 0.0,
        "tail_sir": tail_sir,
        "tail_interference": tail_interference,
    }


def train_and_analyze(cfg: Config):
    set_seed(cfg.seed)
    group_probs = parse_group_probs(cfg.group_probs)
    probs = dict(group_probs)
    vocab, groups, inputs, targets, weights, group_names = build_data(group_probs)
    e0, centers = build_initial_geometry(vocab, groups, group_probs, cfg)
    model = BigramLM(e0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    active_groups = [g for g, _ in group_probs]
    history = []

    for step in range(cfg.steps + 1):
        if step > 0:
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_metrics(model, inputs, targets, weights, group_names, active_groups)
            loss.backward()
            opt.step()
        if step % cfg.record_every == 0 or step == cfg.steps:
            _, metrics = compute_metrics(model, inputs, targets, weights, group_names, active_groups)
            grad_stats = analyze_group_gradients(model, inputs, targets, group_names, active_groups, probs, cfg.eps)
            E = model.E.detach().cpu().numpy().copy()
            E_centered = E - E.mean(axis=0, keepdims=True)
            _, singular_values, _ = np.linalg.svd(E_centered, full_matrices=False)
            rep_tail_eff = residual_tail_effective_rank(E, vocab, groups, active_groups, cfg.eps)
            history.append(
                {
                    "step": step,
                    "metrics": metrics,
                    "singular_values": singular_values.tolist(),
                    "tail_rep_residual_eff_rank": rep_tail_eff,
                    **grad_stats,
                }
            )

    return {
        "config": asdict(cfg),
        "init_centers": centers,
        "groups": active_groups,
        "group_probs": probs,
        "history": history,
    }


def plot_summary(result, outdir: str):
    hist = result["history"]
    groups = result["groups"]
    tail_groups = groups[1:]
    steps = [h["step"] for h in hist]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(steps, [h["metrics"]["loss"] for h in hist], label="weighted")
    for g in groups:
        axes[0, 0].plot(steps, [h["metrics"][f"loss_{g}"] for h in hist], "--", label=g)
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(steps, [h["tail_grad_eff_rank"] for h in hist], label="tail gradient d_eff")
    axes[0, 1].plot(steps, [h["all_grad_eff_rank"] for h in hist], label="all gradient d_eff")
    axes[0, 1].plot(steps, [h["tail_rep_residual_eff_rank"] for h in hist], label="tail representation residual d_eff")
    axes[0, 1].set_title("Effective Rank")
    axes[0, 1].legend(fontsize=8)

    for g in tail_groups:
        axes[1, 0].plot(steps, [h["tail_sir"][g] for h in hist], label=g)
    axes[1, 0].set_title("Tail Gradient SIR")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(steps, [h["tail_grad_cosine_mean"] for h in hist], label="tail-tail grad cosine")
    axes[1, 1].plot(steps, [h["common_tail_grad_cosine_mean"] for h in hist], label="common-tail grad cosine")
    axes[1, 1].set_title("Gradient Cosine")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "gradient_interference_summary.png"), dpi=180)
    plt.close(fig)


def write_compact_csv(result, outdir: str):
    groups = result["groups"]
    tail_groups = groups[1:]
    path = os.path.join(outdir, "gradient_interference_timeseries.csv")
    fields = [
        "step",
        "loss",
        "tail_grad_eff_rank",
        "all_grad_eff_rank",
        "tail_rep_residual_eff_rank",
        "tail_grad_cosine_mean",
        "common_tail_grad_cosine_mean",
    ]
    for g in tail_groups:
        fields.extend([f"loss_{g}", f"margin_{g}", f"sir_{g}"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for h in result["history"]:
            row = [
                h["step"],
                h["metrics"]["loss"],
                h["tail_grad_eff_rank"],
                h["all_grad_eff_rank"],
                h["tail_rep_residual_eff_rank"],
                h["tail_grad_cosine_mean"],
                h["common_tail_grad_cosine_mean"],
            ]
            for g in tail_groups:
                row.extend([h["metrics"][f"loss_{g}"], h["metrics"][f"margin_{g}"], h["tail_sir"][g]])
            f.write(",".join(str(x) for x in row) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--group_probs", required=True)
    p.add_argument("--outdir", default="fdong_embedding_dim/outputs/toy_gradient_interference")
    p.add_argument("--dim", type=int, choices=[2, 3], default=3)
    p.add_argument("--init_layout", choices=["spread", "packed_common", "packed_negative_common"], default="spread")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--record_every", type=int, default=20)
    p.add_argument("--theta_deg", type=float, default=12.0)
    p.add_argument("--init_noise", type=float, default=0.0)
    return Config(**vars(p.parse_args()))


def main():
    cfg = parse_args()
    result = train_and_analyze(cfg)
    outdir = os.path.join(cfg.outdir, cfg.experiment_name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_compact_csv(result, outdir)
    plot_summary(result, outdir)
    final = result["history"][-1]
    compact = {
        "experiment_name": cfg.experiment_name,
        "final_loss": final["metrics"]["loss"],
        "final_tail_grad_eff_rank": final["tail_grad_eff_rank"],
        "final_tail_rep_residual_eff_rank": final["tail_rep_residual_eff_rank"],
        "final_tail_grad_cosine_mean": final["tail_grad_cosine_mean"],
        "final_common_tail_grad_cosine_mean": final["common_tail_grad_cosine_mean"],
        "final_tail_sir": final["tail_sir"],
    }
    with open(os.path.join(outdir, "final_compact.json"), "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)
    print(json.dumps(compact, indent=2))
    print(f"Done: {cfg.experiment_name} -> {outdir}")


if __name__ == "__main__":
    main()
