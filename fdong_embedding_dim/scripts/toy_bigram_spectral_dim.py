#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from collections import Counter
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
    seed: int = 0
    steps: int = 2000
    lr: float = 3e-2
    record_every: int = 20
    theta_deg: float = 12.0
    tail_centers: str = ""
    init_noise: float = 0.0


class BigramLM(nn.Module):
    def __init__(self, e0: torch.Tensor, w0: torch.Tensor):
        super().__init__()
        self.E = nn.Parameter(e0.clone())
        self.W = nn.Parameter(w0.clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.E[input_ids] @ self.W.T


def set_seed(seed: int):
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
    if total <= 0 or len(pairs) < 2:
        raise ValueError("--group_probs must contain at least two positive groups")
    return [(name, prob / total) for name, prob in pairs]


def parse_centers(spec: str, dim: int) -> List[np.ndarray]:
    centers = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        vals = [float(x) for x in item.split(",")]
        if len(vals) != dim:
            raise ValueError(f"center {item} has dim {len(vals)}, expected {dim}")
        centers.append(np.array(vals, dtype=np.float32))
    return centers


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


def rotate_in_first_available_plane(center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Small within-group spread while preserving interpretability in 3D."""
    dim = center.shape[0]
    u = normalize(center)
    candidate = np.zeros(dim, dtype=np.float32)
    candidate[1 if abs(u[1]) < 0.9 and dim > 1 else 0] = 1.0
    v = candidate - float(np.dot(candidate, u)) * u
    v = normalize(v)
    th = math.radians(angle_deg)
    return math.cos(th) * u + math.sin(th) * v


def default_tail_centers(num_tail: int, dim: int) -> List[np.ndarray]:
    if dim != 3:
        raise ValueError("This script is intended for dim=3 visualization.")
    base = [
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    ]
    if num_tail > len(base):
        raise ValueError("Default 3D centers support at most four tail groups")
    return base[:num_tail]


def build_initial_geometry(vocab, groups, group_probs, cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    centers = {group_probs[0][0]: np.eye(cfg.dim, dtype=np.float32)[0]}
    tail_centers = parse_centers(cfg.tail_centers, cfg.dim) if cfg.tail_centers else default_tail_centers(len(group_probs) - 1, cfg.dim)
    if len(tail_centers) != len(group_probs) - 1:
        raise ValueError("--tail_centers count must match number of tail groups")
    for (group, _), center in zip(group_probs[1:], tail_centers):
        centers[group] = normalize(center)

    tok_to_vec = {}
    for group, toks in groups.items():
        c = centers[group]
        for tok, angle in zip(toks, [0.0, cfg.theta_deg, -cfg.theta_deg]):
            vec = rotate_in_first_available_plane(c, angle)
            if cfg.init_noise > 0:
                vec = vec + rng.normal(0.0, cfg.init_noise, size=cfg.dim).astype(np.float32)
            tok_to_vec[tok] = vec.astype(np.float32)
    e0 = torch.tensor(np.stack([tok_to_vec[tok] for tok in vocab]), dtype=torch.float32)
    return e0, e0.clone(), {k: v.tolist() for k, v in centers.items()}


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
    }
    for group in active_groups:
        idx = torch.tensor([i for i, g in enumerate(group_names) if g == group], dtype=torch.long)
        metrics[f"loss_{group}"] = float(losses[idx].mean().item())
        metrics[f"acc_{group}"] = float(correct[idx].mean().item())
        metrics[f"margin_{group}"] = float(margin[idx].mean().item())
    return loss, metrics


def train(cfg: Config):
    set_seed(cfg.seed)
    group_probs = parse_group_probs(cfg.group_probs)
    vocab, groups, inputs, targets, weights, group_names = build_data(group_probs)
    e0, w0, centers = build_initial_geometry(vocab, groups, group_probs, cfg)
    model = BigramLM(e0, w0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    active = [g for g, _ in group_probs]
    history = []
    for step in range(cfg.steps + 1):
        if step > 0:
            opt.zero_grad()
            loss, _ = compute_metrics(model, inputs, targets, weights, group_names, active)
            loss.backward()
            opt.step()
        if step % cfg.record_every == 0 or step == cfg.steps:
            _, metrics = compute_metrics(model, inputs, targets, weights, group_names, active)
            E = model.E.detach().cpu().numpy().copy()
            W = model.W.detach().cpu().numpy().copy()
            _, S, Vt = np.linalg.svd(E - E.mean(axis=0, keepdims=True), full_matrices=False)
            history.append({"step": step, "E": E, "W": W, "S": S, "V": Vt.T, "metrics": metrics})
    return model, history, vocab, groups, active, centers, group_probs


def group_centroids(E: np.ndarray, vocab, groups):
    return {group: E[[vocab.index(tok) for tok in toks]].mean(axis=0) for group, toks in groups.items()}


def residual_basis(common_vec: np.ndarray):
    u = normalize(common_vec)
    _, _, vh = np.linalg.svd(u.reshape(1, -1), full_matrices=True)
    return u, vh[1:].T


def plot_3d(history, vocab, groups, active_groups, outpath):
    colors = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    final_E = history[-1]["E"]
    lim = max(1.0, float(np.abs(np.concatenate([snap["E"] for snap in history], axis=0)).max())) * 1.15
    tok_group = {tok: group for group, toks in groups.items() for tok in toks}
    for i, tok in enumerate(vocab):
        traj = np.stack([snap["E"][i] for snap in history])
        gi = active_groups.index(tok_group[tok])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors(gi), alpha=0.7)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors(gi), marker="x", s=55)
        ax.text(traj[-1, 0], traj[-1, 1], traj[-1, 2], tok, fontsize=8)
    centroids = group_centroids(final_E, vocab, groups)
    for group, c in centroids.items():
        gi = active_groups.index(group)
        ax.scatter(c[0], c[1], c[2], color=colors(gi), s=150, edgecolors="black", label=group)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_residual_plane(history, vocab, groups, active_groups, outpath):
    final_centroids = group_centroids(history[-1]["E"], vocab, groups)
    common_group = active_groups[0]
    common_vec = final_centroids[common_group]
    _u, basis = residual_basis(common_vec)
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 8))
    tok_group = {tok: group for group, toks in groups.items() for tok in toks}
    for i, tok in enumerate(vocab):
        group = tok_group[tok]
        traj = np.stack([snap["E"][i] for snap in history]) @ basis
        gi = active_groups.index(group)
        ax.plot(traj[:, 0], traj[:, 1], color=colors(gi), alpha=0.7)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors(gi), marker="x", s=55)
        ax.text(traj[-1, 0], traj[-1, 1], tok, fontsize=8)
    for group, c in final_centroids.items():
        point = c @ basis
        gi = active_groups.index(group)
        ax.scatter(point[0], point[1], color=colors(gi), s=150, edgecolors="black", label=group)
    ax.axhline(0, color="lightgray"); ax.axvline(0, color="lightgray")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Residual plane after removing final common direction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_metrics(history, active_groups, outpath):
    steps = [snap["step"] for snap in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(steps, [snap["metrics"]["loss"] for snap in history], label="weighted")
    for g in active_groups:
        axes[0].plot(steps, [snap["metrics"][f"loss_{g}"] for snap in history], "--", label=g)
        axes[1].plot(steps, [snap["metrics"][f"margin_{g}"] for snap in history], "--", label=g)
    axes[0].set_title("Loss")
    axes[1].set_title("Margin")
    axes[0].legend(); axes[1].legend()
    axes[0].grid(alpha=0.25); axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def summarize(history, vocab, groups, active_groups, centers, group_probs):
    final = history[-1]
    E = final["E"]
    centroids = group_centroids(E, vocab, groups)
    common_group = active_groups[0]
    common_vec = centroids[common_group]
    _u, basis = residual_basis(common_vec)
    summary = {
        "final_step": final["step"],
        "final_metrics": final["metrics"],
        "singular_values": final["S"].tolist(),
        "init_centers": centers,
        "groups": {},
        "pairwise_centroid_cosine": {},
    }
    probs = dict(group_probs)
    for group, c in centroids.items():
        cn = np.linalg.norm(c)
        residual = c @ basis
        summary["groups"][group] = {
            "prob": probs[group],
            "centroid": c.tolist(),
            "centroid_norm": float(cn),
            "residual_plane_xy": residual.tolist(),
            "cos_to_common": float(np.dot(c, common_vec) / max(cn * np.linalg.norm(common_vec), 1e-12)),
            "loss": final["metrics"][f"loss_{group}"],
            "accuracy": final["metrics"][f"acc_{group}"],
            "margin": final["metrics"][f"margin_{group}"],
        }
    for g1, c1 in centroids.items():
        summary["pairwise_centroid_cosine"][g1] = {}
        for g2, c2 in centroids.items():
            summary["pairwise_centroid_cosine"][g1][g2] = float(
                np.dot(c1, c2) / max(np.linalg.norm(c1) * np.linalg.norm(c2), 1e-12)
            )
    tail_res = np.stack([summary["groups"][g]["residual_plane_xy"] for g in active_groups[1:]])
    if len(tail_res) >= 2:
        _, s, _ = np.linalg.svd(tail_res - tail_res.mean(axis=0, keepdims=True), full_matrices=False)
        summary["tail_residual_singular_values"] = s.tolist()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--group_probs", required=True)
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "toy3d"))
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--record_every", type=int, default=20)
    p.add_argument("--theta_deg", type=float, default=12.0)
    p.add_argument("--tail_centers", default="")
    p.add_argument("--init_noise", type=float, default=0.0)
    return Config(**vars(p.parse_args()))


def main():
    cfg = parse_args()
    model, history, vocab, groups, active_groups, centers, group_probs = train(cfg)
    outdir = os.path.join(cfg.outdir, cfg.experiment_name)
    os.makedirs(outdir, exist_ok=True)
    plot_3d(history, vocab, groups, active_groups, os.path.join(outdir, "01_E_3d_trajectories.png"))
    plot_residual_plane(history, vocab, groups, active_groups, os.path.join(outdir, "02_E_residual_plane.png"))
    plot_metrics(history, active_groups, os.path.join(outdir, "03_training_metrics.png"))
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summarize(history, vocab, groups, active_groups, centers, group_probs), f, indent=2)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Done: {cfg.experiment_name} -> {outdir}")


if __name__ == "__main__":
    main()
