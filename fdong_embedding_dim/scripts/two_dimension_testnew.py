#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    matrix_to_visualize: str = "E"
    seed: int = 0
    steps: int = 2000
    lr: float = 3e-2
    weight_decay: float = 0.0
    record_every: int = 20
    theta_deg: float = 12.0
    init_layout: str = "spread"
    tail_centers: str = ""
    init_noise: float = 0.0
    num_snapshots: int = 6
    use_real_singular_value_length: bool = True
    singular_vector_scale: Optional[float] = None


class LowDimBigramLM(nn.Module):
    def __init__(self, E0: torch.Tensor, W0: torch.Tensor):
        super().__init__()
        self.E = nn.Parameter(E0.clone())
        self.W = nn.Parameter(W0.clone())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.E[input_ids]
        return h @ self.W.T


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_group_probs(spec: str) -> List[Tuple[str, float]]:
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad group prob item: {item}")
        name, value = item.split(":", 1)
        pairs.append((name.strip(), float(value)))
    if len(pairs) < 2:
        raise ValueError("--group_probs must contain at least two groups")
    total = sum(v for _, v in pairs)
    if total <= 0:
        raise ValueError("--group_probs must have positive total probability")
    return [(name, prob / total) for name, prob in pairs]


def parse_tail_centers(spec: str) -> List[Tuple[float, float]]:
    if not spec:
        return []
    centers = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        x_str, y_str = item.split(",", 1)
        centers.append((float(x_str), float(y_str)))
    return centers


def group_prefix(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return alphabet[index]
    return f"G{index}"


def build_vocab_and_transitions(group_probs: Sequence[Tuple[str, float]]):
    vocab: List[str] = []
    inputs: List[int] = []
    targets: List[int] = []
    weights: List[float] = []
    group_names: List[str] = []
    token_to_group: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}

    for group_idx, (group_name, prob) in enumerate(group_probs):
        prefix = group_prefix(group_idx)
        toks = [f"{prefix}{i}" for i in range(3)]
        base = len(vocab)
        vocab.extend(toks)
        groups[group_name] = toks
        for tok in toks:
            token_to_group[tok] = group_name
        for local_i in range(3):
            inputs.append(base + local_i)
            targets.append(base + ((local_i + 1) % 3))
            weights.append(prob / 3.0)
            group_names.append(group_name)

    return (
        vocab,
        groups,
        token_to_group,
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
        group_names,
    )


def unit(v: Tuple[float, float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float32)
    return arr / norm


def rotate(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    th = math.radians(angle_deg)
    rot = np.array(
        [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]],
        dtype=np.float32,
    )
    return rot @ vec


def default_tail_centers(layout: str, num_tail: int, rng: np.random.Generator):
    if layout == "spread":
        base = [(0.0, 1.0), (0.0, -1.0), (-1.0, 0.0), (-0.707, 0.707)]
        if num_tail <= len(base):
            return base[:num_tail]
        angles = np.linspace(90, 450, num_tail, endpoint=False)
        return [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angles]
    if layout == "packed_x_pos":
        return [(1.0, 0.0) for _ in range(num_tail)]
    if layout == "packed_x_neg":
        return [(-1.0, 0.0) for _ in range(num_tail)]
    if layout == "packed_y_pos":
        return [(0.0, 1.0) for _ in range(num_tail)]
    if layout == "packed_y_neg":
        return [(0.0, -1.0) for _ in range(num_tail)]
    if layout == "random":
        angles = rng.uniform(0, 2 * math.pi, size=num_tail)
        return [(math.cos(a), math.sin(a)) for a in angles]
    raise ValueError(f"Unknown init_layout: {layout}")


def build_initial_geometry(
    vocab: Sequence[str],
    groups: Dict[str, List[str]],
    group_probs: Sequence[Tuple[str, float]],
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[float, float]]]:
    rng = np.random.default_rng(cfg.seed)
    tail_override = parse_tail_centers(cfg.tail_centers)
    num_tail = len(group_probs) - 1
    if tail_override:
        if len(tail_override) != num_tail:
            raise ValueError(
                f"--tail_centers has {len(tail_override)} centers but expected {num_tail}"
            )
        tail_centers = tail_override
    else:
        tail_centers = default_tail_centers(cfg.init_layout, num_tail, rng)

    centers: Dict[str, Tuple[float, float]] = {group_probs[0][0]: (1.0, 0.0)}
    for (group_name, _), center in zip(group_probs[1:], tail_centers):
        centers[group_name] = center

    tok_to_vec: Dict[str, np.ndarray] = {}
    for group_name, toks in groups.items():
        center_vec = unit(centers[group_name])
        for tok, angle_offset in zip(toks, [0.0, cfg.theta_deg, -cfg.theta_deg]):
            vec = rotate(center_vec, angle_offset)
            if cfg.init_noise > 0:
                vec = vec + rng.normal(0.0, cfg.init_noise, size=2).astype(np.float32)
            tok_to_vec[tok] = vec.astype(np.float32)

    E0 = torch.tensor(np.stack([tok_to_vec[tok] for tok in vocab]), dtype=torch.float32)
    W0 = E0.clone()
    return E0, W0, centers


def compute_loss_and_metrics(
    model: LowDimBigramLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    group_names: Sequence[str],
    active_groups: Sequence[str],
):
    logits = model(inputs)
    per_example_loss = F.cross_entropy(logits, targets, reduction="none")
    weighted_loss = (per_example_loss * weights).sum() / weights.sum()

    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float()

    margins = []
    for i, target in enumerate(targets.tolist()):
        row = logits[i].detach().clone()
        target_logit = row[target].item()
        row[target] = -float("inf")
        margins.append(target_logit - row.max().item())
    margins_t = torch.tensor(margins, dtype=torch.float32)

    metrics = {
        "loss": float(weighted_loss.item()),
        "acc_all_weighted": float((correct * weights).sum().item() / weights.sum().item()),
        "margin_all_weighted": float((margins_t * weights).sum().item() / weights.sum().item()),
    }

    for group_name in active_groups:
        idx = torch.tensor(
            [i for i, name in enumerate(group_names) if name == group_name],
            dtype=torch.long,
        )
        metrics[f"loss_{group_name}"] = float(per_example_loss[idx].mean().item())
        metrics[f"acc_{group_name}"] = float(correct[idx].mean().item())
        metrics[f"margin_{group_name}"] = float(margins_t[idx].mean().item())

    return weighted_loss, metrics


def aligned_svd(M: np.ndarray, prev_V: Optional[np.ndarray] = None):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T
    if prev_V is not None:
        for k in range(V.shape[1]):
            if np.dot(V[:, k], prev_V[:, k]) < 0:
                V[:, k] *= -1.0
                U[:, k] *= -1.0
    return U, S, V


def record_snapshot(history, step, model, metrics, prev_V_E=None, prev_V_W=None):
    E = model.E.detach().cpu().numpy().copy()
    W = model.W.detach().cpu().numpy().copy()
    U_E, S_E, V_E = aligned_svd(E, prev_V_E)
    U_W, S_W, V_W = aligned_svd(W, prev_V_W)
    history.append(
        {
            "step": step,
            "metrics": metrics,
            "E": E,
            "E_U": U_E,
            "E_S": S_E,
            "E_V": V_E,
            "W": W,
            "W_U": U_W,
            "W_S": S_W,
            "W_V": V_W,
        }
    )
    return V_E, V_W


def train_and_record(cfg: Config):
    set_seed(cfg.seed)
    group_probs = parse_group_probs(cfg.group_probs)
    (
        vocab,
        groups,
        token_to_group,
        inputs,
        targets,
        weights,
        group_names,
    ) = build_vocab_and_transitions(group_probs)
    E0, W0, centers = build_initial_geometry(vocab, groups, group_probs, cfg)
    model = LowDimBigramLM(E0, W0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    prev_V_E, prev_V_W = None, None
    active_groups = [name for name, _ in group_probs]
    _, metrics0 = compute_loss_and_metrics(model, inputs, targets, weights, group_names, active_groups)
    prev_V_E, prev_V_W = record_snapshot(history, 0, model, metrics0, prev_V_E, prev_V_W)

    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad()
        loss, _ = compute_loss_and_metrics(model, inputs, targets, weights, group_names, active_groups)
        loss.backward()
        optimizer.step()
        if step % cfg.record_every == 0 or step == cfg.steps:
            _, metrics_now = compute_loss_and_metrics(
                model, inputs, targets, weights, group_names, active_groups
            )
            prev_V_E, prev_V_W = record_snapshot(
                history, step, model, metrics_now, prev_V_E, prev_V_W
            )

    return model, history, vocab, groups, token_to_group, active_groups, centers, group_probs


def color_map(active_groups: Sequence[str]) -> Dict[str, str]:
    cmap = plt.get_cmap("tab10")
    return {group: cmap(i % 10) for i, group in enumerate(active_groups)}


def auto_singular_vector_scale(history, matrix_key="E"):
    max_token_norm = 1e-12
    max_sigma = 1e-12
    for snap in history:
        M = snap[matrix_key]
        S = snap[f"{matrix_key}_S"]
        max_token_norm = max(max_token_norm, float(np.linalg.norm(M, axis=1).max()))
        max_sigma = max(max_sigma, float(S[0]))
    return 1.05 * max_token_norm / max_sigma


def compute_global_plot_limit(histories, matrix_key, singular_vector_scales):
    max_abs = 1.0
    for history, sv_scale in zip(histories, singular_vector_scales):
        for snap in history:
            M = snap[matrix_key]
            S = snap[f"{matrix_key}_S"]
            V = snap[f"{matrix_key}_V"]
            max_abs = max(max_abs, float(np.abs(M).max()))
            for k in range(min(2, V.shape[1])):
                vec = sv_scale * S[k] * V[:, k]
                max_abs = max(max_abs, float(np.abs(vec).max()))
    return max_abs * 1.20


def setup_axis(ax, lim):
    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", linestyle="--", linewidth=1.2)
    ax.add_artist(circle)
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)


def draw_token_vectors(ax, M, vocab, token_to_group, group_colors):
    for i, tok in enumerate(vocab):
        x_coord, y_coord = M[i]
        group = token_to_group[tok]
        color = group_colors[group]
        ax.plot([0, x_coord], [0, y_coord], color=color, alpha=0.22, linewidth=1.2)
        ax.scatter([x_coord], [y_coord], s=72, color=color, alpha=0.92,
                   edgecolors="black", linewidth=0.55, zorder=4)
        ax.text(x_coord + 0.04, y_coord + 0.04, tok, fontsize=9, color=color)


def draw_singular_vectors(ax, S, V, singular_vector_scale):
    colors = ["black", "purple"]
    for k in range(min(2, V.shape[1])):
        vx, vy = V[:, k]
        L = singular_vector_scale * S[k]
        ax.arrow(0, 0, L * vx, L * vy, head_width=0.045, head_length=0.070,
                 length_includes_head=True, color=colors[k], linewidth=2.8, alpha=0.96)
        ax.plot([0, -L * vx], [0, -L * vy], linestyle="--", color=colors[k],
                linewidth=2.0, alpha=0.65)
    ax.text(
        0.02,
        0.98,
        f"v1 length = sigma1 = {S[0]:.3f}\n"
        f"v2 length = sigma2 = {S[1]:.3f}\n"
        f"scale={singular_vector_scale:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.78, edgecolor="lightgray"),
    )


def plot_snapshots(history, vocab, token_to_group, group_colors, cfg, save_path, sv_scale):
    idxs = np.linspace(0, len(history) - 1, cfg.num_snapshots, dtype=int)
    snaps = [history[i] for i in idxs]
    lim = compute_global_plot_limit([snaps], cfg.matrix_to_visualize, [sv_scale])
    ncols = 3
    nrows = math.ceil(cfg.num_snapshots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, snap in zip(axes, snaps):
        setup_axis(ax, lim)
        M = snap[cfg.matrix_to_visualize]
        S = snap[f"{cfg.matrix_to_visualize}_S"]
        V = snap[f"{cfg.matrix_to_visualize}_V"]
        draw_token_vectors(ax, M, vocab, token_to_group, group_colors)
        draw_singular_vectors(ax, S, V, sv_scale)
        ax.set_title(
            f"{cfg.experiment_name} | {cfg.matrix_to_visualize} @ step {snap['step']}\n"
            f"sigma1/sigma2={S[0] / max(S[1], 1e-12):.3f}, loss={snap['metrics']['loss']:.3f}",
            fontsize=11,
        )
    for j in range(len(snaps), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=190)
    plt.close(fig)


def plot_trajectories(history, vocab, token_to_group, group_colors, cfg, save_path, sv_scale):
    lim = compute_global_plot_limit([history], cfg.matrix_to_visualize, [sv_scale])
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_axis(ax, lim)
    for i, tok in enumerate(vocab):
        traj = np.array([snap[cfg.matrix_to_visualize][i] for snap in history])
        group = token_to_group[tok]
        color = group_colors[group]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.7, alpha=0.75)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color=color, marker="o", s=42,
                   edgecolors="black", linewidth=0.5)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color=color, marker="x", s=70, linewidth=2.0)
        ax.text(traj[-1, 0] + 0.045, traj[-1, 1] + 0.045, tok, fontsize=9, color=color)

    v1_points, v2_points = [], []
    for snap in history:
        S = snap[f"{cfg.matrix_to_visualize}_S"]
        V = snap[f"{cfg.matrix_to_visualize}_V"]
        v1_points.append(sv_scale * S[0] * V[:, 0])
        v2_points.append(sv_scale * S[1] * V[:, 1])
    v1_points = np.array(v1_points)
    v2_points = np.array(v2_points)
    ax.plot(v1_points[:, 0], v1_points[:, 1], color="black", linewidth=3.0, label="sigma1 v1")
    ax.plot(v2_points[:, 0], v2_points[:, 1], color="purple", linewidth=3.0, label="sigma2 v2")
    ax.scatter([v1_points[-1, 0]], [v1_points[-1, 1]], color="black", marker="x", s=90)
    ax.scatter([v2_points[-1, 0]], [v2_points[-1, 1]], color="purple", marker="x", s=90)
    ax.legend(loc="lower left")
    ax.set_title(f"{cfg.experiment_name}: {cfg.matrix_to_visualize} token and singular-vector trajectories")
    fig.tight_layout()
    fig.savefig(save_path, dpi=190)
    plt.close(fig)


def plot_singular_values(history, cfg, save_path):
    steps = np.array([snap["step"] for snap in history])
    s1 = np.array([snap[f"{cfg.matrix_to_visualize}_S"][0] for snap in history])
    s2 = np.array([snap[f"{cfg.matrix_to_visualize}_S"][1] for snap in history])
    ratio = s1 / np.maximum(s2, 1e-12)
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax1.plot(steps, s1, linewidth=2.2, label="sigma1")
    ax1.plot(steps, s2, linewidth=2.2, label="sigma2")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("singular value")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(steps, ratio, linewidth=2.0, linestyle="--", label="sigma1/sigma2")
    ax2.set_ylabel("singular value ratio")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"{cfg.experiment_name}: {cfg.matrix_to_visualize} singular values")
    fig.tight_layout()
    fig.savefig(save_path, dpi=190)
    plt.close(fig)


def plot_training_metrics(history, active_groups, cfg, save_path):
    steps = np.array([snap["step"] for snap in history])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    axes[0].plot(steps, [snap["metrics"]["loss"] for snap in history],
                 linewidth=2.6, label="overall weighted loss")
    axes[1].plot(steps, [snap["metrics"]["acc_all_weighted"] for snap in history],
                 linewidth=2.6, label="overall weighted acc")
    for group in active_groups:
        axes[0].plot(steps, [snap["metrics"][f"loss_{group}"] for snap in history],
                     linewidth=2.0, linestyle="--", label=f"loss_{group}")
        axes[1].plot(steps, [snap["metrics"][f"acc_{group}"] for snap in history],
                     linewidth=2.0, linestyle="--", label=f"acc_{group}")
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("cross entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Accuracy curves")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{cfg.experiment_name}: training dynamics", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=190)
    plt.close(fig)


def summarize(history, vocab, groups, active_groups, cfg, centers, group_probs):
    snap = history[-1]
    M = snap[cfg.matrix_to_visualize]
    S = snap[f"{cfg.matrix_to_visualize}_S"]
    V = snap[f"{cfg.matrix_to_visualize}_V"]
    centroids = {}
    group_summary = {}
    for group in active_groups:
        idxs = [vocab.index(tok) for tok in groups[group]]
        centroid = M[idxs].mean(axis=0)
        centroids[group] = centroid
        norm = float(np.linalg.norm(centroid))
        group_summary[group] = {
            "prob": dict(group_probs)[group],
            "init_center": list(centers[group]),
            "centroid": centroid.tolist(),
            "centroid_norm": norm,
            "loss": snap["metrics"][f"loss_{group}"],
            "accuracy": snap["metrics"][f"acc_{group}"],
            "margin": snap["metrics"][f"margin_{group}"],
            "cos_to_v1": float(np.dot(centroid, V[:, 0]) / max(norm, 1e-12)),
            "cos_to_v2": float(np.dot(centroid, V[:, 1]) / max(norm, 1e-12)),
        }

    pairwise = {}
    for g1 in active_groups:
        pairwise[g1] = {}
        for g2 in active_groups:
            c1, c2 = centroids[g1], centroids[g2]
            pairwise[g1][g2] = float(
                np.dot(c1, c2) / max(np.linalg.norm(c1) * np.linalg.norm(c2), 1e-12)
            )

    return {
        "experiment_name": cfg.experiment_name,
        "final_step": snap["step"],
        "matrix": cfg.matrix_to_visualize,
        "final_metrics": snap["metrics"],
        "singular_values": S.tolist(),
        "sigma1_over_sigma2": float(S[0] / max(S[1], 1e-12)),
        "groups": group_summary,
        "pairwise_centroid_cosine": pairwise,
    }


def render_and_save(cfg: Config):
    (
        _model,
        history,
        vocab,
        groups,
        token_to_group,
        active_groups,
        centers,
        group_probs,
    ) = train_and_record(cfg)
    outdir = os.path.join(cfg.outdir, cfg.experiment_name)
    os.makedirs(outdir, exist_ok=True)
    sv_scale = (
        auto_singular_vector_scale(history, cfg.matrix_to_visualize)
        if cfg.singular_vector_scale is None
        else cfg.singular_vector_scale
    )
    group_colors = color_map(active_groups)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=True)
    plot_snapshots(
        history,
        vocab,
        token_to_group,
        group_colors,
        cfg,
        os.path.join(outdir, f"01_{cfg.matrix_to_visualize}_snapshots_real_scale.png"),
        sv_scale,
    )
    plot_trajectories(
        history,
        vocab,
        token_to_group,
        group_colors,
        cfg,
        os.path.join(outdir, f"02_{cfg.matrix_to_visualize}_all_trajectories_real_scale.png"),
        sv_scale,
    )
    plot_singular_values(
        history,
        cfg,
        os.path.join(outdir, f"03_{cfg.matrix_to_visualize}_singular_values.png"),
    )
    plot_training_metrics(
        history,
        active_groups,
        cfg,
        os.path.join(outdir, "04_training_metrics.png"),
    )
    summary = summarize(history, vocab, groups, active_groups, cfg, centers, group_probs)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"Done: {cfg.experiment_name} -> {outdir}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Configurable 2D toy bigram embedding dynamics.")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--group_probs", required=True)
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "sweeps"))
    parser.add_argument("--matrix_to_visualize", choices=["E", "W"], default="E")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--record_every", type=int, default=20)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument(
        "--init_layout",
        choices=["spread", "packed_x_pos", "packed_x_neg", "packed_y_pos", "packed_y_neg", "random"],
        default="spread",
    )
    parser.add_argument("--tail_centers", default="")
    parser.add_argument("--init_noise", type=float, default=0.0)
    parser.add_argument("--num_snapshots", type=int, default=6)
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    cfg = parse_args()
    render_and_save(cfg)


if __name__ == "__main__":
    main()
