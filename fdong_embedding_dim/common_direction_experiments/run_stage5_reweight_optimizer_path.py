#!/usr/bin/env python3
import argparse
import csv
import json
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

from run_stage2_nested_attraction import clip_top_singular, effective_rank
from run_stage3_feedback_saturation import (
    TiedConcatLM,
    build_data,
    exact_gradient_parts,
    family_group_metrics,
    initialize,
    make_vocab,
    spectrum,
)


@dataclass
class Config:
    outdir: str
    seeds: str = "0,1,2,3,4"
    dim: int = 8
    branches: int = 8
    steps: int = 1200
    lr: float = 0.05
    record_every: int = 10
    zipf_common_prob: float = 0.55
    reweight_alpha: float = 0.5
    clip_start: int = 100
    clip_ratio: float = 1.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def centered(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def top_right_vector(x: torch.Tensor) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(x.detach(), full_matrices=False)
    return vh[0]


def orthogonal_residual(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return x - (x @ v)[:, None] * v[None, :]


def loss_parts(model: TiedConcatLM, data: Dict[str, object]) -> Dict[str, float]:
    logits = model(data["c1"], data["c2"])
    out = family_group_metrics(logits, data)
    out["weighted_loss"] = float((F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum())
    return out


def clone_after_step(model: TiedConcatLM, parts: Dict[str, torch.Tensor], lr: float) -> TiedConcatLM:
    next_model = TiedConcatLM(model.E.detach(), model.M.detach())
    with torch.no_grad():
        next_model.E.copy_(model.E.detach() - lr * parts["g_total_e"])
        next_model.M.copy_(model.M.detach() - lr * parts["g_m"])
    return next_model


def update_path_metrics(
    model: TiedConcatLM,
    data: Dict[str, object],
    cfg: Config,
    condition: str,
) -> Dict[str, float]:
    parts = exact_gradient_parts(model, data)
    g_e = parts["g_total_e"]
    g_m = parts["g_m"]

    # Centered embedding top direction is the measured common semantic/scale
    # channel. The raw top direction is also recorded to catch mean effects.
    v_centered = top_right_vector(centered(model.E))
    v_raw = top_right_vector(model.E)

    e_energy = float(g_e.square().sum())
    m_energy = float(g_m.square().sum())
    total_energy = e_energy + m_energy
    e_common_energy = float((g_e @ v_centered).square().sum())
    e_raw_common_energy = float((g_e @ v_raw).square().sum())
    # M maps concatenated embeddings to hidden. Project row updates onto the
    # same hidden/common direction.
    m_common_energy = float((v_centered @ g_m).square().sum())
    m_raw_common_energy = float((v_raw @ g_m).square().sum())

    g_e_residual = orthogonal_residual(g_e, v_centered)
    e_update_eff_rank = effective_rank(g_e)
    e_residual_update_eff_rank = effective_rank(g_e_residual)

    before = loss_parts(model, data)
    next_model = clone_after_step(model, parts, cfg.lr)
    if condition == "zipf_clip":
        with torch.no_grad():
            clip_top_singular(next_model.E, cfg.clip_ratio)
    after = loss_parts(next_model, data)

    tail_delta = before["tail_loss"] - after["tail_loss"]
    common_delta = before["common_loss"] - after["common_loss"]
    weighted_delta = before["weighted_loss"] - after["weighted_loss"]
    tail_margin_delta = after["tail_margin"] - before["tail_margin"]
    common_margin_delta = after["common_margin"] - before["common_margin"]

    return {
        "grad_energy_total": total_energy,
        "grad_energy_e": e_energy,
        "grad_energy_m": m_energy,
        "e_update_common_share": e_common_energy / max(e_energy, 1e-12),
        "m_update_common_share": m_common_energy / max(m_energy, 1e-12),
        "total_update_common_share": (e_common_energy + m_common_energy) / max(total_energy, 1e-12),
        "e_update_raw_common_share": e_raw_common_energy / max(e_energy, 1e-12),
        "m_update_raw_common_share": m_raw_common_energy / max(m_energy, 1e-12),
        "total_update_raw_common_share": (e_raw_common_energy + m_raw_common_energy) / max(total_energy, 1e-12),
        "e_update_effective_rank": e_update_eff_rank,
        "e_residual_update_effective_rank": e_residual_update_eff_rank,
        "tail_loss_delta_next_step": tail_delta,
        "common_loss_delta_next_step": common_delta,
        "weighted_loss_delta_next_step": weighted_delta,
        "tail_margin_delta_next_step": tail_margin_delta,
        "common_margin_delta_next_step": common_margin_delta,
        "tail_delta_per_grad_energy": tail_delta / max(total_energy, 1e-12),
        "common_delta_per_grad_energy": common_delta / max(total_energy, 1e-12),
    }


def tail_representation_metrics(model: TiedConcatLM, data: Dict[str, object], cfg: Config) -> Dict[str, float]:
    with torch.no_grad():
        h = model.hidden(data["c1"], data["c2"])
        v = top_right_vector(centered(model.E))
        centroids = []
        for group in range(1, cfg.branches):
            mask = torch.tensor([x == group for x in data["groups"]], dtype=torch.bool)
            centroids.append(h[mask].mean(dim=0))
        matrix = torch.stack(centroids)
        residual = orthogonal_residual(matrix, v)
        return {
            "tail_top_direction_energy": float((matrix @ v).square().sum() / matrix.square().sum().clamp_min(1e-12)),
            "tail_residual_effective_rank": effective_rank(residual),
            "hidden_centered_top1_energy": spectrum(centered(h))["top1_energy"],
            "e_centered_top1_energy": spectrum(centered(model.E))["top1_energy"],
            "e_sigma1": spectrum(model.E)["sigma1"],
        }


def measure(model: TiedConcatLM, data: Dict[str, object], cfg: Config, condition: str, seed: int, step: int) -> Dict[str, object]:
    with torch.no_grad():
        row: Dict[str, object] = {
            "condition": condition,
            "seed": seed,
            "step": step,
        }
        row.update(loss_parts(model, data))
        row.update(tail_representation_metrics(model, data, cfg))
        row.update(update_path_metrics(model, data, cfg, condition))
        return row


def run_one(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    vocab, ids = make_vocab(cfg.branches)
    e0, m0 = initialize(len(vocab), cfg.dim, seed)
    model = TiedConcatLM(e0, m0)
    data = build_data(cfg, condition, ids)
    rows: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, cfg, condition, seed, step))
        if step == cfg.steps:
            break
        parts = exact_gradient_parts(model, data)
        with torch.no_grad():
            model.E -= cfg.lr * parts["g_total_e"]
            model.M -= cfg.lr * parts["g_m"]
            if condition == "zipf_clip" and step + 1 >= cfg.clip_start:
                clip_top_singular(model.E, cfg.clip_ratio)
    return rows


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), int(row["step"])), []).append(row)
    output: List[Dict[str, object]] = []
    for (condition, step), items in sorted(groups.items()):
        out: Dict[str, object] = {"condition": condition, "step": step, "num_seeds": len(items)}
        keys = sorted(
            key
            for key in set.intersection(*(set(x) for x in items))
            if key not in {"condition", "seed", "step"}
            and all(isinstance(x[key], (int, float)) for x in items)
        )
        for key in keys:
            vals = np.array([float(x[key]) for x in items])
            out[f"{key}_mean"] = float(vals.mean())
            out[f"{key}_std"] = float(vals.std(ddof=0))
        output.append(out)
    return output


def traj(agg: List[Dict[str, object]], condition: str) -> List[Dict[str, object]]:
    return sorted([x for x in agg if x["condition"] == condition], key=lambda x: int(x["step"]))


def window_mean(items: List[Dict[str, object]], metric: str, start: int, end: int) -> float:
    vals = [float(x[metric]) for x in items if start <= int(x["step"]) <= end]
    return float(np.mean(vals)) if vals else float("nan")


def first_stable_step(items: List[Dict[str, object]], metric: str, threshold: float = 0.999) -> int:
    values = [float(x[metric]) for x in items]
    for i, value in enumerate(values):
        if value >= threshold and all(v >= threshold for v in values[i:]):
            return int(items[i]["step"])
    return -1


def summarize(cfg: Config, agg: List[Dict[str, object]]) -> Dict[str, object]:
    trajectories = {c: traj(agg, c) for c in ["uniform", "zipf", "zipf_reweight", "zipf_clip"]}
    nucleation = (0, 50)
    early = (0, 200)
    mid = (200, 700)
    final_step = cfg.steps
    z = trajectories["zipf"]
    r = trajectories["zipf_reweight"]
    u = trajectories["uniform"]
    c = trajectories["zipf_clip"]
    z_final = z[-1]
    r_final = r[-1]
    c_final = c[-1]
    checks = {
        "reweight_reduces_nucleation_common_update_share": window_mean(r, "total_update_common_share_mean", *nucleation)
        < window_mean(z, "total_update_common_share_mean", *nucleation),
        "reweight_increases_nucleation_update_rank": window_mean(r, "e_update_effective_rank_mean", *nucleation)
        > window_mean(z, "e_update_effective_rank_mean", *nucleation),
        "reweight_increases_nucleation_residual_update_rank": window_mean(
            r, "e_residual_update_effective_rank_mean", *nucleation
        )
        > window_mean(z, "e_residual_update_effective_rank_mean", *nucleation),
        "reweight_improves_nucleation_tail_loss_delta": window_mean(r, "tail_loss_delta_next_step_mean", *nucleation)
        > window_mean(z, "tail_loss_delta_next_step_mean", *nucleation),
        "reweight_improves_tail_stable_speed": first_stable_step(r, "tail_accuracy_mean")
        < first_stable_step(z, "tail_accuracy_mean"),
        "reweight_reduces_final_common_concentration": float(r_final["e_centered_top1_energy_mean"])
        < float(z_final["e_centered_top1_energy_mean"]),
        "clip_reduces_spectrum_but_not_tail_path": float(c_final["e_centered_top1_energy_mean"])
        < float(z_final["e_centered_top1_energy_mean"])
        and first_stable_step(c, "tail_accuracy_mean") >= first_stable_step(z, "tail_accuracy_mean"),
    }
    status = "pass" if all(checks.values()) else "partial" if any(checks.values()) else "fail"
    key_numbers = {
        "nucleation_common_update_share": {
            "uniform": window_mean(u, "total_update_common_share_mean", *nucleation),
            "zipf": window_mean(z, "total_update_common_share_mean", *nucleation),
            "zipf_reweight": window_mean(r, "total_update_common_share_mean", *nucleation),
            "zipf_clip": window_mean(c, "total_update_common_share_mean", *nucleation),
        },
        "nucleation_update_effective_rank": {
            "uniform": window_mean(u, "e_update_effective_rank_mean", *nucleation),
            "zipf": window_mean(z, "e_update_effective_rank_mean", *nucleation),
            "zipf_reweight": window_mean(r, "e_update_effective_rank_mean", *nucleation),
            "zipf_clip": window_mean(c, "e_update_effective_rank_mean", *nucleation),
        },
        "nucleation_residual_update_effective_rank": {
            "uniform": window_mean(u, "e_residual_update_effective_rank_mean", *nucleation),
            "zipf": window_mean(z, "e_residual_update_effective_rank_mean", *nucleation),
            "zipf_reweight": window_mean(r, "e_residual_update_effective_rank_mean", *nucleation),
            "zipf_clip": window_mean(c, "e_residual_update_effective_rank_mean", *nucleation),
        },
        "nucleation_tail_loss_delta_next_step": {
            "uniform": window_mean(u, "tail_loss_delta_next_step_mean", *nucleation),
            "zipf": window_mean(z, "tail_loss_delta_next_step_mean", *nucleation),
            "zipf_reweight": window_mean(r, "tail_loss_delta_next_step_mean", *nucleation),
            "zipf_clip": window_mean(c, "tail_loss_delta_next_step_mean", *nucleation),
        },
        "early_common_update_share": {
            "uniform": window_mean(u, "total_update_common_share_mean", *early),
            "zipf": window_mean(z, "total_update_common_share_mean", *early),
            "zipf_reweight": window_mean(r, "total_update_common_share_mean", *early),
            "zipf_clip": window_mean(c, "total_update_common_share_mean", *early),
        },
        "mid_residual_update_effective_rank": {
            "uniform": window_mean(u, "e_residual_update_effective_rank_mean", *mid),
            "zipf": window_mean(z, "e_residual_update_effective_rank_mean", *mid),
            "zipf_reweight": window_mean(r, "e_residual_update_effective_rank_mean", *mid),
            "zipf_clip": window_mean(c, "e_residual_update_effective_rank_mean", *mid),
        },
        "mid_tail_loss_delta_next_step": {
            "uniform": window_mean(u, "tail_loss_delta_next_step_mean", *mid),
            "zipf": window_mean(z, "tail_loss_delta_next_step_mean", *mid),
            "zipf_reweight": window_mean(r, "tail_loss_delta_next_step_mean", *mid),
            "zipf_clip": window_mean(c, "tail_loss_delta_next_step_mean", *mid),
        },
        "stable_tail_accuracy_step": {
            "uniform": first_stable_step(u, "tail_accuracy_mean"),
            "zipf": first_stable_step(z, "tail_accuracy_mean"),
            "zipf_reweight": first_stable_step(r, "tail_accuracy_mean"),
            "zipf_clip": first_stable_step(c, "tail_accuracy_mean"),
        },
        "final_e_centered_top1_energy": {
            "uniform": float(u[-1]["e_centered_top1_energy_mean"]),
            "zipf": float(z_final["e_centered_top1_energy_mean"]),
            "zipf_reweight": float(r_final["e_centered_top1_energy_mean"]),
            "zipf_clip": float(c_final["e_centered_top1_energy_mean"]),
        },
        "final_tail_residual_effective_rank": {
            "uniform": float(u[-1]["tail_residual_effective_rank_mean"]),
            "zipf": float(z_final["tail_residual_effective_rank_mean"]),
            "zipf_reweight": float(r_final["tail_residual_effective_rank_mean"]),
            "zipf_clip": float(c_final["tail_residual_effective_rank_mean"]),
        },
    }
    return {"status": status, "checks": checks, "key_numbers": key_numbers}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, agg: List[Dict[str, object]]) -> None:
    metrics = [
        ("total_update_common_share_mean", "common-direction share of current update"),
        ("e_residual_update_effective_rank_mean", "residual embedding-update effective rank"),
        ("tail_loss_delta_next_step_mean", "next-step tail loss decrease"),
        ("tail_margin_delta_next_step_mean", "next-step tail margin increase"),
        ("tail_accuracy_mean", "tail macro accuracy"),
        ("tail_residual_effective_rank_mean", "tail residual representation rank"),
        ("e_centered_top1_energy_mean", "centered embedding top-1 energy"),
        ("hidden_centered_top1_energy_mean", "centered hidden top-1 energy"),
    ]
    conditions = ["uniform", "zipf", "zipf_reweight", "zipf_clip"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for condition in conditions:
            items = traj(agg, condition)
            xs = np.array([int(x["step"]) for x in items])
            ys = np.array([float(x[metric]) for x in items])
            sd = np.array([float(x[metric.replace("_mean", "_std")]) for x in items])
            ax.plot(xs, ys, label=condition, linewidth=1.8)
            ax.fill_between(xs, ys - sd, ys + sd, alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 5: does reweighting change the optimizer path?", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument("--zipf_common_prob", type=float, default=0.55)
    parser.add_argument("--reweight_alpha", type=float, default=0.5)
    parser.add_argument("--clip_start", type=int, default=100)
    parser.add_argument("--clip_ratio", type=float, default=1.2)
    cfg = Config(**vars(parser.parse_args()))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for condition in ["uniform", "zipf", "zipf_reweight", "zipf_clip"]:
        for seed in [int(x) for x in cfg.seeds.split(",") if x.strip()]:
            print(f"running condition={condition} seed={seed}", flush=True)
            rows.extend(run_one(cfg, condition, seed))
    agg = aggregate(rows)
    summary = summarize(cfg, agg)
    write_csv(outdir / "history.csv", rows)
    write_csv(outdir / "aggregate.csv", agg)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(outdir / "metrics.png", agg)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
