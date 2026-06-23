#!/usr/bin/env python3
import argparse
import csv
import json
import math
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

from run_stage1_nucleation import TiedConcatLM, exact_gradient_parts, spectrum, top_right_vector
from run_stage2_nested_attraction import clip_top_singular, effective_rank


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
    stop_k_loss: float = 0.1
    clip_start: int = 100
    clip_ratio: float = 1.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_vocab(branches: int) -> Tuple[List[str], Dict[str, int]]:
    names = ["K"]
    for i in range(branches):
        names.extend([f"G{i}_0", f"G{i}_1", f"G{i}_2"])
    return names, {name: i for i, name in enumerate(names)}


def initialize(vocab_size: int, dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    E = torch.randn(vocab_size, dim)
    E = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eye = torch.eye(dim)
    M = torch.cat([eye, eye], dim=1) / math.sqrt(2.0)
    M += 0.01 * torch.randn_like(M)
    return E, M


def build_data(cfg: Config, condition: str, ids: Dict[str, int]) -> Dict[str, object]:
    if condition == "uniform":
        group_probs = [1.0 / cfg.branches] * cfg.branches
    else:
        tail_prob = (1.0 - cfg.zipf_common_prob) / (cfg.branches - 1)
        group_probs = [cfg.zipf_common_prob] + [tail_prob] * (cfg.branches - 1)
    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    families: List[str] = []
    groups: List[int] = []
    base_weights: List[float] = []
    for i, prob in enumerate(group_probs):
        g0, g1, g2 = ids[f"G{i}_0"], ids[f"G{i}_1"], ids[f"G{i}_2"]
        patterns = [
            (g0, g1, ids["K"], "to_k"),
            (g1, ids["K"], g2, "from_k_1"),
            (ids["K"], g2, g0, "from_k_2"),
            (g2, g0, g1, "internal"),
        ]
        for a, b, target, family in patterns:
            c1.append(a)
            c2.append(b)
            targets.append(target)
            families.append(family)
            groups.append(i)
            base_weights.append(prob / 4.0)
    weights = torch.tensor(base_weights, dtype=torch.float32)
    if condition == "zipf_reweight":
        target_mass: Dict[int, float] = {}
        for target, weight in zip(targets, base_weights):
            target_mass[target] = target_mass.get(target, 0.0) + weight
        weights = torch.tensor(
            [w / (target_mass[t] ** cfg.reweight_alpha) for t, w in zip(targets, base_weights)],
            dtype=torch.float32,
        )
        weights /= weights.sum()
    return {
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "base_weights": torch.tensor(base_weights, dtype=torch.float32),
        "weights": weights,
        "families": families,
        "groups": groups,
    }


def subset_data(data: Dict[str, object], mask: torch.Tensor, keep_absolute_weights: bool = True) -> Dict[str, object]:
    weights = data["weights"][mask].clone()
    if not keep_absolute_weights:
        weights /= weights.sum().clamp_min(1e-12)
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    return {
        "c1": data["c1"][mask],
        "c2": data["c2"][mask],
        "targets": data["targets"][mask],
        "weights": weights,
        "families": [data["families"][i] for i in indices],
        "groups": [data["groups"][i] for i in indices],
    }


def flatten_gradient(parts: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([parts["g_total_e"].reshape(-1), parts["g_m"].reshape(-1)])


def gradient_diagnostics(model: TiedConcatLM, data: Dict[str, object], branches: int) -> Dict[str, float]:
    total = flatten_gradient(exact_gradient_parts(model, data))
    group_gradients = []
    for group in range(branches):
        mask = torch.tensor([x == group for x in data["groups"]], dtype=torch.bool)
        group_gradients.append(flatten_gradient(exact_gradient_parts(model, subset_data(data, mask))))
    common = group_gradients[0]
    tail_sir = []
    tail_common_cos = []
    for grad in group_gradients[1:]:
        interference = total - grad
        tail_sir.append(float(grad.norm() / interference.norm().clamp_min(1e-12)))
        denom = grad.norm() * common.norm()
        tail_common_cos.append(float(torch.dot(grad, common) / denom.clamp_min(1e-12)))
    k_mask = torch.tensor([x == "to_k" for x in data["families"]], dtype=torch.bool)
    k_grad = flatten_gradient(exact_gradient_parts(model, subset_data(data, k_mask)))
    return {
        "tail_sir_mean": float(np.mean(tail_sir)),
        "tail_sir_min": float(np.min(tail_sir)),
        "tail_common_gradient_cosine_mean": float(np.mean(tail_common_cos)),
        "k_gradient_norm": float(k_grad.norm()),
        "total_gradient_norm": float(total.norm()),
    }


def tail_residual_metrics(
    h: torch.Tensor, groups: List[int], top_direction: torch.Tensor, branches: int
) -> Dict[str, float]:
    centroids = []
    for group in range(1, branches):
        mask = torch.tensor([x == group for x in groups], dtype=torch.bool)
        centroids.append(h[mask].mean(dim=0))
    matrix = torch.stack(centroids)
    top_energy = float((matrix @ top_direction).square().sum() / matrix.square().sum().clamp_min(1e-12))
    residual = matrix - (matrix @ top_direction)[:, None] * top_direction[None, :]
    return {
        "tail_top_direction_energy": top_energy,
        "tail_residual_effective_rank": effective_rank(residual),
    }


def family_group_metrics(logits: torch.Tensor, data: Dict[str, object]) -> Dict[str, float]:
    targets = data["targets"]
    losses = F.cross_entropy(logits, targets, reduction="none")
    pred = logits.argmax(dim=-1)
    rows = logits.clone()
    target_logits = rows[torch.arange(len(targets)), targets]
    rows[torch.arange(len(targets)), targets] = -float("inf")
    margins = target_logits - rows.max(dim=-1).values
    result: Dict[str, float] = {}
    masks = {
        "k": torch.tensor([x == "to_k" for x in data["families"]], dtype=torch.bool),
        "common": torch.tensor([x == 0 for x in data["groups"]], dtype=torch.bool),
        "tail": torch.tensor([x > 0 for x in data["groups"]], dtype=torch.bool),
    }
    for name, mask in masks.items():
        result[f"{name}_loss"] = float(losses[mask].mean())
        result[f"{name}_accuracy"] = float((pred[mask] == targets[mask]).float().mean())
        result[f"{name}_margin"] = float(margins[mask].mean())
    return result


def measure(
    model: TiedConcatLM,
    data: Dict[str, object],
    condition: str,
    seed: int,
    step: int,
    stop_step: int,
    branches: int,
) -> Dict[str, object]:
    with torch.no_grad():
        h = model.hidden(data["c1"], data["c2"])
        logits = h @ model.E.T
        top = top_right_vector(model.E)
        e_spec = spectrum(model.E)
        hidden_spec = spectrum(h)
        row: Dict[str, object] = {
            "condition": condition,
            "seed": seed,
            "step": step,
            "stop_step": stop_step,
            "loss": float((F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()),
        }
        for key, value in e_spec.items():
            row[f"e_{key}"] = value
        for key, value in hidden_spec.items():
            row[f"hidden_{key}"] = value
        row.update(family_group_metrics(logits, data))
        row.update(tail_residual_metrics(h, data["groups"], top, branches))
        row.update(gradient_diagnostics(model, data, branches))
        return row


def run_one(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    vocab, ids = make_vocab(cfg.branches)
    E0, M0 = initialize(len(vocab), cfg.dim, seed)
    model = TiedConcatLM(E0, M0)
    data = build_data(cfg, condition, ids)
    stop_step = -1
    rows: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, condition, seed, step, stop_step, cfg.branches))
        if step == cfg.steps:
            break
        if condition == "zipf_stop_k" and stop_step < 0:
            with torch.no_grad():
                logits = model(data["c1"], data["c2"])
                losses = F.cross_entropy(logits, data["targets"], reduction="none")
                k_mask = torch.tensor([x == "to_k" for x in data["families"]], dtype=torch.bool)
                if float(losses[k_mask].mean()) < cfg.stop_k_loss:
                    stop_step = step
                    data["weights"] = data["weights"].clone()
                    data["weights"][k_mask] = 0.0
                    data["weights"] /= data["weights"].sum()
        logits = model(data["c1"], data["c2"])
        loss = (F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()
        loss.backward()
        with torch.no_grad():
            model.E -= cfg.lr * model.E.grad
            model.M -= cfg.lr * model.M.grad
            model.E.grad = None
            model.M.grad = None
        if condition == "zipf_clip" and step + 1 >= cfg.clip_start:
            clip_top_singular(model.E, cfg.clip_ratio)
    for i, row in enumerate(rows):
        if i + 1 < len(rows):
            row["e_sigma1_delta_next"] = float(rows[i + 1]["e_sigma1"]) - float(row["e_sigma1"])
        else:
            row["e_sigma1_delta_next"] = 0.0
    return rows


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), int(row["step"])), []).append(row)
    output: List[Dict[str, object]] = []
    for (condition, step), items in sorted(groups.items()):
        out: Dict[str, object] = {"condition": condition, "step": step, "num_seeds": len(items)}
        keys = sorted(
            key for key in set.intersection(*(set(x) for x in items))
            if key not in {"condition", "seed", "step"}
            and all(isinstance(x[key], (int, float)) for x in items)
        )
        for key in keys:
            vals = np.array([float(x[key]) for x in items])
            out[f"{key}_mean"] = float(vals.mean())
            out[f"{key}_std"] = float(vals.std())
        output.append(out)
    return output


def get_trajectory(agg: List[Dict[str, object]], condition: str) -> List[Dict[str, object]]:
    return sorted([x for x in agg if x["condition"] == condition], key=lambda x: int(x["step"]))


def first_stable_step(traj: List[Dict[str, object]], metric: str, threshold: float = 0.999) -> int:
    values = [float(x[metric]) for x in traj]
    for i, value in enumerate(values):
        if value >= threshold and all(v >= threshold for v in values[i:]):
            return int(traj[i]["step"])
    return -1


def summarize(cfg: Config, agg: List[Dict[str, object]]) -> Dict[str, object]:
    trajectories = {condition: get_trajectory(agg, condition) for condition in [
        "uniform", "zipf", "zipf_reweight", "zipf_stop_k", "zipf_clip"
    ]}
    zipf = trajectories["zipf"]
    pre = [x for x in zipf if float(x["k_loss_mean"]) >= cfg.stop_k_loss]
    post = [x for x in zipf if float(x["k_loss_mean"]) < cfg.stop_k_loss and int(x["step"]) < cfg.steps]
    pre_grad = float(np.mean([float(x["k_gradient_norm_mean"]) for x in pre])) if pre else float("nan")
    post_grad = float(np.mean([float(x["k_gradient_norm_mean"]) for x in post])) if post else float("nan")
    pre_delta = float(np.mean([float(x["e_sigma1_delta_next_mean"]) for x in pre])) if pre else float("nan")
    post_delta = float(np.mean([float(x["e_sigma1_delta_next_mean"]) for x in post])) if post else float("nan")
    lag_x = np.array([float(x["hidden_top1_energy_mean"]) for x in zipf[:-1]])
    lag_y = np.array([float(x["e_sigma1_delta_next_mean"]) for x in zipf[:-1]])
    lag_corr = float(np.corrcoef(lag_x, lag_y)[0, 1])
    stable = {condition: first_stable_step(traj, "tail_accuracy_mean") for condition, traj in trajectories.items()}
    zipf_final = zipf[-1]
    intervention_final = {condition: trajectories[condition][-1] for condition in ["zipf_reweight", "zipf_stop_k", "zipf_clip"]}
    successful_interventions = []
    for condition, final in intervention_final.items():
        speed_improved = stable["zipf"] >= 0 and 0 <= stable[condition] < stable["zipf"]
        rank_improved = float(final["tail_residual_effective_rank_mean"]) > float(
            zipf_final["tail_residual_effective_rank_mean"]
        )
        k_preserved = float(final["k_accuracy_mean"]) >= 0.99
        if (speed_improved or rank_improved) and k_preserved:
            successful_interventions.append(condition)
    checks = {
        "k_gradient_saturates": post_grad < pre_grad * 0.5,
        "sigma_growth_saturates": post_delta < pre_delta * 0.5,
        "hidden_occupation_predicts_sigma_growth": lag_corr > 0.3,
        "zipf_delays_tail": stable["zipf"] > stable["uniform"] >= 0,
        "intervention_improves_tail_and_preserves_k": bool(successful_interventions),
    }
    status = "pass" if all(checks.values()) else "partial" if any(checks.values()) else "fail"
    return {
        "status": status,
        "checks": checks,
        "stable_tail_accuracy_step": stable,
        "saturation": {
            "pre_k_gradient_norm": pre_grad,
            "post_k_gradient_norm": post_grad,
            "pre_sigma1_delta": pre_delta,
            "post_sigma1_delta": post_delta,
        },
        "hidden_occupation_to_next_sigma_delta_correlation": lag_corr,
        "successful_interventions": successful_interventions,
        "final_rows": {"zipf": zipf_final, **intervention_final},
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, agg: List[Dict[str, object]]) -> None:
    metrics = [
        ("k_loss_mean", "K-target CE"),
        ("k_gradient_norm_mean", "K-target gradient norm"),
        ("e_sigma1_mean", "Embedding sigma1"),
        ("e_sigma1_delta_next_mean", "Next-interval delta sigma1"),
        ("tail_accuracy_mean", "Tail macro accuracy"),
        ("tail_residual_effective_rank_mean", "Tail residual effective rank"),
        ("tail_sir_mean_mean", "Tail gradient SIR"),
        ("e_top1_energy_mean", "Embedding top-1 energy"),
    ]
    conditions = sorted(set(str(x["condition"]) for x in agg))
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for condition in conditions:
            items = get_trajectory(agg, condition)
            xs = np.array([int(x["step"]) for x in items])
            ys = np.array([float(x[metric]) for x in items])
            std_metric = metric[:-5] + "_std" if metric.endswith("_mean") else metric + "_std"
            sd = np.array([float(x[std_metric]) for x in items])
            ax.plot(xs, ys, label=condition, linewidth=1.8)
            ax.fill_between(xs, ys - sd, ys + sd, alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 3: natural feedback, saturation, and tail harm", fontsize=14)
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
    parser.add_argument("--stop_k_loss", type=float, default=0.1)
    parser.add_argument("--clip_start", type=int, default=100)
    parser.add_argument("--clip_ratio", type=float, default=1.2)
    cfg = Config(**vars(parser.parse_args()))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conditions = ["uniform", "zipf", "zipf_reweight", "zipf_stop_k", "zipf_clip"]
    rows: List[Dict[str, object]] = []
    for condition in conditions:
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
    print(json.dumps(summary["checks"], indent=2), flush=True)
    print(f"status={summary['status']}", flush=True)


if __name__ == "__main__":
    main()
