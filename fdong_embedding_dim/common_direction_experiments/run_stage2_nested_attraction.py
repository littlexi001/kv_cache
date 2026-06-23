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

from run_stage1_nucleation import TiedConcatLM, cosine, spectrum, top_right_vector


@dataclass
class Config:
    outdir: str
    seeds: str = "0,1,2,3,4"
    dim: int = 8
    branches: int = 8
    steps: int = 600
    lr: float = 0.05
    record_every: int = 10
    gain: float = 4.0
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


def build_data(topology: str, branches: int, ids: Dict[str, int]) -> Dict[str, object]:
    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    families: List[str] = []
    branch_ids: List[int] = []
    for i in range(branches):
        perm = i if topology == "nested" else (i + 1) % branches
        g0, g1, g2 = ids[f"G{i}_0"], ids[f"G{i}_1"], ids[f"G{i}_2"]
        p0, p1, p2 = ids[f"G{perm}_0"], ids[f"G{perm}_1"], ids[f"G{perm}_2"]
        patterns = [
            (g0, g1, ids["K"], "to_k"),
            (g1, ids["K"], g2 if topology == "nested" else p2, "from_k_1"),
            (ids["K"], g2, g0 if topology == "nested" else p0, "from_k_2"),
            (g2, g0, g1 if topology == "nested" else p1, "internal"),
        ]
        for a, b, target, family in patterns:
            c1.append(a)
            c2.append(b)
            targets.append(target)
            families.append(family)
            branch_ids.append(i)
    return {
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "weights": torch.full((len(targets),), 1.0 / len(targets)),
        "families": families,
        "branch_ids": branch_ids,
    }


def initialize(vocab_size: int, dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    E = torch.randn(vocab_size, dim)
    E = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eye = torch.eye(dim)
    M = torch.cat([eye, eye], dim=1) / math.sqrt(2.0)
    M += 0.01 * torch.randn_like(M)
    return E, M


def hidden_gradient(model: TiedConcatLM, data: Dict[str, object]) -> torch.Tensor:
    h = model.hidden(data["c1"], data["c2"])
    logits = h @ model.E.T
    probs = logits.softmax(dim=-1)
    probs = probs.clone()
    probs[torch.arange(len(data["targets"])), data["targets"]] -= 1.0
    return probs @ model.E


def orthogonal_direction(direction: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(10000 + seed)
    candidate = torch.randn(direction.shape, generator=gen)
    candidate = candidate - torch.dot(candidate, direction) * direction
    return candidate / candidate.norm().clamp_min(1e-12)


def apply_gain(E: torch.Tensor, direction: torch.Tensor, gain: float) -> torch.Tensor:
    transform = torch.eye(E.shape[1]) + (gain - 1.0) * torch.outer(direction, direction)
    return E @ transform


def clip_top_singular(matrix: torch.Tensor, ratio: float) -> None:
    with torch.no_grad():
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        if len(s) > 1:
            s[0] = torch.minimum(s[0], ratio * s[1])
        matrix.copy_((u * s) @ vh)


def projection_energy(x: torch.Tensor, direction: torch.Tensor) -> float:
    denom = float(x.square().sum())
    if denom < 1e-12:
        return 0.0
    return float((x @ direction).square().sum()) / denom


def effective_rank(x: torch.Tensor) -> float:
    s2 = torch.linalg.svdvals(x.detach()).square()
    denom = float(s2.square().sum())
    if denom < 1e-12:
        return 0.0
    return float(s2.sum().square()) / denom


def extra_residuals(h: torch.Tensor, families: List[str]) -> torch.Tensor:
    pieces = []
    for family in ["from_k_1", "from_k_2", "internal"]:
        idx = torch.tensor([x == family for x in families], dtype=torch.bool)
        subset = h[idx]
        pieces.append(subset - subset.mean(dim=0, keepdim=True))
    return torch.cat(pieces, dim=0)


def family_metrics(logits: torch.Tensor, targets: torch.Tensor, families: List[str]) -> Dict[str, float]:
    losses = F.cross_entropy(logits, targets, reduction="none")
    pred = logits.argmax(dim=-1)
    result: Dict[str, float] = {}
    for family in sorted(set(families)):
        idx = torch.tensor([x == family for x in families], dtype=torch.bool)
        rows = logits[idx].clone()
        tgts = targets[idx]
        target_logits = rows[torch.arange(len(tgts)), tgts]
        rows[torch.arange(len(tgts)), tgts] = -float("inf")
        result[f"{family}_loss"] = float(losses[idx].mean())
        result[f"{family}_accuracy"] = float((pred[idx] == tgts).float().mean())
        result[f"{family}_margin"] = float((target_logits - rows.max(dim=-1).values).mean())
    return result


def measure(
    model: TiedConcatLM,
    data: Dict[str, object],
    condition: str,
    seed: int,
    step: int,
    reference: torch.Tensor,
    initial_gradient_direction: torch.Tensor,
) -> Dict[str, object]:
    with torch.no_grad():
        h = model.hidden(data["c1"], data["c2"])
        logits = h @ model.E.T
        gh = hidden_gradient(model, data)
        extra = extra_residuals(h, data["families"])
        e_spec = spectrum(model.E)
        m_spec = spectrum(model.M)
        top_e = top_right_vector(model.E)
        row: Dict[str, object] = {
            "condition": condition,
            "seed": seed,
            "step": step,
            "loss": float((F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()),
            "accuracy": float((logits.argmax(dim=-1) == data["targets"]).float().mean()),
            "hidden_gradient_reference_energy": projection_energy(gh, reference),
            "hidden_reference_energy": projection_energy(h, reference),
            "extra_reference_energy": projection_energy(extra, reference),
            "extra_effective_rank": effective_rank(extra),
            "e_top_reference_cosine_abs": abs(cosine(top_e, reference)),
            "reference_initial_gradient_cosine_abs": abs(cosine(reference, initial_gradient_direction)),
        }
        for prefix, spec in [("e", e_spec), ("m", m_spec)]:
            for key, value in spec.items():
                row[f"{prefix}_{key}"] = value
        row.update(family_metrics(logits, data["targets"], data["families"]))
        return row


def prepare_model(
    cfg: Config, condition: str, seed: int
) -> Tuple[TiedConcatLM, Dict[str, object], torch.Tensor, torch.Tensor]:
    topology = "rewired" if condition.startswith("rewired") else "nested"
    vocab, ids = make_vocab(cfg.branches)
    E0, M0 = initialize(len(vocab), cfg.dim, seed)
    nested_data = build_data("nested", cfg.branches, ids)
    flat_model = TiedConcatLM(E0, M0)
    initial_gh = hidden_gradient(flat_model, nested_data).detach()
    initial_direction = top_right_vector(initial_gh)
    misaligned = orthogonal_direction(initial_direction, seed)
    if "misaligned_gain" in condition:
        reference = misaligned
        E0 = apply_gain(E0, reference, cfg.gain)
    elif "aligned_gain" in condition:
        reference = initial_direction
        E0 = apply_gain(E0, reference, cfg.gain)
    else:
        reference = initial_direction
    model = TiedConcatLM(E0, M0)
    data = build_data(topology, cfg.branches, ids)
    return model, data, reference, initial_direction


def run_one(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    model, data, reference, initial_direction = prepare_model(cfg, condition, seed)
    rows: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, condition, seed, step, reference, initial_direction))
        if step == cfg.steps:
            break
        logits = model(data["c1"], data["c2"])
        loss = (F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()
        loss.backward()
        with torch.no_grad():
            model.E -= cfg.lr * model.E.grad
            model.M -= cfg.lr * model.M.grad
            model.E.grad = None
            model.M.grad = None
        if condition.endswith("clip") and step + 1 >= cfg.clip_start:
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


def get(agg: List[Dict[str, object]], condition: str, step: int) -> Dict[str, object]:
    return next(x for x in agg if x["condition"] == condition and x["step"] == step)


def summarize(cfg: Config, agg: List[Dict[str, object]]) -> Dict[str, object]:
    flat0 = get(agg, "nested_flat", 0)
    aligned0 = get(agg, "nested_aligned_gain", 0)
    misaligned0 = get(agg, "nested_misaligned_gain", 0)
    nested_final = get(agg, "nested_aligned_gain", cfg.steps)
    rewired_final = get(agg, "rewired_aligned_gain", cfg.steps)
    clipped_final = get(agg, "nested_aligned_gain_clip", cfg.steps)
    checks = {
        "aligned_gain_amplifies_aligned_hidden_gradient": aligned0["hidden_gradient_reference_energy_mean"]
        > max(flat0["hidden_gradient_reference_energy_mean"], misaligned0["hidden_gradient_reference_energy_mean"]) * 1.2,
        "nested_occupies_seeded_direction_more_than_rewired": nested_final["extra_reference_energy_mean"]
        > rewired_final["extra_reference_energy_mean"] * 1.1,
        "clip_reduces_extra_occupation": clipped_final["extra_reference_energy_mean"]
        < nested_final["extra_reference_energy_mean"],
        "clip_reduces_embedding_concentration": clipped_final["e_top1_energy_mean"]
        < nested_final["e_top1_energy_mean"],
        "clip_preserves_accuracy": clipped_final["accuracy_mean"] >= 0.99,
    }
    status = "pass" if all(checks.values()) else "partial" if any(checks.values()) else "fail"
    return {"status": status, "checks": checks, "selected_rows": {
        "flat0": flat0,
        "aligned0": aligned0,
        "misaligned0": misaligned0,
        "nested_final": nested_final,
        "rewired_final": rewired_final,
        "clipped_final": clipped_final,
    }}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, agg: List[Dict[str, object]]) -> None:
    metrics = [
        ("hidden_gradient_reference_energy_mean", "Hidden-gradient energy in seeded direction"),
        ("extra_reference_energy_mean", "Centered extra-feature energy in seeded direction"),
        ("extra_effective_rank_mean", "Centered extra-feature effective rank"),
        ("e_top1_energy_mean", "Embedding top-1 energy"),
        ("loss_mean", "Weighted CE"),
        ("accuracy_mean", "All-pattern accuracy"),
    ]
    conditions = sorted(set(str(x["condition"]) for x in agg))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for condition in conditions:
            items = [x for x in agg if x["condition"] == condition]
            xs = np.array([int(x["step"]) for x in items])
            ys = np.array([float(x[metric]) for x in items])
            sd = np.array([float(x[metric.replace("_mean", "_std")]) for x in items])
            ax.plot(xs, ys, label=condition, linewidth=1.8)
            ax.fill_between(xs, ys - sd, ys + sd, alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 2: frequency-matched nested vs rewired attraction", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument("--gain", type=float, default=4.0)
    parser.add_argument("--clip_start", type=int, default=100)
    parser.add_argument("--clip_ratio", type=float, default=1.2)
    cfg = Config(**vars(parser.parse_args()))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    conditions = [
        "nested_flat",
        "nested_aligned_gain",
        "nested_misaligned_gain",
        "rewired_flat",
        "rewired_aligned_gain",
        "nested_aligned_gain_clip",
    ]
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
