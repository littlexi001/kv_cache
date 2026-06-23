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


@dataclass
class Config:
    outdir: str
    seeds: str = "0,1,2,3,4"
    dim: int = 16
    groups: int = 32
    steps: int = 600
    lr: float = 0.08
    record_every: int = 10
    reweight_alpha: float = 0.5


class TiedConcatLM(torch.nn.Module):
    def __init__(self, e0: torch.Tensor, m0: torch.Tensor):
        super().__init__()
        self.E = torch.nn.Parameter(e0.clone())
        self.M = torch.nn.Parameter(m0.clone())

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


def centered(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def spectrum(x: torch.Tensor) -> Dict[str, float]:
    if x.numel() == 0:
        return {"sigma1": 0.0, "sigma2": 0.0, "sigma1_over_sigma2": 0.0, "top1_energy": 0.0}
    x = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    s = torch.linalg.svdvals(x)
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


def initialize(vocab_size: int, dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    e = torch.randn(vocab_size, dim)
    e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eye = torch.eye(dim)
    m = torch.cat([eye, eye], dim=1) / math.sqrt(2.0)
    m = m + 0.01 * torch.randn_like(m)
    return e, m


def token_id(name: str, names: List[str], index: Dict[str, int]) -> int:
    if name not in index:
        index[name] = len(names)
        names.append(name)
    return index[name]


def add_example(
    c1: List[int],
    c2: List[int],
    targets: List[int],
    families: List[str],
    names: List[str],
    index: Dict[str, int],
    a: str,
    b: str,
    y: str,
    family: str,
) -> None:
    c1.append(token_id(a, names, index))
    c2.append(token_id(b, names, index))
    targets.append(token_id(y, names, index))
    families.append(family)


def build_data(condition: str, groups: int, alpha: float) -> Dict[str, object]:
    names: List[str] = []
    index: Dict[str, int] = {}
    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    families: List[str] = []

    for i in range(groups):
        if condition in {"uniform_disjoint", "uniform_disjoint_relabel"}:
            target_i = (i * 7 + 3) % groups if condition.endswith("relabel") else i
            add_example(c1, c2, targets, families, names, index, f"U_A{i}", f"U_B{i}", f"U_C{target_i}", "probe")
            add_example(c1, c2, targets, families, names, index, f"U_D{i}", f"U_E{i}", f"U_F{i}", "background")
        elif condition in {"shared_target", "shared_target_reweight"}:
            add_example(c1, c2, targets, families, names, index, f"S_A{i}", f"S_B{i}", "K", "probe")
            add_example(c1, c2, targets, families, names, index, f"S_D{i}", f"S_E{i}", f"S_F{i}", "background")
        elif condition == "shared_input_prefix":
            add_example(c1, c2, targets, families, names, index, "P", f"P_B{i}", f"P_C{i}", "probe")
            add_example(c1, c2, targets, families, names, index, f"P_D{i}", f"P_E{i}", f"P_F{i}", "background")
        elif condition == "shared_two_token_prefix":
            add_example(c1, c2, targets, families, names, index, "P", "Q", f"Q_C{i}", "probe")
            add_example(c1, c2, targets, families, names, index, f"Q_D{i}", f"Q_E{i}", f"Q_F{i}", "background")
        else:
            raise ValueError(f"unknown condition: {condition}")

    target_counts: Dict[int, int] = {}
    input_counts: Dict[int, int] = {}
    for t in targets:
        target_counts[t] = target_counts.get(t, 0) + 1
    for a, b in zip(c1, c2):
        input_counts[a] = input_counts.get(a, 0) + 1
        input_counts[b] = input_counts.get(b, 0) + 1

    weights = torch.ones(len(targets), dtype=torch.float32)
    if condition.endswith("reweight"):
        weights = torch.tensor([target_counts[t] ** (-alpha) for t in targets], dtype=torch.float32)
    weights = weights / weights.sum()

    return {
        "names": names,
        "token_to_id": index,
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "weights": weights,
        "families": families,
        "target_counts": target_counts,
        "input_counts": input_counts,
    }


def exact_gradient_parts(model: TiedConcatLM, data: Dict[str, object]) -> Dict[str, torch.Tensor]:
    c1 = data["c1"]
    c2 = data["c2"]
    targets = data["targets"]
    weights = data["weights"]
    e = model.E
    m = model.M
    x = torch.cat([e[c1], e[c2]], dim=-1)
    h = x @ m.T
    logits = h @ e.T
    probs = logits.softmax(dim=-1)
    gz = probs.clone()
    gz[torch.arange(len(targets)), targets] -= 1.0
    gz = gz * weights[:, None]

    g_output = gz.T @ h
    gh = gz @ e
    gx = gh @ m
    g_input = torch.zeros_like(e)
    g_input.index_add_(0, c1, gx[:, : e.shape[1]])
    g_input.index_add_(0, c2, gx[:, e.shape[1] :])
    return {
        "h": h,
        "logits": logits,
        "g_output": g_output,
        "g_input": g_input,
        "g_total_e": g_output + g_input,
        "g_m": gh.T @ x,
    }


def family_stats(logits: torch.Tensor, targets: torch.Tensor, families: List[str]) -> Dict[str, float]:
    losses = F.cross_entropy(logits, targets, reduction="none")
    pred = logits.argmax(dim=-1)
    out: Dict[str, float] = {}
    for fam in sorted(set(families)):
        mask = torch.tensor([x == fam for x in families], dtype=torch.bool)
        out[f"{fam}_loss"] = float(losses[mask].mean())
        out[f"{fam}_accuracy"] = float((pred[mask] == targets[mask]).float().mean())
    return out


def probe_mask(families: List[str]) -> torch.Tensor:
    return torch.tensor([x == "probe" for x in families], dtype=torch.bool)


def measure(model: TiedConcatLM, data: Dict[str, object], condition: str, seed: int, step: int) -> Dict[str, object]:
    with torch.no_grad():
        parts = exact_gradient_parts(model, data)
        h = parts["h"]
        logits = parts["logits"]
        names: List[str] = data["names"]
        token_to_id: Dict[str, int] = data["token_to_id"]
        target_counts: Dict[int, int] = data["target_counts"]
        input_counts: Dict[int, int] = data["input_counts"]
        mask = probe_mask(data["families"])
        probe_h = h[mask]
        probe_mean = probe_h.mean(dim=0)

        max_target_count = max(target_counts.values())
        max_input_count = max(input_counts.values())
        most_common_target = max(target_counts, key=lambda k: target_counts[k])
        most_common_input = max(input_counts, key=lambda k: input_counts[k])
        e_top = top_right_vector(centered(model.E))
        common_target_cosine = abs(cosine(model.E[most_common_target], e_top))
        common_input_cosine = abs(cosine(model.E[most_common_input], e_top))
        target_grad_context_cosine = 0.0
        if max_target_count > 1:
            neg_grad = -parts["g_output"][most_common_target]
            target_grad_context_cosine = cosine(neg_grad, probe_mean)

        row: Dict[str, object] = {
            "condition": condition,
            "seed": seed,
            "step": step,
            "vocab_size": len(names),
            "num_examples": int(len(data["targets"])),
            "max_target_count": int(max_target_count),
            "max_input_count": int(max_input_count),
            "weighted_loss": float((F.cross_entropy(logits, data["targets"], reduction="none") * data["weights"]).sum()),
            "common_target_top_e_cosine_abs": common_target_cosine,
            "common_input_top_e_cosine_abs": common_input_cosine,
            "target_grad_context_cosine": target_grad_context_cosine,
        }
        for name, matrix in [
            ("e", model.E),
            ("e_centered", centered(model.E)),
            ("hidden", h),
            ("hidden_centered", centered(h)),
            ("probe_hidden", probe_h),
            ("probe_hidden_centered", centered(probe_h)),
            ("g_output", parts["g_output"]),
            ("g_output_centered", centered(parts["g_output"])),
            ("g_input", parts["g_input"]),
            ("g_input_centered", centered(parts["g_input"])),
            ("g_total_e", parts["g_total_e"]),
            ("g_total_e_centered", centered(parts["g_total_e"])),
            ("g_m", parts["g_m"]),
        ]:
            for metric, value in spectrum(matrix).items():
                row[f"{name}_{metric}"] = value
        row.update(family_stats(logits, data["targets"], data["families"]))
        return row


def run_one(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    data = build_data(condition, cfg.groups, cfg.reweight_alpha)
    e0, m0 = initialize(len(data["names"]), cfg.dim, seed)
    model = TiedConcatLM(e0, m0)
    rows: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, condition, seed, step))
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
            if key not in {"seed", "step"} and all(isinstance(item[key], (int, float)) for item in items)
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
    early_step = min(50, cfg.steps)
    u0 = get_agg(agg, "uniform_disjoint", 0)
    relabel0 = get_agg(agg, "uniform_disjoint_relabel", 0)
    shared0 = get_agg(agg, "shared_target", 0)
    prefix0 = get_agg(agg, "shared_input_prefix", 0)
    two_prefix0 = get_agg(agg, "shared_two_token_prefix", 0)
    u_final = get_agg(agg, "uniform_disjoint", cfg.steps)
    shared_final = get_agg(agg, "shared_target", cfg.steps)
    rew_final = get_agg(agg, "shared_target_reweight", cfg.steps)
    prefix_final = get_agg(agg, "shared_input_prefix", cfg.steps)

    checks = {
        "uniform_matches_relabel_null_at_step0": abs(
            u0["g_total_e_centered_top1_energy_mean"] - relabel0["g_total_e_centered_top1_energy_mean"]
        ) < 0.03,
        "shared_target_has_larger_initial_output_gradient_mode": shared0["g_output_centered_top1_energy_mean"]
        > u0["g_output_centered_top1_energy_mean"] * 1.35,
        "shared_target_gradient_aligns_context_mean": shared0["target_grad_context_cosine_mean"] > 0.8,
        "shared_target_gradient_precedes_embedding_concentration": (
            shared0["g_output_centered_top1_energy_mean"] - u0["g_output_centered_top1_energy_mean"]
        )
        > 20.0 * abs(shared0["e_centered_top1_energy_mean"] - u0["e_centered_top1_energy_mean"])
        and shared_final["e_centered_top1_energy_mean"] > get_agg(agg, "uniform_disjoint", cfg.steps)["e_centered_top1_energy_mean"] + 0.03,
        "shared_prefix_creates_raw_hidden_common_component": prefix0["probe_hidden_top1_energy_mean"]
        > u0["probe_hidden_top1_energy_mean"] * 2.0,
        "identical_two_token_prefix_is_hidden_common_but_not_solved": two_prefix0["probe_hidden_top1_energy_mean"] > 0.95
        and get_agg(agg, "shared_two_token_prefix", cfg.steps)["probe_accuracy_mean"] < 0.5,
        "uniform_final_less_concentrated_than_shared_target": u_final["e_centered_top1_energy_mean"]
        < shared_final["e_centered_top1_energy_mean"] - 0.03,
        "reweight_reduces_shared_target_concentration": rew_final["e_centered_top1_energy_mean"]
        < shared_final["e_centered_top1_energy_mean"],
        "reweight_preserves_probe_learning": rew_final["probe_accuracy_mean"] >= 0.99,
        "shared_input_prefix_not_equivalent_to_shared_target_output_mode": prefix0["g_output_centered_top1_energy_mean"]
        < shared0["g_output_centered_top1_energy_mean"] * 0.9,
    }
    status = "pass" if all(checks.values()) else "partial" if any(checks.values()) else "fail"
    key_numbers = {
        "step0_uniform_g_output_centered_top1": u0["g_output_centered_top1_energy_mean"],
        "step0_relabel_g_total_centered_top1": relabel0["g_total_e_centered_top1_energy_mean"],
        "step0_shared_target_g_output_centered_top1": shared0["g_output_centered_top1_energy_mean"],
        "step0_shared_target_grad_context_cosine": shared0["target_grad_context_cosine_mean"],
        "step0_uniform_probe_hidden_centered_top1": u0["probe_hidden_centered_top1_energy_mean"],
        "step0_shared_prefix_probe_hidden_centered_top1": prefix0["probe_hidden_centered_top1_energy_mean"],
        "step0_uniform_probe_hidden_raw_top1": u0["probe_hidden_top1_energy_mean"],
        "step0_shared_prefix_probe_hidden_raw_top1": prefix0["probe_hidden_top1_energy_mean"],
        "final_uniform_e_centered_top1": u_final["e_centered_top1_energy_mean"],
        "final_shared_target_e_centered_top1": shared_final["e_centered_top1_energy_mean"],
        "final_reweight_e_centered_top1": rew_final["e_centered_top1_energy_mean"],
        "final_prefix_e_centered_top1": prefix_final["e_centered_top1_energy_mean"],
    }
    return {"status": status, "checks": checks, "key_numbers": key_numbers}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(path: Path, agg: List[Dict[str, object]]) -> None:
    conditions = sorted(set(str(x["condition"]) for x in agg))
    panels = [
        ("g_output_centered_top1_energy_mean", "centered output-gradient top-1 energy"),
        ("g_total_e_centered_top1_energy_mean", "centered total-embedding-gradient top-1 energy"),
        ("probe_hidden_centered_top1_energy_mean", "probe hidden centered top-1 energy"),
        ("e_centered_top1_energy_mean", "embedding centered top-1 energy"),
        ("weighted_loss_mean", "weighted CE"),
        ("probe_accuracy_mean", "probe accuracy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (metric, title) in zip(axes.flat, panels):
        for condition in conditions:
            items = [x for x in agg if x["condition"] == condition and metric in x]
            if not items:
                continue
            xs = np.array([int(x["step"]) for x in items])
            ys = np.array([float(x[metric]) for x in items])
            std = np.array([float(x.get(metric.replace("_mean", "_std"), 0.0)) for x in items])
            ax.plot(xs, ys, label=condition, linewidth=2)
            ax.fill_between(xs, ys - std, ys + std, alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Stage 4: uniform disjoint data versus shared statistical features", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument("--reweight_alpha", type=float, default=0.5)
    cfg = Config(**vars(parser.parse_args()))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in cfg.seeds.split(",") if x.strip()]
    conditions = [
        "uniform_disjoint",
        "uniform_disjoint_relabel",
        "shared_target",
        "shared_target_reweight",
        "shared_input_prefix",
        "shared_two_token_prefix",
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
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
