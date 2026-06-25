#!/usr/bin/env python3
"""Causal Common-Residual Split (CRS) attention experiment.

Question:
  Can splitting each linear map Wh into common and residual branches reduce
  long-tail slow convergence in a single-layer residual causal attention model?

Model variants:
  dense:
      y = W h
  crs:
      y_k = alpha W_c P_{c,k} h_k + W_r (I - P_{c,k}) h_k
      P_{c,k} is estimated from the prefix mean of positions < k only.

The split is applied to Q/K/V/O projections.  No router softmax is used.
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
    steps: int = 3000
    lr: float = 0.02
    record_every: int = 20
    theta_deg: float = 12.0
    init_noise: float = 0.01
    alpha: float = 0.5
    data_mode: str = "cycle"


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


def build_data(cfg: Config) -> Dict[str, object]:
    """Build weighted synthetic sequences.

    Token 0 is shared K.  Each group contributes four causal sequences.  The
    target K appears after many different prefixes; K also appears as an input
    token in later positions.  Group A has Zipf weight 0.70; B/C/D are tails.
    """
    centers = {
        "A": np.eye(cfg.dim, dtype=np.float32)[0],
        "B": np.eye(cfg.dim, dtype=np.float32)[1],
        "C": np.eye(cfg.dim, dtype=np.float32)[2],
        "D": np.eye(cfg.dim, dtype=np.float32)[3],
    }
    e_rows: List[np.ndarray] = [np.ones(cfg.dim, dtype=np.float32) / math.sqrt(cfg.dim)]
    token_groups: List[str] = ["K"]
    group_ids: Dict[str, List[int]] = {}
    for group, center in centers.items():
        ids = []
        for off in [0.0, cfg.theta_deg, -cfg.theta_deg]:
            ids.append(len(e_rows))
            e_rows.append(rot(center, off, cfg.dim))
            token_groups.append(group)
        group_ids[group] = ids

    sequences: List[List[int]] = []
    seq_groups: List[str] = []
    for group in ["A", "B", "C", "D"]:
        i0, i1, i2 = group_ids[group]
        k = 0
        if cfg.data_mode == "cycle":
            # Conflict-free causal cycle. K is both frequent target and frequent
            # input, but no prefix maps to two different next tokens. This is
            # necessary when full tail accuracy is used as a convergence metric.
            group_sequences = [
                [i0, i1, k, i2, i0, i1, k, i2, i0],
            ]
        elif cfg.data_mode == "conflict":
            # Conflict data: the one-token prefix K maps to four different
            # group-specific next tokens across the dataset. This is closer to
            # natural language ambiguity. Full top-1 accuracy is not the right
            # target; CE/KL to the empirical conditional distribution is.
            group_sequences = [
                [i0, i1, k, i2, i0, i1],
                [i1, k, i2, i0, i1, k],
                [k, i2, i0, i1, k, i2],
                [i2, i0, i1, k, i2, i0],
            ]
        else:
            raise ValueError(f"unknown data_mode={cfg.data_mode}")
        for seq in group_sequences:
            sequences.append(seq)
            seq_groups.append(group)
    probs = {"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10}
    weights = torch.tensor([probs[g] / seq_groups.count(g) for g in seq_groups], dtype=torch.float32)
    weights = weights / weights.sum()
    tokens = torch.tensor([s[:-1] for s in sequences], dtype=torch.long)
    targets = torch.tensor([s[1:] for s in sequences], dtype=torch.long)
    target_groups = [[token_groups[t] for t in row] for row in targets.tolist()]
    prefix_targets: Dict[Tuple[int, ...], List[int]] = {}
    for seq in sequences:
        toks = seq[:-1]
        tgts = seq[1:]
        for pos, y in enumerate(tgts):
            prefix_targets.setdefault(tuple(toks[: pos + 1]), []).append(y)
    conflict_prefixes = {p: ys for p, ys in prefix_targets.items() if len(set(ys)) > 1}
    return {
        "E0": torch.tensor(np.stack(e_rows), dtype=torch.float32),
        "tokens": tokens,
        "targets": targets,
        "seq_weights": weights,
        "seq_groups": seq_groups,
        "target_groups": target_groups,
        "token_groups": token_groups,
        "conflict_prefixes": conflict_prefixes,
    }


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * self.weight


def causal_prefix_unit_mean(x: torch.Tensor) -> torch.Tensor:
    """Return u_{b,k} from mean of x[b, :k].  Position 0 returns zeros.

    x: [B, L, D]
    """
    bsz, seq_len, dim = x.shape
    cumsum = torch.cumsum(x.detach(), dim=1)
    zeros = torch.zeros(bsz, 1, dim, dtype=x.dtype, device=x.device)
    prefix_sum = torch.cat([zeros, cumsum[:, :-1]], dim=1)
    counts = torch.arange(seq_len, dtype=x.dtype, device=x.device).clamp_min(1.0).view(1, seq_len, 1)
    mean = prefix_sum / counts
    norm = mean.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-8, mean / norm.clamp_min(1e-8), torch.zeros_like(mean))


def split_common_residual(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = causal_prefix_unit_mean(x)
    coeff = (x * u).sum(dim=-1, keepdim=True)
    common = coeff * u
    residual = x - common
    return common, residual, u


class DenseLinear(torch.nn.Module):
    def __init__(self, dim: int, seed: int, init_noise: float):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        eye = torch.eye(dim, dtype=torch.float32) * 0.1
        self.W = torch.nn.Parameter(eye + init_noise * torch.randn(dim, dim, generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T

    def dense_equiv(self) -> torch.Tensor:
        return self.W.detach()

    def branch_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.W.detach(), self.W.detach()


class CRSLinear(torch.nn.Module):
    def __init__(self, dim: int, seed: int, init_noise: float, alpha: float):
        super().__init__()
        self.alpha = alpha
        gen = torch.Generator().manual_seed(seed)
        eye = torch.eye(dim, dtype=torch.float32) * 0.1
        base = eye + init_noise * torch.randn(dim, dim, generator=gen)
        self.Wc = torch.nn.Parameter(base.clone())
        self.Wr = torch.nn.Parameter(base.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        common, residual, _ = split_common_residual(x)
        return self.alpha * (common @ self.Wc.T) + residual @ self.Wr.T

    def dense_equiv(self) -> torch.Tensor:
        # There is no exact dense equivalent because P_c is data dependent.
        # For coarse spectrum reporting, use the sum of branch weights.
        return (self.alpha * self.Wc + self.Wr).detach()

    def branch_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Wc.detach(), self.Wr.detach()


class ToyAttention(torch.nn.Module):
    def __init__(self, e0: torch.Tensor, cfg: Config, seed: int, variant: str, alpha: float):
        super().__init__()
        self.variant = variant
        self.E = torch.nn.Parameter(e0.clone())
        self.norm = RMSNorm(cfg.dim)
        Linear = DenseLinear if variant == "dense" else CRSLinear
        kwargs = {} if variant == "dense" else {"alpha": alpha}
        self.q = Linear(cfg.dim, seed + 101, cfg.init_noise, **kwargs)
        self.k = Linear(cfg.dim, seed + 102, cfg.init_noise, **kwargs)
        self.v = Linear(cfg.dim, seed + 103, cfg.init_noise, **kwargs)
        self.o = Linear(cfg.dim, seed + 104, cfg.init_noise, **kwargs)
        self.scale = math.sqrt(cfg.dim)

    def forward(self, tokens: torch.Tensor, return_cache: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x0 = self.E[tokens]
        xn = self.norm(x0)
        q = self.q(xn)
        k = self.k(xn)
        v = self.v(xn)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
        seq_len = tokens.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device), diagonal=1)
        scores = scores.masked_fill(mask[None, :, :], -1e9)
        attn = torch.softmax(scores, dim=-1) @ v
        o = self.o(attn)
        h = x0 + o
        logits = h @ self.E.T
        cache = {
            "x0": x0.detach(),
            "xn": xn.detach(),
            "q": q.detach(),
            "k": k.detach(),
            "v": v.detach(),
            "attn": attn.detach(),
            "o": o.detach(),
            "h": h.detach(),
        }
        if self.variant != "dense":
            _, residual, u = split_common_residual(xn)
            cache["crs_u"] = u.detach()
            cache["crs_residual"] = residual.detach()
        return logits, cache


def centered(x: torch.Tensor) -> torch.Tensor:
    return x.float() - x.float().mean(dim=0, keepdim=True)


def svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(matrix.detach().float(), full_matrices=False)


def spectrum(matrix: torch.Tensor) -> Dict[str, float]:
    _, s, _ = svd(matrix)
    e = s.square()
    total = e.sum().clamp_min(1e-12)
    return {
        "sigma1": float(s[0]),
        "sigma4": float(s[min(3, len(s) - 1)]),
        "sigma1_over_sigma4": float(s[0] / s[min(3, len(s) - 1)].clamp_min(1e-12)),
        "top1_energy": float(e[0] / total),
    }


def top_right(matrix: torch.Tensor) -> torch.Tensor:
    _, _, vh = svd(matrix)
    return vh[0]


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    return float(((a @ b) / denom).square())


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


def pc1(x: torch.Tensor) -> torch.Tensor:
    _, _, vh = svd(centered(flatten(x)))
    return vh[0]


def loss_metrics(model: ToyAttention, data: Dict[str, object], reweight: bool = False) -> Dict[str, float]:
    logits, _ = model(data["tokens"])
    targets = data["targets"]
    bsz, seq_len = targets.shape
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view(bsz, seq_len)
    pred = logits.argmax(dim=-1)
    seq_w = data["seq_weights"].view(-1, 1)
    if reweight:
        token_groups = data["token_groups"]
        freq: Dict[str, int] = {}
        for row in targets.tolist():
            for tok in row:
                freq[token_groups[tok]] = freq.get(token_groups[tok], 0) + 1
        target_w = torch.tensor([[1.0 / math.sqrt(freq[token_groups[tok]]) for tok in row] for row in targets.tolist()])
        target_w = target_w / target_w.mean()
    else:
        target_w = torch.ones_like(losses)
    weighted = losses * seq_w * target_w

    groups = data["target_groups"]
    common_mask = torch.tensor([[g == "K" or g == "A" for g in row] for row in groups], dtype=torch.bool)
    tail_mask = torch.tensor([[g in {"B", "C", "D"} for g in row] for row in groups], dtype=torch.bool)
    k_mask = torch.tensor([[g == "K" for g in row] for row in groups], dtype=torch.bool)
    out = {
        "loss": float(weighted.sum() / (seq_w * target_w).sum().clamp_min(1e-12)),
        "accuracy": float((pred == targets).float().mean()),
        "common_loss": float(losses[common_mask].mean()),
        "tail_loss": float(losses[tail_mask].mean()),
        "k_loss": float(losses[k_mask].mean()),
        "common_accuracy": float((pred[common_mask] == targets[common_mask]).float().mean()),
        "tail_accuracy": float((pred[tail_mask] == targets[tail_mask]).float().mean()),
        "k_accuracy": float((pred[k_mask] == targets[k_mask]).float().mean()),
    }
    tmp = logits.detach().clone()
    target_logits = tmp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    tmp.scatter_(-1, targets.unsqueeze(-1), -float("inf"))
    margins = target_logits - tmp.max(dim=-1).values
    out["tail_margin"] = float(margins[tail_mask].mean())
    out["common_margin"] = float(margins[common_mask].mean())
    out["k_margin"] = float(margins[k_mask].mean())

    conflict_prefixes: Dict[Tuple[int, ...], List[int]] = data.get("conflict_prefixes", {})
    if conflict_prefixes:
        log_probs = F.log_softmax(logits.detach(), dim=-1)
        probs = log_probs.exp()
        conflict_ces = []
        conflict_kls = []
        conflict_entropies = []
        conflict_acc_ceilings = []
        for prefix, ys in conflict_prefixes.items():
            positions: List[Tuple[int, int]] = []
            for b, row in enumerate(data["tokens"].tolist()):
                for pos in range(len(row)):
                    if tuple(row[: pos + 1]) == prefix:
                        positions.append((b, pos))
            if not positions:
                continue
            counts: Dict[int, int] = {}
            for y in ys:
                counts[y] = counts.get(y, 0) + 1
            total = float(sum(counts.values()))
            p_items = [(y, c / total) for y, c in counts.items()]
            entropy = -sum(p * math.log(max(p, 1e-12)) for _, p in p_items)
            acc_ceiling = max(p for _, p in p_items)
            # Use the average predicted distribution over identical-prefix
            # occurrences. In a strictly causal deterministic model these should
            # match; averaging makes the diagnostic robust to implementation
            # details and numerical differences.
            q = torch.stack([probs[b, pos] for b, pos in positions], dim=0).mean(dim=0)
            ce = -sum(p * float(torch.log(q[y].clamp_min(1e-12))) for y, p in p_items)
            conflict_entropies.append(entropy)
            conflict_ces.append(ce)
            conflict_kls.append(ce - entropy)
            conflict_acc_ceilings.append(acc_ceiling)
        out["conflict_ce"] = float(np.mean(conflict_ces)) if conflict_ces else 0.0
        out["conflict_kl_to_bayes"] = float(np.mean(conflict_kls)) if conflict_kls else 0.0
        out["conflict_bayes_ce"] = float(np.mean(conflict_entropies)) if conflict_entropies else 0.0
        out["conflict_top1_ceiling"] = float(np.mean(conflict_acc_ceilings)) if conflict_acc_ceilings else 1.0
    else:
        out["conflict_ce"] = 0.0
        out["conflict_kl_to_bayes"] = 0.0
        out["conflict_bayes_ce"] = 0.0
        out["conflict_top1_ceiling"] = 1.0
    return out


def train_loss(model: ToyAttention, data: Dict[str, object], reweight: bool) -> torch.Tensor:
    logits, _ = model(data["tokens"])
    targets = data["targets"]
    bsz, seq_len = targets.shape
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view(bsz, seq_len)
    seq_w = data["seq_weights"].view(-1, 1)
    if reweight:
        token_groups = data["token_groups"]
        freq: Dict[str, int] = {}
        for row in targets.tolist():
            for tok in row:
                freq[token_groups[tok]] = freq.get(token_groups[tok], 0) + 1
        target_w = torch.tensor([[1.0 / math.sqrt(freq[token_groups[tok]]) for tok in row] for row in targets.tolist()])
        target_w = target_w / target_w.mean()
    else:
        target_w = torch.ones_like(losses)
    return (losses * seq_w * target_w).sum()


def first_stable_step(rows: List[Dict[str, object]], key: str, threshold: float, start_step: int = 0) -> int | None:
    ordered = sorted(rows, key=lambda r: int(r["step"]))
    for row in ordered:
        if int(row["step"]) < start_step:
            continue
        if all(float(r[key]) >= threshold for r in ordered if int(r["step"]) >= int(row["step"])):
            return int(row["step"])
    return None


def first_stable_below_step(rows: List[Dict[str, object]], key: str, threshold: float, start_step: int = 0) -> int | None:
    ordered = sorted(rows, key=lambda r: int(r["step"]))
    for row in ordered:
        if int(row["step"]) < start_step:
            continue
        if all(float(r[key]) <= threshold for r in ordered if int(r["step"]) >= int(row["step"])):
            return int(row["step"])
    return None


def measure(model: ToyAttention, data: Dict[str, object], cfg: Config, variant: str, seed: int, step: int, reweight: bool) -> Dict[str, object]:
    row: Dict[str, object] = {"variant": variant, "seed": seed, "step": step, **loss_metrics(model, data, reweight=reweight)}
    _, cache = model(data["tokens"], return_cache=True)
    x_pc1 = pc1(cache["xn"])
    h_pc1 = pc1(cache["h"])
    row["repr_top1_energy"] = spectrum(centered(flatten(cache["h"])))["top1_energy"]
    if "crs_residual" in cache:
        row["residual_repr_top1_energy"] = spectrum(centered(flatten(cache["crs_residual"])))["top1_energy"]
        u = cache["crs_u"]
        x = cache["xn"]
        common_fraction = ((x * u).sum(dim=-1).square() / x.square().sum(dim=-1).clamp_min(1e-12)).mean()
        residual_fraction = (cache["crs_residual"].square().sum(dim=-1) / x.square().sum(dim=-1).clamp_min(1e-12)).mean()
        row["crs_common_input_energy_fraction"] = float(common_fraction)
        row["crs_residual_input_energy_fraction"] = float(residual_fraction)
    else:
        row["residual_repr_top1_energy"] = 0.0
        row["crs_common_input_energy_fraction"] = 0.0
        row["crs_residual_input_energy_fraction"] = 0.0

    for name, layer in [("q", model.q), ("k", model.k), ("v", model.v), ("o", model.o)]:
        Wc, Wr = layer.branch_weights()
        for branch, W in [("common", Wc), ("residual", Wr), ("combined", layer.dense_equiv())]:
            spec = spectrum(W)
            for k, v in spec.items():
                row[f"{name}_{branch}_{k}"] = v
            row[f"{name}_{branch}_align_x_pc1"] = sqcos(top_right(W), x_pc1)
            row[f"{name}_{branch}_align_h_pc1"] = sqcos(top_right(W), h_pc1)
    Bqk = model.q.dense_equiv().T @ model.k.dense_equiv()
    for k, v in spectrum(Bqk).items():
        row[f"Bqk_combined_{k}"] = v
    row["Bqk_combined_align_x_pc1"] = sqcos(top_right(Bqk), x_pc1)
    return row


def train_one(cfg: Config, variant: str, seed: int) -> List[Dict[str, object]]:
    set_seed(seed)
    data = build_data(cfg)
    if variant == "dense_reweight":
        model_variant = "dense"
        reweight = True
        alpha = 1.0
    elif variant == "crs_alpha1":
        model_variant = "crs"
        reweight = False
        alpha = 1.0
    elif variant == "crs_alpha05":
        model_variant = "crs"
        reweight = False
        alpha = cfg.alpha
    elif variant == "dense":
        model_variant = "dense"
        reweight = False
        alpha = 1.0
    else:
        raise ValueError(variant)
    model = ToyAttention(data["E0"], cfg, seed, model_variant, alpha)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    rows: List[Dict[str, object]] = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, cfg, variant, seed, step, reweight))
        if step == cfg.steps:
            break
        opt.zero_grad(set_to_none=True)
        loss = train_loss(model, data, reweight)
        loss.backward()
        opt.step()
    return rows


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant"]), int(row["step"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (variant, step), items in sorted(grouped.items()):
        agg: Dict[str, object] = {"variant": variant, "step": step, "num_seeds": len(items)}
        keys = sorted(
            k
            for k in set.intersection(*(set(x) for x in items))
            if k not in {"variant", "seed", "step"} and all(isinstance(x[k], (int, float)) for x in items)
        )
        for key in keys:
            vals = np.array([float(x[key]) for x in items], dtype=np.float64)
            agg[f"{key}_mean"] = float(vals.mean())
            agg[f"{key}_std"] = float(vals.std(ddof=0))
        out.append(agg)
    return out


def rows_for(agg: List[Dict[str, object]], variant: str) -> List[Dict[str, object]]:
    return sorted([x for x in agg if x["variant"] == variant], key=lambda x: int(x["step"]))


def summarize(agg: List[Dict[str, object]], history: List[Dict[str, object]]) -> Dict[str, object]:
    variants = ["dense", "dense_reweight", "crs_alpha1", "crs_alpha05"]
    summary: Dict[str, object] = {"variants": {}}
    for variant in variants:
        ar = rows_for(agg, variant)
        final = ar[-1]
        raw_rows = [r for r in history if r["variant"] == variant]
        per_seed_steps = []
        per_seed_conflict_kl005_steps = []
        per_seed_conflict_kl001_steps = []
        for seed in sorted(set(int(r["seed"]) for r in raw_rows)):
            sr = [r for r in raw_rows if int(r["seed"]) == seed]
            per_seed_steps.append(first_stable_step(sr, "tail_accuracy", 1.0))
            per_seed_conflict_kl005_steps.append(first_stable_below_step(sr, "conflict_kl_to_bayes", 0.05))
            per_seed_conflict_kl001_steps.append(first_stable_below_step(sr, "conflict_kl_to_bayes", 0.01))
        stable_vals = [x for x in per_seed_steps if x is not None]
        stable_kl005_vals = [x for x in per_seed_conflict_kl005_steps if x is not None]
        stable_kl001_vals = [x for x in per_seed_conflict_kl001_steps if x is not None]
        summary["variants"][variant] = {
            "final_loss": final["loss_mean"],
            "final_tail_loss": final["tail_loss_mean"],
            "final_common_loss": final["common_loss_mean"],
            "final_tail_accuracy": final["tail_accuracy_mean"],
            "final_common_accuracy": final["common_accuracy_mean"],
            "final_tail_margin": final["tail_margin_mean"],
            "final_common_margin": final["common_margin_mean"],
            "mean_first_stable_tail_acc_step": float(np.mean(stable_vals)) if stable_vals else None,
            "num_seeds_reached_stable_tail_acc": len(stable_vals),
            "final_conflict_ce": final["conflict_ce_mean"],
            "final_conflict_kl_to_bayes": final["conflict_kl_to_bayes_mean"],
            "final_conflict_bayes_ce": final["conflict_bayes_ce_mean"],
            "final_conflict_top1_ceiling": final["conflict_top1_ceiling_mean"],
            "mean_first_stable_conflict_kl_le_0p05_step": float(np.mean(stable_kl005_vals)) if stable_kl005_vals else None,
            "num_seeds_reached_conflict_kl_le_0p05": len(stable_kl005_vals),
            "mean_first_stable_conflict_kl_le_0p01_step": float(np.mean(stable_kl001_vals)) if stable_kl001_vals else None,
            "num_seeds_reached_conflict_kl_le_0p01": len(stable_kl001_vals),
            "final_Bqk_top1_energy": final["Bqk_combined_top1_energy_mean"],
            "final_Bqk_sigma1_over_sigma4": final["Bqk_combined_sigma1_over_sigma4_mean"],
            "final_q_residual_top1_energy": final["q_residual_top1_energy_mean"],
            "final_q_common_top1_energy": final["q_common_top1_energy_mean"],
            "final_crs_common_input_energy_fraction": final["crs_common_input_energy_fraction_mean"],
            "final_crs_residual_input_energy_fraction": final["crs_residual_input_energy_fraction_mean"],
        }
    dense_step = summary["variants"]["dense"]["mean_first_stable_tail_acc_step"]
    summary["main_checks"] = {
        "crs_alpha1_tail_faster_than_dense": (
            dense_step is not None
            and summary["variants"]["crs_alpha1"]["mean_first_stable_tail_acc_step"] is not None
            and summary["variants"]["crs_alpha1"]["mean_first_stable_tail_acc_step"] < dense_step
        ),
        "crs_alpha05_tail_faster_than_dense": (
            dense_step is not None
            and summary["variants"]["crs_alpha05"]["mean_first_stable_tail_acc_step"] is not None
            and summary["variants"]["crs_alpha05"]["mean_first_stable_tail_acc_step"] < dense_step
        ),
        "crs_alpha05_final_tail_loss_lower_than_dense": summary["variants"]["crs_alpha05"]["final_tail_loss"]
        < summary["variants"]["dense"]["final_tail_loss"],
        "reweight_reference_tail_faster_than_dense": (
            dense_step is not None
            and summary["variants"]["dense_reweight"]["mean_first_stable_tail_acc_step"] is not None
            and summary["variants"]["dense_reweight"]["mean_first_stable_tail_acc_step"] < dense_step
        ),
        "crs_alpha1_conflict_kl005_faster_than_dense": (
            summary["variants"]["dense"]["mean_first_stable_conflict_kl_le_0p05_step"] is not None
            and summary["variants"]["crs_alpha1"]["mean_first_stable_conflict_kl_le_0p05_step"] is not None
            and summary["variants"]["crs_alpha1"]["mean_first_stable_conflict_kl_le_0p05_step"]
            < summary["variants"]["dense"]["mean_first_stable_conflict_kl_le_0p05_step"]
        ),
        "crs_alpha05_conflict_kl005_faster_than_dense": (
            summary["variants"]["dense"]["mean_first_stable_conflict_kl_le_0p05_step"] is not None
            and summary["variants"]["crs_alpha05"]["mean_first_stable_conflict_kl_le_0p05_step"] is not None
            and summary["variants"]["crs_alpha05"]["mean_first_stable_conflict_kl_le_0p05_step"]
            < summary["variants"]["dense"]["mean_first_stable_conflict_kl_le_0p05_step"]
        ),
    }
    return summary


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, agg: List[Dict[str, object]]) -> None:
    variants = ["dense", "dense_reweight", "crs_alpha1", "crs_alpha05"]
    panels = [
        ("tail_loss_mean", "Tail CE loss"),
        ("tail_accuracy_mean", "Tail accuracy"),
        ("conflict_kl_to_bayes_mean", "Conflict KL to Bayes distribution"),
        ("conflict_ce_mean", "Conflict CE"),
        ("tail_margin_mean", "Tail margin"),
        ("common_margin_mean", "Common margin"),
        ("Bqk_combined_top1_energy_mean", "Bqk top-1 energy"),
        ("Bqk_combined_sigma1_over_sigma4_mean", "Bqk sigma1/sigma4"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, (metric, title) in zip(axes.flat, panels):
        for variant in variants:
            rs = rows_for(agg, variant)
            ax.plot([int(r["step"]) for r in rs], [float(r[metric]) for r in rs], label=variant, linewidth=1.6)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xlabel("step")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("CRS split single-layer residual attention on Zipf/shared-K synthetic data")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--record_every", type=int, default=20)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument("--init_noise", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--data_mode", choices=["cycle", "conflict"], default="cycle")
    cfg = Config(**vars(parser.parse_args()))
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for variant in ["dense", "dense_reweight", "crs_alpha1", "crs_alpha05"]:
        for seed in [int(x) for x in cfg.seeds.split(",") if x.strip()]:
            print(f"running variant={variant} seed={seed}", flush=True)
            rows.extend(train_one(cfg, variant, seed))
    agg = aggregate(rows)
    summary = summarize(agg, rows)
    write_csv(outdir / "history.csv", rows)
    write_csv(outdir / "aggregate.csv", agg)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(outdir / "crs_split_attention.png", agg)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
