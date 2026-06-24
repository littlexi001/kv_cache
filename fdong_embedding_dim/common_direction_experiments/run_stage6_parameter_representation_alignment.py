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
    dim: int = 4
    steps: int = 1200
    lr: float = 0.03
    record_every: int = 20
    theta_deg: float = 12.0
    init_noise: float = 0.005
    residual_alpha: float = 0.0
    use_o_proj: bool = False


class AttnLM(torch.nn.Module):
    def __init__(self, e0: torch.Tensor, dim: int, seed: int, init_noise: float, residual_alpha: float, use_o_proj: bool):
        super().__init__()
        self.dim = dim
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

    def forward(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        e_override: torch.Tensor = None,
        wq_override: torch.Tensor = None,
        wk_override: torch.Tensor = None,
        wv_override: torch.Tensor = None,
        bqk_override: torch.Tensor = None,
    ) -> torch.Tensor:
        E = self.E if e_override is None else e_override
        Wq = self.Wq if wq_override is None else wq_override
        Wk = self.Wk if wk_override is None else wk_override
        Wv = self.Wv if wv_override is None else wv_override
        h1 = E[c1]
        h2 = E[c2]
        V = torch.stack([h1 @ Wv.T, h2 @ Wv.T], dim=1)
        if bqk_override is None:
            K = torch.stack([h1 @ Wk.T, h2 @ Wk.T], dim=1)
            Q = (h2 @ Wq.T).unsqueeze(1)
            scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        else:
            s1 = torch.einsum("bd,df,bf->b", h2, bqk_override, h1)
            s2 = torch.einsum("bd,df,bf->b", h2, bqk_override, h2)
            scores = torch.stack([s1, s2], dim=-1).unsqueeze(1) / self.scale
        attn_out = torch.bmm(F.softmax(scores, dim=-1), V).squeeze(1)
        if self.use_o_proj:
            attn_out = attn_out @ self.Wo.T
        out = attn_out + self.residual_alpha * h2
        return out @ E.T


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
    groups = list(centers)
    with_k = condition.startswith("withK")
    e_rows: List[np.ndarray] = []
    token_groups: List[str] = []
    if with_k:
        e_rows.append(np.ones(cfg.dim, dtype=np.float32) / math.sqrt(cfg.dim))
        token_groups.append("K")
    group_ids: Dict[str, List[int]] = {}
    for group in groups:
        ids = []
        for off in [0.0, cfg.theta_deg, -cfg.theta_deg]:
            ids.append(len(e_rows))
            e_rows.append(rot(centers[group], off, cfg.dim))
            token_groups.append(group)
        group_ids[group] = ids
    e0 = torch.tensor(np.stack(e_rows), dtype=torch.float32)

    c1: List[int] = []
    c2: List[int] = []
    targets: List[int] = []
    example_groups: List[str] = []
    families: List[str] = []
    for group in groups:
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
            example_groups.append(group)
            families.append(family)

    if condition.endswith("zipf"):
        probs = {"A": 0.70, "B": 0.10, "C": 0.10, "D": 0.10}
    else:
        probs = {g: 0.25 for g in groups}
    weights = torch.tensor(
        [probs[g] / sum(1 for x in example_groups if x == g) for g in example_groups],
        dtype=torch.float32,
    )
    weights = weights / weights.sum()
    return {
        "E0": e0,
        "c1": torch.tensor(c1, dtype=torch.long),
        "c2": torch.tensor(c2, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "weights": weights,
        "groups": example_groups,
        "families": families,
        "token_groups": token_groups,
        "with_k": with_k,
    }


def centered(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def weighted_centered(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.float().reshape(-1, 1)
    total = w.sum().clamp_min(1e-12)
    mu = (x.float() * w).sum(dim=0, keepdim=True) / total
    return (x.float() - mu) * torch.sqrt(w / total)


def svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(matrix.detach().float(), full_matrices=False)
    return u, s, vh


def spectrum(matrix: torch.Tensor) -> Dict[str, float]:
    _, s, _ = svd(matrix)
    energy = s.square()
    total = float(energy.sum())
    if total <= 1e-12:
        return {"top1_energy": 0.0, "sigma1": 0.0, "sigma2": 0.0, "sigma1_over_sigma2": 0.0}
    return {
        "top1_energy": float(energy[0] / total),
        "sigma1": float(s[0]),
        "sigma2": float(s[1]) if len(s) > 1 else 0.0,
        "sigma1_over_sigma2": float(s[0] / s[1].clamp_min(1e-12)) if len(s) > 1 else 0.0,
    }


def effective_rank_from_singular_values(s: torch.Tensor) -> float:
    energy = s.square().float()
    total = energy.sum()
    if float(total) <= 1e-12:
        return 0.0
    p = energy / total
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def top_right(matrix: torch.Tensor) -> torch.Tensor:
    _, _, vh = svd(matrix)
    return vh[0]


def top_left(matrix: torch.Tensor) -> torch.Tensor:
    u, _, _ = svd(matrix)
    return u[:, 0]


def sqcos(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) <= 1e-12:
        return 0.0
    value = torch.dot(a, b) / denom
    return float(value.square())


def weighted_direction_variance_fraction(
    x: torch.Tensor,
    weights: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Fraction of weighted variance along a direction contributed by a subset."""
    w = weights.float()
    total_w = w.sum().clamp_min(1e-12)
    mu = (x.float() * w[:, None]).sum(dim=0) / total_w
    d = direction.float() / direction.float().norm().clamp_min(1e-12)
    scores = ((x.float() - mu) @ d).square() * w
    total = scores.sum().clamp_min(1e-12)
    return float(scores[mask].sum() / total)


def loss_metrics(model: AttnLM, data: Dict[str, object], **overrides) -> Dict[str, float]:
    logits = model(data["c1"], data["c2"], **overrides)
    targets = data["targets"]
    losses = F.cross_entropy(logits, targets, reduction="none")
    pred = logits.argmax(dim=-1)
    rows = logits.detach().clone()
    target_logits = rows[torch.arange(len(targets)), targets]
    rows[torch.arange(len(targets)), targets] = -float("inf")
    margins = target_logits - rows.max(dim=-1).values
    masks = {
        "common": torch.tensor([g == "A" for g in data["groups"]], dtype=torch.bool),
        "tail": torch.tensor([g != "A" for g in data["groups"]], dtype=torch.bool),
        "internal": torch.tensor(["internal" in f for f in data["families"]], dtype=torch.bool),
        "k_related": torch.tensor(["K" in f for f in data["families"]], dtype=torch.bool),
    }
    out = {"loss": float((losses * data["weights"]).sum())}
    for name, mask in masks.items():
        if bool(mask.any()):
            out[f"{name}_loss"] = float(losses[mask].mean())
            out[f"{name}_accuracy"] = float((pred[mask] == targets[mask]).float().mean())
            out[f"{name}_margin"] = float(margins[mask].mean())
        else:
            out[f"{name}_loss"] = 0.0
            out[f"{name}_accuracy"] = 0.0
            out[f"{name}_margin"] = 0.0
    return out


def compute_gradients(model: AttnLM, data: Dict[str, object], mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    if mask is None:
        c1, c2, targets, weights = data["c1"], data["c2"], data["targets"], data["weights"]
    else:
        c1, c2, targets, weights = data["c1"][mask], data["c2"][mask], data["targets"][mask], data["weights"][mask]
    logits = model(c1, c2)
    loss = (F.cross_entropy(logits, targets, reduction="none") * weights).sum()
    loss.backward()
    grads = {
        "E": model.E.grad.detach().clone(),
        "Wq": model.Wq.grad.detach().clone(),
        "Wk": model.Wk.grad.detach().clone(),
        "Wv": model.Wv.grad.detach().clone(),
        "Wo": model.Wo.grad.detach().clone() if model.Wo.grad is not None else torch.zeros_like(model.Wo),
    }
    model.zero_grad(set_to_none=True)
    return grads


def sigma1_growth_contribution(matrix: torch.Tensor, grad: torch.Tensor, lr: float) -> float:
    u, _, vh = svd(matrix)
    # Parameter update is M <- M - lr * grad. Positive value means sigma1 increases.
    return float(-lr * (u[:, 0] @ grad @ vh[0]))


def ablate_top1(matrix: torch.Tensor) -> torch.Tensor:
    u, s, vh = svd(matrix)
    return matrix.detach() - s[0] * torch.outer(u[:, 0], vh[0])


def centroid_matrix(vectors: torch.Tensor, groups: List[str], selected) -> torch.Tensor:
    rows = []
    for group in selected:
        mask = torch.tensor([g == group for g in groups], dtype=torch.bool)
        if bool(mask.any()):
            rows.append(vectors[mask].mean(dim=0))
    return torch.stack(rows) if rows else torch.zeros(0, vectors.shape[-1])


def projection_mass(vectors: torch.Tensor, basis: torch.Tensor) -> float:
    if vectors.numel() == 0:
        return 0.0
    norms = vectors.norm(dim=-1).clamp_min(1e-12)
    proj = vectors @ basis.T
    return float((proj.square().sum(dim=-1) / norms.square()).mean())


def measure(model: AttnLM, data: Dict[str, object], cfg: Config, condition: str, seed: int, step: int) -> Dict[str, object]:
    with torch.no_grad():
        E = model.E.detach()
        Wq = model.Wq.detach()
        Wk = model.Wk.detach()
        Wv = model.Wv.detach()
        Bqk = Wq.T @ Wk
        h1 = E[data["c1"]]
        h2 = E[data["c2"]]
        weights = data["weights"]
        x_query = weighted_centered(h2, weights)
        x_key = weighted_centered(torch.cat([h1, h2], dim=0), torch.cat([weights, weights], dim=0))
        qtok = E @ Wq.T
        ktok = E @ Wk.T
        vtok = E @ Wv.T
        out = {
            "condition": condition,
            "seed": seed,
            "step": step,
            **loss_metrics(model, data),
        }
        matrices = {
            "E_centered": centered(E),
            "Wq": Wq,
            "Wk": Wk,
            "Wv": Wv,
            "Bqk": Bqk,
            "X_query_weighted_centered": x_query,
            "X_key_weighted_centered": x_key,
            "Q_tok_centered": centered(qtok),
            "K_tok_centered": centered(ktok),
            "V_tok_centered": centered(vtok),
        }
        for name, matrix in matrices.items():
            spec = spectrum(matrix)
            for key, value in spec.items():
                out[f"{name}_{key}"] = value
            _, s, _ = svd(matrix)
            out[f"{name}_effective_rank"] = effective_rank_from_singular_values(s)

        rE = top_right(centered(E))
        rQ = top_right(centered(qtok))
        rK = top_right(centered(ktok))
        rV = top_right(centered(vtok))
        rXq = top_right(x_query)
        rXk = top_right(x_key)

        activation_masks = {
            "common": torch.tensor([g == "A" for g in data["groups"]], dtype=torch.bool),
            "tail": torch.tensor([g != "A" for g in data["groups"]], dtype=torch.bool),
            "k_related": torch.tensor(["K" in f for f in data["families"]], dtype=torch.bool),
            "internal": torch.tensor(["internal" in f for f in data["families"]], dtype=torch.bool),
            "to_K": torch.tensor([f == "to_K" for f in data["families"]], dtype=torch.bool),
            "from_K": torch.tensor([f.startswith("from_K") for f in data["families"]], dtype=torch.bool),
        }
        key_x = torch.cat([h1, h2], dim=0)
        key_w = torch.cat([weights, weights], dim=0)
        for split, mask in activation_masks.items():
            if bool(mask.any()):
                out[f"X_query_top1_varfrac_{split}"] = weighted_direction_variance_fraction(h2, weights, rXq, mask)
                out[f"X_key_top1_varfrac_{split}"] = weighted_direction_variance_fraction(
                    key_x, key_w, rXk, torch.cat([mask, mask], dim=0)
                )
            else:
                out[f"X_query_top1_varfrac_{split}"] = 0.0
                out[f"X_key_top1_varfrac_{split}"] = 0.0

        for pname, matrix in [("Wq", Wq), ("Wk", Wk), ("Wv", Wv), ("Bqk", Bqk)]:
            u1 = top_left(matrix)
            v1 = top_right(matrix)
            out[f"align_rE_to_{pname}_input_sqcos"] = sqcos(rE, v1)
            out[f"align_rXq_to_{pname}_input_sqcos"] = sqcos(rXq, v1)
            out[f"align_rXk_to_{pname}_input_sqcos"] = sqcos(rXk, v1)
            out[f"align_rQ_to_{pname}_output_sqcos"] = sqcos(rQ, u1)
            out[f"align_rK_to_{pname}_output_sqcos"] = sqcos(rK, u1)
            out[f"align_rV_to_{pname}_output_sqcos"] = sqcos(rV, u1)

        example_groups = data["groups"]
        common_groups = ["A"]
        tail_groups = ["B", "C", "D"]
        example_reprs = {
            "query_input_h2": h2,
            "key_input_h1h2": torch.cat([h1, h2], dim=0),
        }
        key_groups = example_groups + example_groups
        for pname, matrix in [("Wq", Wq), ("Wk", Wk), ("Wv", Wv), ("Bqk", Bqk)]:
            basis = top_right(matrix).reshape(1, -1)
            common_q = centroid_matrix(example_reprs["query_input_h2"], example_groups, common_groups)
            tail_q = centroid_matrix(example_reprs["query_input_h2"], example_groups, tail_groups)
            common_k = centroid_matrix(example_reprs["key_input_h1h2"], key_groups, common_groups)
            tail_k = centroid_matrix(example_reprs["key_input_h1h2"], key_groups, tail_groups)
            out[f"{pname}_common_query_input_top1_mass"] = projection_mass(common_q, basis)
            out[f"{pname}_tail_query_input_top1_mass"] = projection_mass(tail_q, basis)
            out[f"{pname}_common_key_input_top1_mass"] = projection_mass(common_k, basis)
            out[f"{pname}_tail_key_input_top1_mass"] = projection_mass(tail_k, basis)

        before = loss_metrics(model, data)
        ablations = {
            "E_top1": {"e_override": ablate_top1(E)},
            "Wq_top1": {"wq_override": ablate_top1(Wq)},
            "Wk_top1": {"wk_override": ablate_top1(Wk)},
            "Wv_top1": {"wv_override": ablate_top1(Wv)},
            "Bqk_top1": {"bqk_override": ablate_top1(Bqk)},
        }
        for ab_name, kwargs in ablations.items():
            after = loss_metrics(model, data, **kwargs)
            for split in ["common", "tail", "internal", "k_related"]:
                out[f"ablate_{ab_name}_{split}_loss_delta"] = after[f"{split}_loss"] - before[f"{split}_loss"]
                out[f"ablate_{ab_name}_{split}_margin_delta"] = after[f"{split}_margin"] - before[f"{split}_margin"]

    full_grads = compute_gradients(model, data)
    family_masks = {
        "common": torch.tensor([g == "A" for g in data["groups"]], dtype=torch.bool),
        "tail": torch.tensor([g != "A" for g in data["groups"]], dtype=torch.bool),
        "k_related": torch.tensor(["K" in f for f in data["families"]], dtype=torch.bool),
        "internal": torch.tensor(["internal" in f for f in data["families"]], dtype=torch.bool),
    }
    for matrix_name, matrix in [("E_centered", centered(model.E.detach())), ("Wq", model.Wq.detach()), ("Wk", model.Wk.detach()), ("Wv", model.Wv.detach())]:
        grad_key = "E" if matrix_name == "E_centered" else matrix_name
        grad = centered(full_grads[grad_key]) if matrix_name == "E_centered" else full_grads[grad_key]
        out[f"{matrix_name}_full_grad_sigma1_growth"] = sigma1_growth_contribution(matrix, grad, cfg.lr)
        for family, mask in family_masks.items():
            if bool(mask.any()):
                grads = compute_gradients(model, data, mask)
                g = centered(grads[grad_key]) if matrix_name == "E_centered" else grads[grad_key]
                out[f"{matrix_name}_{family}_grad_sigma1_growth"] = sigma1_growth_contribution(matrix, g, cfg.lr)
            else:
                out[f"{matrix_name}_{family}_grad_sigma1_growth"] = 0.0
    return out


def train_one(cfg: Config, condition: str, seed: int) -> List[Dict[str, object]]:
    set_seed(seed)
    data = build_init_and_data(cfg, condition)
    model = AttnLM(data["E0"], cfg.dim, seed, cfg.init_noise, cfg.residual_alpha, cfg.use_o_proj)
    rows = []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0 or step == cfg.steps:
            rows.append(measure(model, data, cfg, condition, seed, step))
        if step == cfg.steps:
            break
        grads = compute_gradients(model, data)
        with torch.no_grad():
            model.E -= cfg.lr * grads["E"]
            model.Wq -= cfg.lr * grads["Wq"]
            model.Wk -= cfg.lr * grads["Wk"]
            model.Wv -= cfg.lr * grads["Wv"]
            if cfg.use_o_proj:
                model.Wo -= cfg.lr * grads["Wo"]
    return rows


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["condition"], int(row["step"])), []).append(row)
    out_rows = []
    for (condition, step), items in sorted(groups.items()):
        out = {"condition": condition, "step": step, "num_seeds": len(items)}
        keys = sorted(
            k for k in set.intersection(*(set(x) for x in items))
            if k not in {"condition", "seed", "step"} and all(isinstance(x[k], (int, float)) for x in items)
        )
        for key in keys:
            vals = np.array([float(x[key]) for x in items], dtype=np.float64)
            out[f"{key}_mean"] = float(vals.mean())
            out[f"{key}_std"] = float(vals.std(ddof=0))
        out_rows.append(out)
    return out_rows


def trajectory(agg: List[Dict[str, object]], condition: str) -> List[Dict[str, object]]:
    return sorted([x for x in agg if x["condition"] == condition], key=lambda x: int(x["step"]))


def summarize(cfg: Config, agg: List[Dict[str, object]]) -> Dict[str, object]:
    final = {c: trajectory(agg, c)[-1] for c in ["noK_uniform", "withK_uniform", "withK_zipf"]}
    early = {c: trajectory(agg, c)[min(5, len(trajectory(agg, c)) - 1)] for c in ["noK_uniform", "withK_uniform", "withK_zipf"]}
    checks = {
        "withK_zipf_increases_output_embedding_concentration": final["withK_zipf"]["E_centered_top1_energy_mean"]
        > final["noK_uniform"]["E_centered_top1_energy_mean"] * 1.10,
        "qk_routing_more_concentrated_withK": final["withK_zipf"]["Bqk_top1_energy_mean"]
        > final["noK_uniform"]["Bqk_top1_energy_mean"] * 1.05,
        "qk_input_aligns_representation_common_direction": final["withK_zipf"]["align_rE_to_Bqk_input_sqcos_mean"] > 0.25,
        "common_direction_enters_at_least_one_attention_parameter": max(
            final["withK_zipf"]["align_rE_to_Wq_input_sqcos_mean"],
            final["withK_zipf"]["align_rE_to_Wk_input_sqcos_mean"],
            final["withK_zipf"]["align_rE_to_Wv_input_sqcos_mean"],
        )
        > 0.5,
        "k_related_pushes_embedding_sigma1_early": early["withK_zipf"]["E_centered_k_related_grad_sigma1_growth_mean"] > 0.0,
        "qk_top_ablation_damages_k_related_more_than_internal": final["withK_zipf"][
            "ablate_Bqk_top1_k_related_loss_delta_mean"
        ]
        > final["withK_zipf"]["ablate_Bqk_top1_internal_loss_delta_mean"],
        "qk_top_ablation_shows_tail_depends_on_shared_routing": final["withK_zipf"][
            "ablate_Bqk_top1_tail_loss_delta_mean"
        ]
        > final["withK_zipf"]["ablate_Bqk_top1_common_loss_delta_mean"],
    }
    status = "pass" if all(checks.values()) else "partial" if any(checks.values()) else "fail"
    key_numbers = {
        "final_E_centered_top1_energy": {c: final[c]["E_centered_top1_energy_mean"] for c in final},
        "final_Bqk_top1_energy": {c: final[c]["Bqk_top1_energy_mean"] for c in final},
        "final_align_rE_to_Bqk_input_sqcos": {c: final[c]["align_rE_to_Bqk_input_sqcos_mean"] for c in final},
        "final_align_rE_to_Wq_Wk_Wv_input_sqcos_withK_zipf": {
            "Wq": final["withK_zipf"]["align_rE_to_Wq_input_sqcos_mean"],
            "Wk": final["withK_zipf"]["align_rE_to_Wk_input_sqcos_mean"],
            "Wv": final["withK_zipf"]["align_rE_to_Wv_input_sqcos_mean"],
        },
        "early_sigma1_growth_contributions_withK_zipf": {
            "E_k_related": early["withK_zipf"]["E_centered_k_related_grad_sigma1_growth_mean"],
            "E_tail": early["withK_zipf"]["E_centered_tail_grad_sigma1_growth_mean"],
            "Wq_k_related": early["withK_zipf"]["Wq_k_related_grad_sigma1_growth_mean"],
            "Wk_k_related": early["withK_zipf"]["Wk_k_related_grad_sigma1_growth_mean"],
            "Wv_k_related": early["withK_zipf"]["Wv_k_related_grad_sigma1_growth_mean"],
        },
        "final_Bqk_top1_ablation_loss_delta_withK_zipf": {
            "common": final["withK_zipf"]["ablate_Bqk_top1_common_loss_delta_mean"],
            "tail": final["withK_zipf"]["ablate_Bqk_top1_tail_loss_delta_mean"],
            "k_related": final["withK_zipf"]["ablate_Bqk_top1_k_related_loss_delta_mean"],
            "internal": final["withK_zipf"]["ablate_Bqk_top1_internal_loss_delta_mean"],
        },
    }
    return {"status": status, "checks": checks, "key_numbers": key_numbers}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, agg: List[Dict[str, object]]) -> None:
    conditions = ["noK_uniform", "withK_uniform", "withK_zipf"]
    panels = [
        ("E_centered_top1_energy_mean", "Embedding centered top-1 energy"),
        ("Bqk_top1_energy_mean", "QK bilinear top-1 energy"),
        ("align_rE_to_Bqk_input_sqcos_mean", "align E common dir to Bqk input dir"),
        ("align_rE_to_Wq_input_sqcos_mean", "align E common dir to Wq input dir"),
        ("align_rE_to_Wk_input_sqcos_mean", "align E common dir to Wk input dir"),
        ("align_rE_to_Wv_input_sqcos_mean", "align E common dir to Wv input dir"),
        ("Bqk_tail_query_input_top1_mass_mean", "tail query mass on Bqk top dir"),
        ("ablate_Bqk_top1_common_loss_delta_mean", "Bqk top1 ablation common loss delta"),
        ("ablate_Bqk_top1_tail_loss_delta_mean", "Bqk top1 ablation tail loss delta"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, (metric, title) in zip(axes.flat, panels):
        for condition in conditions:
            items = trajectory(agg, condition)
            xs = np.array([int(x["step"]) for x in items])
            ys = np.array([float(x[metric]) for x in items])
            sd = np.array([float(x.get(metric.replace("_mean", "_std"), 0.0)) for x in items])
            ax.plot(xs, ys, label=condition, linewidth=1.8)
            ax.fill_between(xs, ys - sd, ys + sd, alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Stage 6: parameter-representation singular direction alignment", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--record_every", type=int, default=20)
    parser.add_argument("--theta_deg", type=float, default=12.0)
    parser.add_argument("--init_noise", type=float, default=0.005)
    parser.add_argument("--residual_alpha", type=float, default=0.0)
    parser.add_argument("--use_o_proj", action="store_true")
    cfg = Config(**vars(parser.parse_args()))
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
    plot(outdir / "metrics.png", agg)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
