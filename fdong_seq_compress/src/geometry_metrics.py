from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch


EPS = 1e-12


def _to_cpu_float64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu").to(dtype=torch.float64).contiguous()


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.mean().item())


def _safe_std(x: torch.Tensor) -> float:
    if x.numel() <= 1:
        return 0.0 if x.numel() == 1 else float("nan")
    return float(x.std(unbiased=False).item())


def parse_int_list(value: str, default: Optional[List[int]] = None) -> List[int]:
    if value is None or value == "":
        if default is None:
            raise ValueError("Empty integer list.")
        return default
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def select_indices(total: int, selector: str) -> List[int]:
    if selector == "all":
        return list(range(total))
    result = []
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(part))
    return [idx for idx in result if 0 <= idx < total]


def extract_cache_tensor(past_key_values, layer_idx: int, kind: str) -> torch.Tensor:
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        cache = past_key_values.key_cache if kind == "K" else past_key_values.value_cache
        return cache[layer_idx][0]

    layer_cache = past_key_values[layer_idx]
    if isinstance(layer_cache, (list, tuple)):
        return layer_cache[0 if kind == "K" else 1][0]

    key = layer_cache.keys if hasattr(layer_cache, "keys") else layer_cache.key
    value = layer_cache.values if hasattr(layer_cache, "values") else layer_cache.value
    return (key if kind == "K" else value)[0]


def svd_stats(x: torch.Tensor, energy_thresholds: Iterable[float]) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    x = _to_cpu_float64(x)
    n, dim = x.shape
    if n == 0:
        raise ValueError("Cannot analyze empty matrix.")

    centered = x - x.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total_energy = energy.sum().clamp_min(EPS)
    probs = energy / total_energy
    cumulative = torch.cumsum(probs, dim=0)

    stats: Dict[str, float] = {
        "num_tokens": n,
        "head_dim": dim,
        "fro_norm_centered": float(torch.linalg.vector_norm(centered).item()),
        "top_singular_value": float(singular_values[0].item()) if singular_values.numel() else float("nan"),
        "top_energy_ratio": float(probs[0].item()) if probs.numel() else float("nan"),
        "stable_rank": float((energy.sum() / energy.max().clamp_min(EPS)).item()) if energy.numel() else float("nan"),
        "effective_rank": float(torch.exp(-(probs * torch.log(probs.clamp_min(EPS))).sum()).item()) if probs.numel() else float("nan"),
    }

    for threshold in energy_thresholds:
        rank = int(torch.searchsorted(cumulative, torch.tensor(float(threshold), dtype=cumulative.dtype)).item() + 1)
        stats[f"rank_energy_{int(threshold * 100)}"] = min(rank, singular_values.numel())

    return stats, singular_values, centered


def cosine_stats(x: torch.Tensor, max_pairs: int = 200_000) -> Dict[str, float]:
    x = _to_cpu_float64(x)
    centered = x - x.mean(dim=0, keepdim=True)
    return {
        **_cosine_stats_one(x, "raw", max_pairs),
        **_cosine_stats_one(centered, "centered", max_pairs),
        "mean_vector_norm": float(torch.linalg.vector_norm(x.mean(dim=0)).item()),
        "mean_row_norm": float(torch.linalg.vector_norm(x, dim=1).mean().item()),
        "centered_mean_row_norm": float(torch.linalg.vector_norm(centered, dim=1).mean().item()),
    }


def _cosine_stats_one(x: torch.Tensor, prefix: str, max_pairs: int) -> Dict[str, float]:
    n = x.shape[0]
    normalized = torch.nn.functional.normalize(x, p=2, dim=1, eps=EPS)
    if n <= 1:
        return {
            f"{prefix}_offdiag_cos_mean": float("nan"),
            f"{prefix}_offdiag_cos_std": float("nan"),
            f"{prefix}_offdiag_cos_p05": float("nan"),
            f"{prefix}_offdiag_cos_p50": float("nan"),
            f"{prefix}_offdiag_cos_p95": float("nan"),
        }

    num_pairs = n * (n - 1) // 2
    if num_pairs <= max_pairs:
        sim = normalized @ normalized.T
        mask = ~torch.eye(n, dtype=torch.bool)
        vals = sim[mask]
    else:
        gen = torch.Generator(device="cpu").manual_seed(20260528 + n + x.shape[1])
        i = torch.randint(0, n, (max_pairs,), generator=gen)
        j = torch.randint(0, n - 1, (max_pairs,), generator=gen)
        j = j + (j >= i).long()
        vals = (normalized[i] * normalized[j]).sum(dim=1)

    q = torch.quantile(vals, torch.tensor([0.05, 0.50, 0.95], dtype=vals.dtype))
    return {
        f"{prefix}_offdiag_cos_mean": _safe_mean(vals),
        f"{prefix}_offdiag_cos_std": _safe_std(vals),
        f"{prefix}_offdiag_cos_p05": float(q[0].item()),
        f"{prefix}_offdiag_cos_p50": float(q[1].item()),
        f"{prefix}_offdiag_cos_p95": float(q[2].item()),
    }


def temporal_stats(x: torch.Tensor) -> Dict[str, float]:
    x = _to_cpu_float64(x)
    if x.shape[0] < 2:
        return {
            "adjacent_delta_l2_mean": float("nan"),
            "adjacent_delta_l2_std": float("nan"),
            "adjacent_cos_mean": float("nan"),
            "second_diff_l2_mean": float("nan"),
        }
    delta = x[1:] - x[:-1]
    delta_norm = torch.linalg.vector_norm(delta, dim=1)
    cos = (
        torch.nn.functional.normalize(x[1:], p=2, dim=1, eps=EPS)
        * torch.nn.functional.normalize(x[:-1], p=2, dim=1, eps=EPS)
    ).sum(dim=1)
    if x.shape[0] >= 3:
        second = x[2:] - 2 * x[1:-1] + x[:-2]
        second_norm_mean = float(torch.linalg.vector_norm(second, dim=1).mean().item())
    else:
        second_norm_mean = float("nan")
    return {
        "adjacent_delta_l2_mean": _safe_mean(delta_norm),
        "adjacent_delta_l2_std": _safe_std(delta_norm),
        "adjacent_cos_mean": _safe_mean(cos),
        "second_diff_l2_mean": second_norm_mean,
    }


def subspace_basis(x: torch.Tensor, rank: int) -> torch.Tensor:
    x = _to_cpu_float64(x)
    centered = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = max(1, min(rank, vh.shape[0]))
    return vh[:rank].T.contiguous()


def subspace_overlap(prev_basis: Optional[torch.Tensor], curr_basis: torch.Tensor) -> Dict[str, float]:
    if prev_basis is None:
        return {
            "subspace_overlap_mean_cos": float("nan"),
            "subspace_overlap_min_cos": float("nan"),
            "subspace_max_angle_deg": float("nan"),
        }
    rank = min(prev_basis.shape[1], curr_basis.shape[1])
    cross = prev_basis[:, :rank].T @ curr_basis[:, :rank]
    singular = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
    min_cos = float(singular.min().item())
    return {
        "subspace_overlap_mean_cos": float(singular.mean().item()),
        "subspace_overlap_min_cos": min_cos,
        "subspace_max_angle_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, min_cos))))),
    }


def novelty_against_previous(
    curr_x: torch.Tensor,
    prev_prefix_len: Optional[int],
    prev_basis: Optional[torch.Tensor],
) -> Dict[str, float]:
    if prev_prefix_len is None or prev_basis is None or curr_x.shape[0] <= prev_prefix_len:
        return {
            "novelty_residual_ratio_mean": float("nan"),
            "novelty_residual_ratio_p95": float("nan"),
        }
    x = _to_cpu_float64(curr_x[prev_prefix_len:])
    centered = x - x.mean(dim=0, keepdim=True)
    projection = centered @ prev_basis @ prev_basis.T
    residual = centered - projection
    residual_ratio = torch.linalg.vector_norm(residual, dim=1) / torch.linalg.vector_norm(centered, dim=1).clamp_min(EPS)
    return {
        "novelty_residual_ratio_mean": _safe_mean(residual_ratio),
        "novelty_residual_ratio_p95": float(torch.quantile(residual_ratio, 0.95).item()),
    }


def block_geometry_metrics(x: torch.Tensor, block_sizes: Iterable[int]) -> List[Dict[str, float]]:
    x = _to_cpu_float64(x)
    rows: List[Dict[str, float]] = []
    global_mean = x.mean(dim=0, keepdim=True)
    for block_size in block_sizes:
        num_blocks = x.shape[0] // block_size
        if num_blocks < 2:
            continue
        trimmed = x[: num_blocks * block_size]
        blocks = trimmed.reshape(num_blocks, block_size, x.shape[1])
        centroids = blocks.mean(dim=1)
        within = ((blocks - centroids[:, None, :]).square().sum(dim=2)).mean()
        between = ((centroids - global_mean).square().sum(dim=1)).mean()
        centroid_delta = torch.linalg.vector_norm(centroids[1:] - centroids[:-1], dim=1)
        rows.append(
            {
                "block_size": block_size,
                "num_blocks": num_blocks,
                "within_var": float(within.item()),
                "between_centroid_var": float(between.item()),
                "within_between_ratio": float((within / between.clamp_min(EPS)).item()),
                "centroid_delta_l2_mean": _safe_mean(centroid_delta),
                "centroid_delta_l2_std": _safe_std(centroid_delta),
            }
        )
    return rows
