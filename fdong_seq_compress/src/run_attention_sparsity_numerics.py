from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geometry_metrics import select_indices  # noqa: E402
from model_loader import load_model_and_tokenizer  # noqa: E402
from text_loader import load_tokenized_text  # noqa: E402


EPS = 1e-12


def parse_float_list(value: str) -> List[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item <= 0.0 or item >= 1.0 for item in result):
        raise ValueError("Ratios must be comma-separated values strictly between 0 and 1.")
    return result


def parse_int_list(value: str) -> List[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item <= 0 for item in result):
        raise ValueError("Ranks must be positive comma-separated integers.")
    return result


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float(torch.dot(a, b).div(denom).item())


def finite_mean(values: Iterable[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(valid) / len(valid)) if valid else float("nan")


def quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x, torch.tensor(q, dtype=x.dtype)).item())


def compute_layer_qkv(
    model,
    hidden_states: torch.Tensor,
    layer_idx: int,
    position_embeddings,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    with torch.no_grad():
        x = layer.input_layernorm(hidden_states)
        shape = (*x.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(x).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(shape)).transpose(1, 2)
        v = attn.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
    return q[0], k[0], v[0]


def sampled_query_indices(seq_len: int, query_window: int, query_stride: int) -> List[int]:
    start = max(1, seq_len - max(1, query_window))
    result = list(range(start, seq_len, max(1, query_stride)))
    if result[-1] != seq_len - 1:
        result.append(seq_len - 1)
    return result


def subset_indices(scores: torch.Tensor, ratio: float, largest: bool) -> torch.Tensor:
    count = max(1, int(math.ceil(scores.numel() * ratio)))
    return torch.topk(scores, k=count, largest=largest, sorted=False).indices


def score_distribution_metrics(scores: torch.Tensor) -> Dict[str, float]:
    mean = scores.mean()
    std = scores.std(unbiased=False).clamp_min(EPS)
    centered = (scores - mean) / std
    sorted_scores = torch.sort(scores, descending=True).values
    top_gap = sorted_scores[0] - sorted_scores[1] if sorted_scores.numel() > 1 else torch.tensor(0.0)
    p99 = quantile(scores, 0.99)
    p90 = quantile(scores, 0.90)
    p50 = quantile(scores, 0.50)
    p10 = quantile(scores, 0.10)
    p01 = quantile(scores, 0.01)
    return {
        "score_mean": float(mean.item()),
        "score_std": float(std.item()),
        "score_skewness": float(centered.pow(3).mean().item()),
        "score_excess_kurtosis": float(centered.pow(4).mean().item() - 3.0),
        "score_min": float(sorted_scores[-1].item()),
        "score_p01": p01,
        "score_p10": p10,
        "score_p50": p50,
        "score_p90": p90,
        "score_p99": p99,
        "score_max": float(sorted_scores[0].item()),
        "score_top1_z": float(centered.max().item()),
        "score_top_gap": float(top_gap.item()),
        "score_top_gap_over_std": float((top_gap / std).item()),
        "score_upper_tail_span_over_std": (p99 - p50) / float(std.item()),
        "score_lower_tail_span_over_std": (p50 - p01) / float(std.item()),
    }


def build_svd_bases(x: torch.Tensor, max_rank: int) -> Dict[str, torch.Tensor]:
    x = x.detach().cpu().to(dtype=torch.float32)
    q = min(max_rank, x.shape[0], x.shape[1])
    bases: Dict[str, torch.Tensor] = {}
    for mode, matrix in (("raw", x), ("centered", x - x.mean(dim=0, keepdim=True))):
        _, _, basis = torch.pca_lowrank(matrix, q=q, center=False, niter=4)
        bases[mode] = basis
    return bases


def per_token_projection_energy(x: torch.Tensor, basis: torch.Tensor, rank: int, center: torch.Tensor | None) -> torch.Tensor:
    x = x.detach().cpu().to(dtype=torch.float32)
    if center is not None:
        x = x - center
    selected_basis = basis[:, : min(rank, basis.shape[1])]
    projected_energy = (x @ selected_basis).pow(2).sum(dim=1)
    total_energy = x.pow(2).sum(dim=1).clamp_min(EPS)
    return projected_energy / total_energy


def build_projection_profiles(
    x: torch.Tensor,
    bases: Dict[str, torch.Tensor],
    ranks: Sequence[int],
) -> Dict[Tuple[str, int], torch.Tensor]:
    x_cpu = x.detach().cpu().to(dtype=torch.float32)
    centers = {"raw": None, "centered": x_cpu.mean(dim=0, keepdim=True)}
    return {
        (mode, rank): per_token_projection_energy(x_cpu, bases[mode], rank, centers[mode])
        for mode in ("raw", "centered")
        for rank in ranks
    }


def contribution_metrics(
    values: torch.Tensor,
    probs: torch.Tensor,
    indices: torch.Tensor,
    full_output: torch.Tensor,
) -> Dict[str, float]:
    selected_probs = probs[indices]
    selected_values = values[indices]
    contribution = selected_probs @ selected_values
    mass = selected_probs.sum().clamp_min(EPS)
    renormalized = contribution / mass
    component_norm_sum = (selected_probs * torch.linalg.vector_norm(selected_values, dim=1)).sum().clamp_min(EPS)
    return {
        "mass": float(mass.item()),
        "contribution_cos_full": cosine(contribution, full_output),
        "contribution_norm_ratio": float(
            (torch.linalg.vector_norm(contribution) / torch.linalg.vector_norm(full_output).clamp_min(EPS)).item()
        ),
        "renormalized_cos_full": cosine(renormalized, full_output),
        "renormalized_norm_ratio": float(
            (torch.linalg.vector_norm(renormalized) / torch.linalg.vector_norm(full_output).clamp_min(EPS)).item()
        ),
        "cancellation_ratio": float((torch.linalg.vector_norm(contribution) / component_norm_sum).item()),
    }


def analyze_query_head(
    q_t: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scaling: float,
    ratios: Sequence[float],
    svd_ranks: Sequence[int],
    k_projection_profiles: Dict[Tuple[str, int], torch.Tensor],
    v_projection_profiles: Dict[Tuple[str, int], torch.Tensor],
    identity: Dict,
) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    scores = (q_t @ keys.T) * scaling
    probs = torch.softmax(scores, dim=-1)
    full_output = probs @ values

    distribution_row = dict(identity)
    distribution_row.update(score_distribution_metrics(scores))
    distribution_row["attention_entropy"] = float((-(probs.clamp_min(EPS) * probs.clamp_min(EPS).log()).sum()).item())
    distribution_row["attention_effective_tokens"] = float(torch.exp(torch.tensor(distribution_row["attention_entropy"])).item())

    mass_rows: List[Dict] = []
    svd_rows: List[Dict] = []
    value_rows: List[Dict] = []

    bucket_contributions: Dict[Tuple[str, float], torch.Tensor] = {}
    for ratio in ratios:
        for bucket, largest in (("top", True), ("tail", False)):
            indices = subset_indices(scores, ratio, largest=largest)
            selected_probs = probs[indices]
            mass_rows.append(
                {
                    **identity,
                    "bucket": bucket,
                    "ratio": ratio,
                    "token_count": int(indices.numel()),
                    "softmax_mass": float(selected_probs.sum().item()),
                    "mean_score": float(scores[indices].mean().item()),
                    "min_score": float(scores[indices].min().item()),
                    "max_score": float(scores[indices].max().item()),
                }
            )

            for representation, profiles in (
                ("K", k_projection_profiles),
                ("V", v_projection_profiles),
            ):
                for mode in ("raw", "centered"):
                    for rank in svd_ranks:
                        svd_rows.append(
                            {
                                **identity,
                                "bucket": bucket,
                                "ratio": ratio,
                                "representation": representation,
                                "basis_mode": mode,
                                "rank": rank,
                                "projection_energy_ratio": float(profiles[(mode, rank)][indices].mean().item()),
                            }
                        )

            metrics = contribution_metrics(values, probs, indices, full_output)
            contribution = probs[indices] @ values[indices]
            bucket_contributions[(bucket, ratio)] = contribution
            value_rows.append({**identity, "bucket": bucket, "ratio": ratio, **metrics})

    for ratio in ratios:
        top_contribution = bucket_contributions[("top", ratio)]
        tail_contribution = bucket_contributions[("tail", ratio)]
        value_rows.append(
            {
                **identity,
                "bucket": "top_vs_tail",
                "ratio": ratio,
                "contribution_cos_full": float("nan"),
                "contribution_norm_ratio": float("nan"),
                "renormalized_cos_full": cosine(top_contribution, tail_contribution),
                "renormalized_norm_ratio": float("nan"),
                "cancellation_ratio": float("nan"),
                "mass": float("nan"),
            }
        )
    return distribution_row, mass_rows, svd_rows, value_rows


def group_mean(rows: Sequence[Dict], group_keys: Sequence[str], value_keys: Sequence[str]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in groups.items():
        result = {name: value for name, value in zip(group_keys, key)}
        result["sample_count"] = len(members)
        for value_key in value_keys:
            result[value_key] = finite_mean(member.get(value_key, float("nan")) for member in members)
        output.append(result)
    return output


def plot_mass_curve(rows: Sequence[Dict], output_path: Path) -> None:
    summary = group_mean(rows, ["bucket", "ratio"], ["softmax_mass"])
    plt.figure(figsize=(7.2, 4.8))
    for bucket, color in (("top", "#b42318"), ("tail", "#35618d")):
        subset = sorted((row for row in summary if row["bucket"] == bucket), key=lambda row: row["ratio"])
        plt.plot([100 * row["ratio"] for row in subset], [row["softmax_mass"] for row in subset], marker="o", label=bucket, color=color)
    plt.xscale("log")
    plt.xlabel("Selected score-ranked tokens (%)")
    plt.ylabel("Mean full-softmax mass")
    plt.title("Score top/tail mass curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_v_direction(rows: Sequence[Dict], output_path: Path) -> None:
    usable = [row for row in rows if row["bucket"] in {"top", "tail"}]
    summary = group_mean(usable, ["bucket", "ratio"], ["renormalized_cos_full", "cancellation_ratio"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for bucket, color in (("top", "#b42318"), ("tail", "#35618d")):
        subset = sorted((row for row in summary if row["bucket"] == bucket), key=lambda row: row["ratio"])
        x = [100 * row["ratio"] for row in subset]
        axes[0].plot(x, [row["renormalized_cos_full"] for row in subset], marker="o", label=bucket, color=color)
        axes[1].plot(x, [row["cancellation_ratio"] for row in subset], marker="o", label=bucket, color=color)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Selected score-ranked tokens (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Cosine with full attention output")
    axes[0].set_title("Renormalized subset V output")
    axes[1].set_ylabel("||sum weighted V|| / sum ||weighted V||")
    axes[1].set_title("Within-subset directional cancellation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_svd_projection(rows: Sequence[Dict], output_path: Path, target_ratio: float) -> None:
    selected = [row for row in rows if abs(float(row["ratio"]) - target_ratio) < 1e-9]
    summary = group_mean(
        selected,
        ["representation", "basis_mode", "bucket", "rank"],
        ["projection_energy_ratio"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, representation in zip(axes, ("K", "V")):
        for mode, bucket, style, color in (
            ("raw", "top", "-", "#b42318"),
            ("raw", "tail", "-", "#35618d"),
            ("centered", "top", "--", "#b42318"),
            ("centered", "tail", "--", "#35618d"),
        ):
            subset = sorted(
                (
                    row
                    for row in summary
                    if row["representation"] == representation
                    and row["basis_mode"] == mode
                    and row["bucket"] == bucket
                ),
                key=lambda row: row["rank"],
            )
            axis.plot(
                [row["rank"] for row in subset],
                [row["projection_energy_ratio"] for row in subset],
                linestyle=style,
                marker="o",
                color=color,
                label=f"{bucket}-{mode}",
            )
        axis.set_title(f"{representation} projection, score ratio={100 * target_ratio:g}%")
        axis.set_xlabel("SVD subspace rank")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Mean projection energy ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the numerical origin of oracle attention sparsity.")
    parser.add_argument("--model-path", default="fdong/Qwen3-0.6B")
    parser.add_argument("--text-path", default="fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--q-heads", default="all")
    parser.add_argument("--query-window", type=int, default=512)
    parser.add_argument("--query-stride", type=int, default=32)
    parser.add_argument("--ratios", default="0.001,0.005,0.01,0.02,0.04,0.06,0.1,0.2,0.5")
    parser.add_argument("--svd-ranks", default="1,4,8,16")
    parser.add_argument("--svd-plot-ratio", type=float, default=0.01)
    args = parser.parse_args()

    ratios = parse_float_list(args.ratios)
    svd_ranks = parse_int_list(args.svd_ranks)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"fdong_seq_compress/outputs/attention_sparsity_numerics_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    _, input_ids = load_tokenized_text(tokenizer, args.text_path, args.max_tokens)
    seq_len = int(input_ids.numel())
    model_max = getattr(model.config, "max_position_embeddings", None)
    if model_max is not None and seq_len > int(model_max):
        raise ValueError(f"seq_len={seq_len} exceeds model max_position_embeddings={model_max}")

    print(f"device={device} seq_len={seq_len} text={args.text_path}", flush=True)
    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids[None, :].to(device),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)

    num_layers = int(model.config.num_hidden_layers)
    num_q_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    num_groups = num_q_heads // num_kv_heads
    layers = select_indices(num_layers, args.layers)
    q_heads = select_indices(num_q_heads, args.q_heads)
    query_ids = sampled_query_indices(seq_len, args.query_window, args.query_stride)

    distribution_rows: List[Dict] = []
    mass_rows: List[Dict] = []
    svd_rows: List[Dict] = []
    value_rows: List[Dict] = []

    for layer_idx in layers:
        q, k, v = compute_layer_qkv(model, hidden_states[layer_idx], layer_idx, position_embeddings)
        q = q.detach().cpu().to(dtype=torch.float32)
        k = k.detach().cpu().to(dtype=torch.float32)
        v = v.detach().cpu().to(dtype=torch.float32)
        scaling = float(model.model.layers[layer_idx].self_attn.scaling)

        bases_by_kv_head = {}
        max_rank = max(svd_ranks)
        for kv_head_idx in sorted({head // num_groups for head in q_heads}):
            k_bank = k[kv_head_idx]
            v_bank = v[kv_head_idx]
            bases_by_kv_head[kv_head_idx] = {
                "K": build_svd_bases(k_bank, max_rank),
                "V": build_svd_bases(v_bank, max_rank),
            }
            bases_by_kv_head[kv_head_idx]["K_profiles"] = build_projection_profiles(
                k_bank, bases_by_kv_head[kv_head_idx]["K"], svd_ranks
            )
            bases_by_kv_head[kv_head_idx]["V_profiles"] = build_projection_profiles(
                v_bank, bases_by_kv_head[kv_head_idx]["V"], svd_ranks
            )

        for q_head_idx in q_heads:
            kv_head_idx = q_head_idx // num_groups
            bases = bases_by_kv_head[kv_head_idx]
            for query_idx in query_ids:
                identity = {
                    "layer": layer_idx,
                    "q_head": q_head_idx,
                    "kv_head": kv_head_idx,
                    "query_index": query_idx,
                    "key_count": query_idx + 1,
                }
                rows = analyze_query_head(
                    q[q_head_idx, query_idx],
                    k[kv_head_idx, : query_idx + 1],
                    v[kv_head_idx, : query_idx + 1],
                    scaling,
                    ratios,
                    svd_ranks,
                    bases["K_profiles"],
                    bases["V_profiles"],
                    identity,
                )
                distribution_rows.append(rows[0])
                mass_rows.extend(rows[1])
                svd_rows.extend(rows[2])
                value_rows.extend(rows[3])
        print(f"finished layer={layer_idx}", flush=True)

    distribution_summary = group_mean(
        distribution_rows,
        ["layer", "q_head", "kv_head"],
        [
            "score_std",
            "score_skewness",
            "score_excess_kurtosis",
            "score_top1_z",
            "score_top_gap_over_std",
            "score_upper_tail_span_over_std",
            "score_lower_tail_span_over_std",
            "attention_entropy",
            "attention_effective_tokens",
        ],
    )
    mass_summary = group_mean(mass_rows, ["layer", "q_head", "kv_head", "bucket", "ratio"], ["softmax_mass", "mean_score"])
    svd_summary = group_mean(
        svd_rows,
        ["layer", "q_head", "kv_head", "bucket", "ratio", "representation", "basis_mode", "rank"],
        ["projection_energy_ratio"],
    )
    value_summary = group_mean(
        value_rows,
        ["layer", "q_head", "kv_head", "bucket", "ratio"],
        [
            "mass",
            "contribution_cos_full",
            "contribution_norm_ratio",
            "renormalized_cos_full",
            "renormalized_norm_ratio",
            "cancellation_ratio",
        ],
    )

    write_csv(output_dir / "score_distribution_by_query.csv", distribution_rows)
    write_csv(output_dir / "score_distribution_by_layer_head.csv", distribution_summary)
    write_csv(output_dir / "softmax_mass_by_query.csv", mass_rows)
    write_csv(output_dir / "softmax_mass_by_layer_head.csv", mass_summary)
    write_csv(output_dir / "svd_projection_by_query.csv", svd_rows)
    write_csv(output_dir / "svd_projection_by_layer_head.csv", svd_summary)
    write_csv(output_dir / "value_direction_by_query.csv", value_rows)
    write_csv(output_dir / "value_direction_by_layer_head.csv", value_summary)

    plot_mass_curve(mass_rows, output_dir / "softmax_mass_curve.png")
    plot_v_direction(value_rows, output_dir / "value_direction_curve.png")
    plot_svd_projection(svd_rows, output_dir / "svd_projection_top_tail.png", args.svd_plot_ratio)

    summary = {
        "model_path": args.model_path,
        "text_path": args.text_path,
        "output_dir": str(output_dir),
        "device": str(device),
        "seq_len": seq_len,
        "model_max_position_embeddings": model_max,
        "layers": layers,
        "q_heads": q_heads,
        "query_indices": query_ids,
        "ratios": ratios,
        "svd_ranks": svd_ranks,
        "questions": {
            "A1": "Is the QK score distribution smooth or dominated by tails, gaps, and extremes?",
            "A2": "How much full-softmax mass is covered by score top/tail ratios?",
            "A3": "Do score top/tail K and V tokens occupy common principal or residual subspaces?",
            "A4": "Do score-ranked V contributions align, cancel, repeat, or perturb the full output direction?",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
