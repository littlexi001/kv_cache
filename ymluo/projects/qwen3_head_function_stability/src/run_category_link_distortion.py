from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_head_function_stability import (
    CATEGORIES,
    build_controlled_samples,
    build_token_probes,
    parse_index_spec,
    robust_center_scale,
    str2bool,
)


METRICS = (
    "removed_mass",
    "target_contribution_relative_l2",
    "relative_output_l2",
    "output_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact per-head local attention-output distortion after removing "
            "category-specific attention links."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--min_history", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recent_window", type=int, default=16)
    parser.add_argument("--manual_query_tail", type=int, default=1)
    parser.add_argument("--exclude_local_from_typed", type=str2bool, default=True)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def category_link_metrics(
    attention: torch.Tensor,
    values: torch.Tensor,
    probes: Sequence[tuple[int, tuple[int, ...]]],
) -> dict[str, torch.Tensor | int]:
    """Remove target links, renormalize the row, and compare exact head outputs.

    ``attention`` has shape ``[heads, queries, keys]`` and is already causal.
    ``values`` has shape ``[heads, keys, head_dim]`` after GQA repetition.
    Metrics are averaged over the supplied query/key probes.
    """

    if attention.ndim != 3 or values.ndim != 3:
        raise ValueError("attention and values must both be three-dimensional")
    if attention.shape[0] != values.shape[0] or attention.shape[-1] != values.shape[1]:
        raise ValueError("attention/value head and key dimensions must agree")
    head_count = attention.shape[0]
    sums = {
        metric: torch.zeros(head_count, dtype=torch.float64, device=attention.device)
        for metric in METRICS
    }
    valid_rows = torch.zeros(head_count, dtype=torch.float64, device=attention.device)
    for query, keys_tuple in probes:
        if query < 0 or query >= attention.shape[1]:
            continue
        keys = torch.tensor(
            sorted({key for key in keys_tuple if 0 <= key <= query}),
            dtype=torch.long,
            device=attention.device,
        )
        if keys.numel() == 0:
            continue
        probabilities = attention[:, query, : query + 1].float()
        row_values = values[:, : query + 1].float()
        full_output = torch.einsum("hs,hsd->hd", probabilities, row_values)
        removed_mass = probabilities.index_select(1, keys).sum(dim=-1)
        target_contribution = torch.einsum(
            "hs,hsd->hd",
            probabilities.index_select(1, keys),
            row_values.index_select(1, keys),
        )
        kept = probabilities.clone()
        kept[:, keys] = 0.0
        denominator = kept.sum(dim=-1, keepdim=True)
        renormalized = kept / denominator.clamp_min(1.0e-12)
        ablated_output = torch.einsum("hs,hsd->hd", renormalized, row_values)
        full_norm = torch.linalg.vector_norm(full_output, dim=-1).clamp_min(1.0e-8)
        difference = full_output - ablated_output
        relative_output_l2 = torch.linalg.vector_norm(difference, dim=-1) / full_norm
        contribution_relative_l2 = (
            torch.linalg.vector_norm(target_contribution, dim=-1) / full_norm
        )
        cosine = F.cosine_similarity(full_output, ablated_output, dim=-1)
        valid = torch.isfinite(relative_output_l2) & torch.isfinite(cosine)
        sums["removed_mass"] += torch.where(valid, removed_mass.double(), 0.0)
        sums["target_contribution_relative_l2"] += torch.where(
            valid, contribution_relative_l2.double(), 0.0
        )
        sums["relative_output_l2"] += torch.where(valid, relative_output_l2.double(), 0.0)
        sums["output_cosine"] += torch.where(valid, cosine.double(), 0.0)
        valid_rows += valid.double()
    denominator = valid_rows.clamp_min(1.0)
    return {
        **{metric: sums[metric] / denominator for metric in METRICS},
        "query_count_by_head": valid_rows,
        "probe_count": len(probes),
    }


def aggregate_rows(
    sample_rows: Sequence[dict[str, Any]],
    selected_layers: Sequence[int],
    selected_heads: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[(int(row["layer"]), int(row["head"]), str(row["category"]))].append(row)

    category_rows: list[dict[str, Any]] = []
    for layer in selected_layers:
        for head in selected_heads:
            for category in CATEGORIES:
                rows = grouped.get((layer, head, category), [])
                total_queries = sum(int(row["query_count"]) for row in rows)
                if not rows or total_queries <= 0:
                    continue
                output: dict[str, Any] = {
                    "layer": layer,
                    "head": head,
                    "category": category,
                    "sample_count": len(rows),
                    "query_count": total_queries,
                }
                for metric in METRICS:
                    output[f"mean_{metric}"] = sum(
                        float(row[metric]) * int(row["query_count"]) for row in rows
                    ) / total_queries
                category_rows.append(output)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in category_rows:
        by_category[str(row["category"])].append(row)
    ranking_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        rows = by_category.get(category, [])
        values = [float(row["mean_relative_output_l2"]) for row in rows]
        center, scale = robust_center_scale(values)
        for row in rows:
            row["distortion_robust_z"] = (
                (float(row["mean_relative_output_l2"]) - center) / scale
                if math.isfinite(scale) and scale > 0
                else 0.0
            )
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["mean_relative_output_l2"]),
                int(row["layer"]),
                int(row["head"]),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            ranking_rows.append(
                {
                    "category": category,
                    "distortion_rank": rank,
                    **row,
                }
            )

    lookup = {
        (int(row["layer"]), int(row["head"]), str(row["category"])): row
        for row in category_rows
    }
    profile_rows: list[dict[str, Any]] = []
    for layer in selected_layers:
        for head in selected_heads:
            available = [
                lookup[(layer, head, category)]
                for category in CATEGORIES
                if (layer, head, category) in lookup
            ]
            dominant = max(
                available,
                key=lambda row: float(row.get("distortion_robust_z", -math.inf)),
            )
            profile: dict[str, Any] = {
                "layer": layer,
                "head": head,
                "distortion_dominant_category": dominant["category"],
                "dominant_distortion_robust_z": dominant["distortion_robust_z"],
                "dominant_mean_relative_output_l2": dominant["mean_relative_output_l2"],
            }
            for category in CATEGORIES:
                row = lookup.get((layer, head, category))
                profile[f"distortion_{category}"] = (
                    row["mean_relative_output_l2"] if row is not None else ""
                )
                profile[f"distortion_z_{category}"] = (
                    row["distortion_robust_z"] if row is not None else ""
                )
            profile_rows.append(profile)
    return category_rows, ranking_rows, profile_rows


def make_plots(output_dir: Path, profile_rows: Sequence[dict[str, Any]], dpi: int = 180) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
        from matplotlib.patches import Patch
    except ImportError:
        return []
    layers = sorted({int(row["layer"]) for row in profile_rows})
    heads = sorted({int(row["head"]) for row in profile_rows})
    lookup = {(int(row["layer"]), int(row["head"])): row for row in profile_rows}
    category_index = {category: index for index, category in enumerate(CATEGORIES)}
    grid = np.asarray(
        [
            [category_index[str(lookup[(layer, head)]["distortion_dominant_category"])] for head in heads]
            for layer in layers
        ],
        dtype=np.int64,
    )
    colors = [
        "#4C78A8",
        "#72B7B2",
        "#54A24B",
        "#9D755D",
        "#E45756",
        "#F58518",
        "#B279A2",
        "#FF9DA6",
        "#59A14F",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(range(len(CATEGORIES) + 1), cmap.N)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(heads)), heads)
    ax.set_yticks(range(0, len(layers), 2), layers[::2])
    ax.set_xlabel("Query head")
    ax.set_ylabel("Layer")
    ax.set_title("Dominant category by local link-ablation output distortion")
    ax.legend(
        handles=[Patch(facecolor=colors[index], label=category) for category, index in category_index.items()],
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    dominant_path = plot_dir / "dominant_link_distortion_map.png"
    fig.savefig(dominant_path, dpi=dpi)
    plt.close(fig)

    z_grid = np.asarray(
        [
            [float(lookup[(layer, head)]["dominant_distortion_robust_z"]) for head in heads]
            for layer in layers
        ]
    )
    bound = max(1.0, float(np.nanmax(np.abs(z_grid))))
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    image = ax.imshow(
        z_grid,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
    )
    ax.set_xticks(range(len(heads)), heads)
    ax.set_yticks(range(0, len(layers), 2), layers[::2])
    ax.set_xlabel("Query head")
    ax.set_ylabel("Layer")
    ax.set_title("Strength of dominant category-link distortion")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Robust z-score")
    fig.tight_layout()
    strength_path = plot_dir / "dominant_link_distortion_strength.png"
    fig.savefig(strength_path, dpi=dpi)
    plt.close(fig)
    return [str(dominant_path), str(strength_path)]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = build_controlled_samples()
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    selected_layers = parse_index_spec(args.layers, num_layers, "layer")
    selected_heads = parse_index_spec(args.heads, num_heads, "head")
    captured_hidden: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    for layer in selected_layers:
        def capture(
            module: torch.nn.Module,
            hook_args: tuple[Any, ...],
            hook_kwargs: dict[str, Any],
            *,
            layer_index: int = layer,
        ) -> None:
            hidden = hook_kwargs.get("hidden_states")
            if hidden is None:
                hidden = hook_args[0]
            captured_hidden[layer_index] = hidden.detach()

        handles.append(
            model.model.layers[layer].self_attn.register_forward_pre_hook(
                capture, with_kwargs=True
            )
        )

    sample_metric_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for sample_index, sample in enumerate(samples, start=1):
            encoded = tokenizer(
                sample.text,
                add_special_tokens=False,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_seq_length,
            )
            offsets_tensor = encoded.pop("offset_mapping")[0]
            offsets = [(int(item[0]), int(item[1])) for item in offsets_tensor.tolist()]
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            probes = build_token_probes(
                sample,
                offsets,
                min_history=args.min_history,
                sink_tokens=args.sink_tokens,
                recent_window=args.recent_window,
                manual_query_tail=args.manual_query_tail,
                exclude_local_from_typed=args.exclude_local_from_typed,
            )
            captured_hidden.clear()
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
            if outputs.attentions is None or any(item is None for item in outputs.attentions):
                raise RuntimeError("Qwen3 did not return attention weights; use eager attention")
            token_count = int(model_inputs["input_ids"].shape[1])
            applicable = [category for category in CATEGORIES if probes.get(category)]
            for layer in selected_layers:
                attention = outputs.attentions[layer][0].float()
                module = model.model.layers[layer].self_attn
                hidden = captured_hidden[layer]
                hidden_shape = (*hidden.shape[:-1], -1, int(module.head_dim))
                # The pre-hook captures an inference tensor.  Keep the dependent
                # projection in inference mode as well; otherwise recent PyTorch
                # versions try to save that tensor for autograd and reject it.
                with torch.inference_mode():
                    values = module.v_proj(hidden).view(hidden_shape).transpose(1, 2)[0].float()
                if values.shape[0] != num_heads:
                    repeat_groups = num_heads // values.shape[0]
                    values = values.repeat_interleave(repeat_groups, dim=0)
                for category in applicable:
                    result = category_link_metrics(attention, values, probes[category])
                    query_counts = result["query_count_by_head"]
                    assert isinstance(query_counts, torch.Tensor)
                    for head in selected_heads:
                        row = {
                            "sample_id": sample.sample_id,
                            "domain": sample.domain,
                            "pair_id": sample.pair_id,
                            "layer": layer,
                            "head": head,
                            "category": category,
                            "query_count": int(query_counts[head].item()),
                            "token_count": token_count,
                        }
                        for metric in METRICS:
                            tensor = result[metric]
                            assert isinstance(tensor, torch.Tensor)
                            row[metric] = float(tensor[head].item())
                        sample_metric_rows.append(row)
            del outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"[sample {sample_index:02d}/{len(samples):02d}] {sample.sample_id} "
                f"tokens={token_count} categories={','.join(applicable)}",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    category_rows, ranking_rows, profile_rows = aggregate_rows(
        sample_metric_rows, selected_layers, selected_heads
    )
    write_csv(output_dir / "per_sample_link_distortion.csv", sample_metric_rows)
    write_csv(output_dir / "head_category_link_distortion.csv", category_rows)
    write_csv(output_dir / "category_link_distortion_rankings.csv", ranking_rows)
    write_csv(output_dir / "head_link_distortion_profiles.csv", profile_rows)
    plot_paths = make_plots(output_dir, profile_rows) if args.make_plots else []
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "sample_count": len(samples),
        "layers": selected_layers,
        "heads": selected_heads,
        "categories": list(CATEGORIES),
        "row_count": len(sample_metric_rows),
        "profile_count": len(profile_rows),
        "dominant_category_counts": dict(
            Counter(str(row["distortion_dominant_category"]) for row in profile_rows)
        ),
        "runtime_seconds": time.time() - started,
        "plot_paths": plot_paths,
        "interpretation": (
            "Exact local effect of removing category links and renormalizing the attention row. "
            "This is stronger than attention mass but is not an end-to-end NLL intervention."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
