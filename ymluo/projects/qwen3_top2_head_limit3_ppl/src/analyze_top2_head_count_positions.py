from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    from transformers import AutoModelWithLMHead as AutoModelForCausalLM
    from transformers import AutoTokenizer


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze where historical tokens selected by 1..N top2 heads are located."
    )
    parser.add_argument("--model_name_or_path", default="ymluo/models/Qwen3-0.6B")
    parser.add_argument(
        "--text_path",
        default="external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt",
    )
    parser.add_argument("--output_dir", default="outputs/head_count_position_distribution")
    parser.add_argument("--prefill_tokens", type=int, default=1024)
    parser.add_argument("--eval_tokens", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument("--require_total_tokens", type=str2bool, default=True)
    parser.add_argument("--relative_bins", type=int, default=50)
    parser.add_argument("--sink_cutoffs", default="16,64,128,256")
    parser.add_argument("--recent_cutoffs", default="0.01,0.04,0.08,0.12,0.16")
    parser.add_argument("--sample_rows_per_count", type=int, default=200)
    parser.add_argument("--plot_dpi", type=int, default=180)
    return parser.parse_args()


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if device.type == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_chars) if max_chars > 0 else handle.read()


def model_forward(model: torch.nn.Module, kwargs: dict[str, Any]) -> Any:
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "cache_position" in kwargs and "cache_position" in str(exc):
            kwargs = dict(kwargs)
            kwargs.pop("cache_position")
            return model(**kwargs)
        raise


def pick_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def prefill_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    chunk_size: int,
    device: torch.device,
) -> Any:
    past = None
    for idx, start in enumerate(range(0, prefill_tokens, chunk_size), start=1):
        end = min(start + chunk_size, prefill_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(start, end, device=device),
        }
        if past is not None:
            kwargs["past_key_values"] = past
        print(f"prefill {idx}: {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past = outputs.past_key_values
    return past


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bucket_index(value: float, bins: int) -> int:
    return min(bins - 1, max(0, int(value * bins)))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sink_cutoffs = [int(part) for part in args.sink_cutoffs.split(",") if part.strip()]
    recent_cutoffs = [float(part) for part in args.recent_cutoffs.split(",") if part.strip()]
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top_fraction must be in (0, 1].")
    if args.relative_bins <= 0:
        raise ValueError("--relative_bins must be positive.")

    text = read_text_prefix(Path(args.text_path), args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    needed = args.prefill_tokens + args.eval_tokens
    if args.require_total_tokens and len(token_ids) < needed:
        raise ValueError(f"Need {needed} tokens, got {len(token_ids)}.")
    input_ids = torch.tensor(token_ids[:needed], dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype, requested_device)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": model_dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    device = pick_input_device(model, requested_device)
    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    past = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, device)

    count_cases = torch.zeros(head_count + 1, dtype=torch.long)
    rel_hist = torch.zeros((head_count + 1, args.relative_bins), dtype=torch.long)
    layer_count_cases = torch.zeros((layer_count, head_count + 1), dtype=torch.long)
    layer_rel_hist = torch.zeros((layer_count, head_count + 1, args.relative_bins), dtype=torch.long)
    layer_recent16 = torch.zeros((layer_count, head_count + 1), dtype=torch.long)
    distance_values: dict[int, list[float]] = {count: [] for count in range(1, head_count + 1)}
    rel_values: dict[int, list[float]] = {count: [] for count in range(1, head_count + 1)}
    abs_values: dict[int, list[float]] = {count: [] for count in range(1, head_count + 1)}
    sink_hits = {cutoff: torch.zeros(head_count + 1, dtype=torch.long) for cutoff in sink_cutoffs}
    recent_hits = {cutoff: torch.zeros(head_count + 1, dtype=torch.long) for cutoff in recent_cutoffs}
    sample_rows: list[dict[str, Any]] = []
    sample_seen = torch.zeros(head_count + 1, dtype=torch.long)

    eval_end = args.prefill_tokens + args.eval_tokens
    for chunk_idx, start in enumerate(range(args.prefill_tokens, eval_end, args.chunk_size), start=1):
        end = min(start + args.chunk_size, eval_end)
        kwargs = {
            "input_ids": input_ids[:, start:end].to(device),
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": True,
            "cache_position": torch.arange(start, end, device=device),
        }
        print(f"head-count position chunk {chunk_idx}: {start}-{end - 1}", flush=True)
        outputs = model_forward(model, kwargs)
        past = outputs.past_key_values
        for layer in range(layer_count):
            attention = outputs.attentions[layer][0].detach().float().cpu()
            for local_query in range(end - start):
                query_token = start + local_query
                history_count = query_token
                top_k = min(history_count, max(1, math.ceil(args.top_fraction * history_count)))
                if top_k <= 0:
                    continue
                _, indices = torch.topk(attention[:, local_query, :query_token], k=top_k, dim=-1)
                selected_counts = torch.zeros(history_count, dtype=torch.long)
                selected_counts.scatter_add_(0, indices.reshape(-1), torch.ones(indices.numel(), dtype=torch.long))
                selected_tokens = torch.nonzero(selected_counts > 0, as_tuple=False).flatten()
                for key_index in selected_tokens.tolist():
                    selected_head_count = int(selected_counts[key_index].item())
                    rel_position = key_index / max(1, history_count - 1)
                    distance = query_token - key_index
                    bin_index = bucket_index(rel_position, args.relative_bins)
                    count_cases[selected_head_count] += 1
                    rel_hist[selected_head_count, bin_index] += 1
                    layer_count_cases[layer, selected_head_count] += 1
                    layer_rel_hist[layer, selected_head_count, bin_index] += 1
                    if distance <= max(1, math.ceil(0.16 * history_count)):
                        layer_recent16[layer, selected_head_count] += 1
                    rel_values[selected_head_count].append(rel_position)
                    distance_values[selected_head_count].append(float(distance))
                    abs_values[selected_head_count].append(float(key_index))
                    for cutoff in sink_cutoffs:
                        if key_index < cutoff:
                            sink_hits[cutoff][selected_head_count] += 1
                    for cutoff in recent_cutoffs:
                        if distance <= max(1, math.ceil(cutoff * history_count)):
                            recent_hits[cutoff][selected_head_count] += 1
                    if sample_seen[selected_head_count] < args.sample_rows_per_count:
                        sample_rows.append(
                            {
                                "layer": layer,
                                "query_token": query_token,
                                "key_token": key_index,
                                "selected_head_count": selected_head_count,
                                "distance_from_query": distance,
                                "relative_key_position": rel_position,
                                "top_k_per_head": top_k,
                            }
                        )
                        sample_seen[selected_head_count] += 1
        del outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    total_selected_tokens = int(count_cases.sum().item())
    for selected_head_count in range(1, head_count + 1):
        case_count = int(count_cases[selected_head_count].item())
        rel_sorted = sorted(rel_values[selected_head_count])
        dist_sorted = sorted(distance_values[selected_head_count])
        abs_sorted = sorted(abs_values[selected_head_count])
        row: dict[str, Any] = {
            "selected_head_count": selected_head_count,
            "token_cases": case_count,
            "fraction_of_selected_token_cases": case_count / total_selected_tokens if total_selected_tokens else 0.0,
            "relative_position_mean": sum(rel_sorted) / case_count if case_count else 0.0,
            "relative_position_p05": percentile(rel_sorted, 0.05),
            "relative_position_p25": percentile(rel_sorted, 0.25),
            "relative_position_p50": percentile(rel_sorted, 0.50),
            "relative_position_p75": percentile(rel_sorted, 0.75),
            "relative_position_p95": percentile(rel_sorted, 0.95),
            "distance_mean": sum(dist_sorted) / case_count if case_count else 0.0,
            "distance_p05": percentile(dist_sorted, 0.05),
            "distance_p25": percentile(dist_sorted, 0.25),
            "distance_p50": percentile(dist_sorted, 0.50),
            "distance_p75": percentile(dist_sorted, 0.75),
            "distance_p95": percentile(dist_sorted, 0.95),
            "absolute_position_p50": percentile(abs_sorted, 0.50),
        }
        for cutoff in sink_cutoffs:
            row[f"fraction_key_lt_{cutoff}"] = (
                int(sink_hits[cutoff][selected_head_count].item()) / case_count if case_count else 0.0
            )
        for cutoff in recent_cutoffs:
            label = str(cutoff).replace(".", "p")
            row[f"fraction_recent_{label}"] = (
                int(recent_hits[cutoff][selected_head_count].item()) / case_count if case_count else 0.0
            )
        summary_rows.append(row)

    summary_fields = [
        "selected_head_count",
        "token_cases",
        "fraction_of_selected_token_cases",
        "relative_position_mean",
        "relative_position_p05",
        "relative_position_p25",
        "relative_position_p50",
        "relative_position_p75",
        "relative_position_p95",
        "distance_mean",
        "distance_p05",
        "distance_p25",
        "distance_p50",
        "distance_p75",
        "distance_p95",
        "absolute_position_p50",
    ]
    summary_fields.extend([f"fraction_key_lt_{cutoff}" for cutoff in sink_cutoffs])
    summary_fields.extend([f"fraction_recent_{str(cutoff).replace('.', 'p')}" for cutoff in recent_cutoffs])
    write_csv(output_dir / "head_count_position_summary.csv", summary_rows, summary_fields)

    hist_rows: list[dict[str, Any]] = []
    for selected_head_count in range(1, head_count + 1):
        case_count = int(count_cases[selected_head_count].item())
        for bin_index in range(args.relative_bins):
            bin_count = int(rel_hist[selected_head_count, bin_index].item())
            hist_rows.append(
                {
                    "selected_head_count": selected_head_count,
                    "relative_bin_left": bin_index / args.relative_bins,
                    "relative_bin_right": (bin_index + 1) / args.relative_bins,
                    "token_cases": bin_count,
                    "fraction_within_selected_head_count": bin_count / case_count if case_count else 0.0,
                    "fraction_of_all_selected_token_cases": bin_count / total_selected_tokens if total_selected_tokens else 0.0,
                }
            )
    write_csv(
        output_dir / "relative_position_histogram_by_head_count.csv",
        hist_rows,
        [
            "selected_head_count",
            "relative_bin_left",
            "relative_bin_right",
            "token_cases",
            "fraction_within_selected_head_count",
            "fraction_of_all_selected_token_cases",
        ],
    )

    layer_rows: list[dict[str, Any]] = []
    for layer in range(layer_count):
        for selected_head_count in range(1, head_count + 1):
            case_count = int(layer_count_cases[layer, selected_head_count].item())
            layer_rows.append(
                {
                    "layer": layer,
                    "selected_head_count": selected_head_count,
                    "token_cases": case_count,
                    "fraction_recent_0p16": int(layer_recent16[layer, selected_head_count].item()) / case_count
                    if case_count
                    else 0.0,
                }
            )
    write_csv(output_dir / "layer_head_count_summary.csv", layer_rows, ["layer", "selected_head_count", "token_cases", "fraction_recent_0p16"])
    write_csv(
        output_dir / "sample_token_cases.csv",
        sample_rows,
        [
            "layer",
            "query_token",
            "key_token",
            "selected_head_count",
            "distance_from_query",
            "relative_key_position",
            "top_k_per_head",
        ],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heat = torch.zeros((args.relative_bins, head_count), dtype=torch.float32)
    for selected_head_count in range(1, head_count + 1):
        case_count = int(count_cases[selected_head_count].item())
        if case_count:
            heat[:, selected_head_count - 1] = rel_hist[selected_head_count].float() / case_count
    fig, ax = plt.subplots(figsize=(10, 7), dpi=args.plot_dpi)
    image = ax.imshow(heat.numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Relative position distribution by number of selecting heads")
    ax.set_xlabel("Number of heads that selected the same historical token")
    ax.set_ylabel("Relative historical position: 0 = sequence start, 1 = query-near")
    ax.set_xticks(list(range(head_count)))
    ax.set_xticklabels([str(i) for i in range(1, head_count + 1)])
    y_ticks = [0, args.relative_bins // 4, args.relative_bins // 2, 3 * args.relative_bins // 4, args.relative_bins - 1]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick / args.relative_bins:.2f}" for tick in y_ticks])
    fig.colorbar(image, ax=ax, label="Fraction within this selected-head-count group")
    fig.tight_layout()
    fig.savefig(output_dir / "relative_position_distribution_by_head_count.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=args.plot_dpi)
    x_values = [row["selected_head_count"] for row in summary_rows]
    ax.plot(x_values, [row["fraction_key_lt_64"] for row in summary_rows], marker="o", label="key position < 64")
    ax.plot(x_values, [row["fraction_recent_0p01"] for row in summary_rows], marker="o", label="recent 1%")
    ax.plot(x_values, [row["fraction_recent_0p08"] for row in summary_rows], marker="o", label="recent 8%")
    ax.plot(x_values, [row["fraction_recent_0p16"] for row in summary_rows], marker="o", label="recent 16%")
    ax.set_title("Sink and recent-token fraction by number of selecting heads")
    ax.set_xlabel("Number of heads that selected the same historical token")
    ax.set_ylabel("Fraction of selected token cases")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sink_recent_fraction_by_head_count.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=args.plot_dpi)
    ax.plot(x_values, [row["relative_position_p50"] for row in summary_rows], marker="o", label="median")
    ax.fill_between(
        x_values,
        [row["relative_position_p25"] for row in summary_rows],
        [row["relative_position_p75"] for row in summary_rows],
        alpha=0.2,
        label="25%-75%",
    )
    ax.set_title("Median relative position by number of selecting heads")
    ax.set_xlabel("Number of heads that selected the same historical token")
    ax.set_ylabel("Relative historical position: 0 = sequence start, 1 = query-near")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "relative_position_quantiles_by_head_count.png")
    plt.close(fig)

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "layer_count": layer_count,
                "head_count": head_count,
                "total_selected_token_cases": total_selected_tokens,
                "sink_cutoffs": sink_cutoffs,
                "recent_cutoffs": recent_cutoffs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
