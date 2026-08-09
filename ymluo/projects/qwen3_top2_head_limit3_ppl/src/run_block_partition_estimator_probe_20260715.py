from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_controlled_public_kv_benchmark_v1 as lb  # noqa: E402
from run_critical_position_budget_probe_20260715 import load_model  # noqa: E402
from run_head_top2_targeted_ppl_20260714 import (  # noqa: E402
    _fit_attention_mask,
    parse_int_list,
    set_attention_implementation,
)
from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    encode_topic_stream,
    make_bundle,
    topic_names,
)


_ORIGINAL_LLAMA_EAGER: Any | None = None
_ACTIVE_COLLECTOR: list[dict[str, Any]] | None = None
_ACTIVE_BLOCK_SIZES: tuple[int, ...] = ()
_ACTIVE_THRESHOLDS: tuple[float, ...] = ()
_ACTIVE_SAMPLE_FRACTIONS: tuple[float, ...] = ()
_ACTIVE_BUDGET_FRACTIONS: tuple[float, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-position block softmax-partition estimator diagnostic."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--topics", default="sports,medicine")
    parser.add_argument("--window_indices", default="0,1,2")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--block_sizes", default="32,64,128,256,512")
    parser.add_argument("--mass_thresholds", default="0.75,0.80,0.85,0.90")
    parser.add_argument("--sample_fractions", default="0.0025,0.005,0.01")
    parser.add_argument("--budget_fractions", default="0.0025,0.005,0.01,0.02,0.04")
    parser.add_argument("--prefill_chunk_tokens", type=int, default=2048)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def parse_int_values(spec: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in spec.split(",") if item.strip()}))
    if not values or values[0] <= 0:
        raise ValueError("expected positive integers")
    return values


def parse_float_values(spec: str) -> tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in spec.split(",") if item.strip()}))
    if not values or values[0] <= 0.0 or values[-1] >= 1.0:
        raise ValueError("expected values in (0, 1)")
    return values


def _pearson_per_head(left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(left.dtype)
    count = mask.sum(dim=-1).clamp_min(1.0)
    left_mean = (left * mask).sum(dim=-1) / count
    right_mean = (right * mask).sum(dim=-1) / count
    left_centered = (left - left_mean.unsqueeze(-1)) * mask
    right_centered = (right - right_mean.unsqueeze(-1)) * mask
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = torch.sqrt(
        left_centered.square().sum(dim=-1) * right_centered.square().sum(dim=-1)
    ).clamp_min(1e-12)
    return numerator / denominator


def block_partition_metrics(
    scores: torch.Tensor,
    block_size: int,
    thresholds: tuple[float, ...],
) -> list[dict[str, float | str]]:
    """Evaluate block log-partition estimators from exact token scores.

    ``scores`` has shape [batch, heads, history]. The centroid estimator is
    mathematically log(n) + q dot mean(K). The diagonal/projected-variance
    estimator adds 0.5 Var(q dot K), matching a Gaussian moment model.
    """
    if scores.ndim != 3 or scores.shape[-1] == 0:
        raise ValueError("scores must have shape [batch, heads, nonempty history]")
    batch_count, head_count, history_count = scores.shape
    block_count = math.ceil(history_count / block_size)
    padded_count = block_count * block_size
    pad_count = padded_count - history_count
    work = scores.float()
    if pad_count:
        work = torch.cat(
            (work, torch.zeros((batch_count, head_count, pad_count), device=work.device)), dim=-1
        )
    blocks = work.view(batch_count, head_count, block_count, block_size)
    valid_flat = torch.arange(padded_count, device=work.device) < history_count
    valid_tokens = valid_flat.view(1, 1, block_count, block_size)
    counts = valid_tokens.sum(dim=-1).float().expand(batch_count, head_count, -1).clamp_min(1.0)
    means = (blocks * valid_tokens).sum(dim=-1) / counts
    centered = (blocks - means.unsqueeze(-1)) * valid_tokens
    variances = centered.square().sum(dim=-1) / counts
    masked_blocks = blocks.masked_fill(~valid_tokens, -torch.inf)
    true_logz = torch.logsumexp(masked_blocks, dim=-1)
    centroid_logz = counts.log() + means
    gaussian_logz = centroid_logz + 0.5 * variances
    valid_blocks = counts > 0

    rows: list[dict[str, float | str]] = []
    for estimator, estimate in [("centroid", centroid_logz), ("gaussian_variance", gaussian_logz)]:
        errors = (estimate - true_logz).abs()[valid_blocks]
        correlations = _pearson_per_head(estimate, true_logz, valid_blocks)
        true_total = torch.logsumexp(true_logz, dim=-1)
        estimated_total = torch.logsumexp(estimate, dim=-1)
        base = {
            "estimator": estimator,
            "block_size": float(block_size),
            "block_logz_mae": float(errors.mean().item()),
            "block_logz_p90_error": float(torch.quantile(errors, 0.90).item()),
            "mean_head_correlation": float(correlations.mean().item()),
            "total_logz_mae": float((estimated_total - true_total).abs().mean().item()),
        }

        order = torch.argsort(estimate, dim=-1, descending=True)
        estimated_probs = torch.softmax(estimate, dim=-1)
        true_probs = torch.softmax(true_logz, dim=-1)
        ordered_estimated = estimated_probs.gather(-1, order)
        ordered_true = true_probs.gather(-1, order)
        cumulative_estimated = ordered_estimated.cumsum(dim=-1)
        cumulative_true = ordered_true.cumsum(dim=-1)
        for threshold in thresholds:
            reached = cumulative_estimated >= threshold
            reached[..., -1] = True
            selected_count = reached.to(torch.int64).argmax(dim=-1) + 1
            actual_mass = cumulative_true.gather(-1, (selected_count - 1).unsqueeze(-1)).squeeze(-1)
            token_ratio = (selected_count.float() * float(block_size) / float(history_count)).clamp_max(1.0)
            rows.append(
                {
                    **base,
                    "mass_threshold": float(threshold),
                    "mean_selected_token_ratio": float(token_ratio.mean().item()),
                    "mean_actual_mass": float(actual_mass.mean().item()),
                    "actual_mass_p10": float(torch.quantile(actual_mass, 0.10).item()),
                    "coverage_rate": float((actual_mass >= threshold).float().mean().item()),
                }
            )
    return rows


def sampled_tail_partition_metrics(
    scores: torch.Tensor,
    budget_fractions: tuple[float, ...],
    thresholds: tuple[float, ...],
    sample_fraction: float,
    seed: int,
) -> list[dict[str, float | str]]:
    """Combine exact high-score candidates with a uniform residual sample."""
    if scores.ndim != 3 or scores.shape[-1] <= 1:
        raise ValueError("scores must contain history and self")
    history = scores[..., :-1].float()
    self_scores = scores[..., -1].float()
    history_count = int(history.shape[-1])
    fractions = tuple(sorted({float(value) for value in budget_fractions}))
    keep_counts = tuple(
        min(history_count, max(1, math.ceil(fraction * history_count))) for fraction in fractions
    )
    top_scores = torch.topk(history, k=keep_counts[-1], dim=-1, largest=True, sorted=True).values
    full_logz = torch.logsumexp(scores.float(), dim=-1)

    sample_count = min(history_count, max(1, math.ceil(sample_fraction * history_count)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    sample_indices = torch.randperm(history_count, generator=generator)[:sample_count].to(history.device)
    sample_scores = history.index_select(-1, sample_indices)

    estimated_masses = []
    actual_masses = []
    estimated_totals = []
    for keep_count in keep_counts:
        candidate_logz = torch.logaddexp(
            torch.logsumexp(top_scores[..., :keep_count], dim=-1), self_scores
        )
        cutoff = top_scores[..., keep_count - 1 : keep_count]
        residual_sample = sample_scores.masked_fill(sample_scores >= cutoff, -torch.inf)
        residual_logz = torch.logsumexp(residual_sample, dim=-1) + math.log(
            float(history_count) / float(sample_count)
        )
        estimated_total = torch.logaddexp(candidate_logz, residual_logz)
        estimated_masses.append(torch.exp(candidate_logz - estimated_total).clamp(max=1.0))
        actual_masses.append(torch.exp(candidate_logz - full_logz).clamp(max=1.0))
        estimated_totals.append(estimated_total)

    estimated_stack = torch.stack(estimated_masses, dim=0)
    actual_stack = torch.stack(actual_masses, dim=0)
    total_stack = torch.stack(estimated_totals, dim=0)
    rows: list[dict[str, float | str]] = []
    fraction_options = torch.tensor(fractions, dtype=torch.float32, device=scores.device)
    for threshold in thresholds:
        reached = estimated_stack >= threshold
        reached[-1] = True
        chosen = reached.to(torch.int64).argmax(dim=0)
        gather_index = chosen.unsqueeze(0)
        estimated_mass = estimated_stack.gather(0, gather_index).squeeze(0)
        actual_mass = actual_stack.gather(0, gather_index).squeeze(0)
        estimated_total = total_stack.gather(0, gather_index).squeeze(0)
        selected_fraction = fraction_options[chosen]
        rows.append(
            {
                "estimator": f"tail_sample_{100.0 * sample_fraction:g}pct".replace(".", "p"),
                "block_size": 0.0,
                "block_logz_mae": 0.0,
                "block_logz_p90_error": 0.0,
                "mean_head_correlation": 0.0,
                "total_logz_mae": float((estimated_total - full_logz).abs().mean().item()),
                "mass_threshold": float(threshold),
                "mean_selected_token_ratio": float(selected_fraction.mean().item()),
                "mean_actual_mass": float(actual_mass.mean().item()),
                "actual_mass_p10": float(torch.quantile(actual_mass, 0.10).item()),
                "coverage_rate": float((actual_mass >= threshold).float().mean().item()),
                "estimated_mass_mae": float((estimated_mass - actual_mass).abs().mean().item()),
            }
        )
    return rows


def _patched_llama_eager(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if _ORIGINAL_LLAMA_EAGER is None:
        raise RuntimeError("collector patch is not installed")
    if _ACTIVE_COLLECTOR is not None:
        if query.shape[-2] != 1:
            raise RuntimeError("partition probe requires one query token")
        if key.shape[1] != query.shape[1]:
            groups = query.shape[1] // key.shape[1]
            expanded_key = key.repeat_interleave(groups, dim=1)
        else:
            expanded_key = key
        fitted_mask = _fit_attention_mask(attention_mask, int(expanded_key.shape[-2]))
        scores = torch.matmul(query, expanded_key.transpose(2, 3)) * float(scaling)
        if fitted_mask is not None:
            scores = scores + fitted_mask
        history_scores = scores[..., :-1].squeeze(2)
        layer_idx = int(getattr(module, "layer_idx", len(_ACTIVE_COLLECTOR)))
        for block_size in _ACTIVE_BLOCK_SIZES:
            for row in block_partition_metrics(history_scores, block_size, _ACTIVE_THRESHOLDS):
                row["layer"] = layer_idx
                _ACTIVE_COLLECTOR.append(row)
        for sample_fraction in _ACTIVE_SAMPLE_FRACTIONS:
            for row in sampled_tail_partition_metrics(
                scores.squeeze(2),
                _ACTIVE_BUDGET_FRACTIONS,
                _ACTIVE_THRESHOLDS,
                sample_fraction,
                seed=20260715 + layer_idx,
            ):
                row["layer"] = layer_idx
                _ACTIVE_COLLECTOR.append(row)
    return _ORIGINAL_LLAMA_EAGER(
        module,
        query,
        key,
        value,
        _fit_attention_mask(attention_mask, int(key.shape[-2])),
        scaling,
        dropout=dropout,
        **kwargs,
    )


def install_patch() -> None:
    global _ORIGINAL_LLAMA_EAGER
    import transformers.models.llama.modeling_llama as modeling_llama

    if _ORIGINAL_LLAMA_EAGER is None:
        _ORIGINAL_LLAMA_EAGER = modeling_llama.eager_attention_forward
    modeling_llama.eager_attention_forward = _patched_llama_eager
    if hasattr(modeling_llama, "ALL_ATTENTION_FUNCTIONS"):
        modeling_llama.ALL_ATTENTION_FUNCTIONS["eager"] = _patched_llama_eager


@contextmanager
def collect_partition_stats(
    records: list[dict[str, Any]],
    block_sizes: tuple[int, ...],
    thresholds: tuple[float, ...],
    sample_fractions: tuple[float, ...],
    budget_fractions: tuple[float, ...],
) -> Iterator[None]:
    global _ACTIVE_COLLECTOR, _ACTIVE_BLOCK_SIZES, _ACTIVE_THRESHOLDS
    global _ACTIVE_SAMPLE_FRACTIONS, _ACTIVE_BUDGET_FRACTIONS
    previous = (
        _ACTIVE_COLLECTOR,
        _ACTIVE_BLOCK_SIZES,
        _ACTIVE_THRESHOLDS,
        _ACTIVE_SAMPLE_FRACTIONS,
        _ACTIVE_BUDGET_FRACTIONS,
    )
    _ACTIVE_COLLECTOR = records
    _ACTIVE_BLOCK_SIZES = block_sizes
    _ACTIVE_THRESHOLDS = thresholds
    _ACTIVE_SAMPLE_FRACTIONS = sample_fractions
    _ACTIVE_BUDGET_FRACTIONS = budget_fractions
    try:
        yield
    finally:
        (
            _ACTIVE_COLLECTOR,
            _ACTIVE_BLOCK_SIZES,
            _ACTIVE_THRESHOLDS,
            _ACTIVE_SAMPLE_FRACTIONS,
            _ACTIVE_BUDGET_FRACTIONS,
        ) = previous


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["estimator"]), int(row["block_size"]), float(row["mass_threshold"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (estimator, block_size, threshold), subset in sorted(groups.items()):
        output.append(
            {
                "estimator": estimator,
                "block_size": block_size,
                "mass_threshold": threshold,
                "layers": len(subset),
                **{
                    key: sum(float(row[key]) for row in subset) / len(subset)
                    for key in [
                        "block_logz_mae",
                        "block_logz_p90_error",
                        "mean_head_correlation",
                        "total_logz_mae",
                        "mean_selected_token_ratio",
                        "mean_actual_mass",
                        "actual_mass_p10",
                        "coverage_rate",
                    ]
                },
            }
        )
    return output


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    block_sizes = parse_int_values(args.block_sizes)
    thresholds = parse_float_values(args.mass_thresholds)
    sample_fractions = parse_float_values(args.sample_fractions)
    budget_fractions = parse_float_values(args.budget_fractions)
    topics = topic_names(args.topics)
    windows = parse_int_list(args.window_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    tokenizer, model, input_device = load_model(args)
    install_patch()
    required_tokens = max(windows) * args.window_stride_tokens + args.history_tokens + args.eval_tokens
    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    for topic in topics:
        stream = encode_topic_stream(tokenizer, TOPICS[topic], required_tokens, args.dataset_cache_dir, args.seed)
        for window in windows:
            start = window * args.window_stride_tokens
            history = stream[start : start + args.history_tokens]
            remote_ids = history[: -args.query_tokens]
            query_ids = history[-args.query_tokens :]
            target_id = int(stream[start + args.history_tokens])
            bundle, _ = make_bundle(tokenizer, remote_ids, page_tokens=16)
            set_attention_implementation(model, "sdpa")
            cache, prefill_seconds = lb.prefill_prefix(model, bundle, input_device, args.prefill_chunk_tokens)
            if len(query_ids) > 1:
                cache, _, query_seconds = lb.run_token_segment(
                    model,
                    torch.tensor([query_ids[:-1]], dtype=torch.long, device=input_device),
                    cache,
                    len(remote_ids),
                    input_device,
                    64,
                    False,
                )
            else:
                query_seconds = 0.0

            set_attention_implementation(model, "eager")
            records: list[dict[str, Any]] = []
            with collect_partition_stats(
                records, block_sizes, thresholds, sample_fractions, budget_fractions
            ):
                outputs = lb.model_forward(
                    model,
                    {
                        "input_ids": torch.tensor([[query_ids[-1]]], dtype=torch.long, device=input_device),
                        "past_key_values": cache,
                        "use_cache": True,
                        "return_dict": True,
                        "cache_position": torch.tensor([len(history) - 1], dtype=torch.long, device=input_device),
                    },
                )
            log_probs = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)
            case_summary = summarize(records)
            for row in records:
                row.update({"topic": topic, "window": window})
            for row in case_summary:
                row.update(
                    {
                        "topic": topic,
                        "window": window,
                        "target_nll_full": -float(log_probs[target_id].item()),
                        "prefill_seconds": prefill_seconds,
                        "query_seconds": query_seconds,
                    }
                )
            all_rows.extend(records)
            all_summaries.extend(case_summary)
            print(json.dumps(case_summary, indent=2), flush=True)
            write_csv(args.output_dir / "layer_results.csv", all_rows)
            write_csv(args.output_dir / "summary.csv", all_summaries)

    (args.output_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
