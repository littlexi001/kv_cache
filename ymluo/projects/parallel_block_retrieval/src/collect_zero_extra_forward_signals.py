from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from evaluate_past_only_10m_retrieval_ppl import (
    barrier,
    read_jsonl,
    selected_context,
    setup_distributed,
)
from evaluate_xsum_news_ppl_retrieval import synchronize
from profile_real_qk import resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect deployment-visible surprisal, hidden-trajectory, and "
            "attention-response signals from one normal forward per current workset."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--retrieval_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--memory_tokens", type=int, default=100_000_000)
    parser.add_argument("--state_suffix_tokens", default="128,512")
    parser.add_argument("--current_scope_depths", default="3,8,16")
    parser.add_argument("--probe_tokens", type=int, default=64)
    parser.add_argument("--windows", default="8,16,32,64")
    parser.add_argument("--retrieval_blocks", type=int, default=8)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("integer list must contain positive values")
    return values


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=-1, eps=1.0e-8)


def vector_sequence_stats(values: torch.Tensor, windows: list[int]) -> dict[str, torch.Tensor]:
    values = values.float()
    norms = torch.linalg.vector_norm(values, dim=-1)
    mean_vector = values.mean(dim=0)
    output = {
        "norm_mean": norms.mean(),
        "norm_std": norms.std(unbiased=False),
        "adjacent_cosine_mean": (
            cosine(values[1:], values[:-1]).mean()
            if len(values) > 1
            else torch.ones((), device=values.device)
        ),
        "dispersion": 1.0 - cosine(values, mean_vector[None, :]).mean(),
    }
    for window in windows:
        if 2 * window > len(values):
            continue
        previous = values[-2 * window : -window].mean(dim=0)
        current = values[-window:].mean(dim=0)
        output[f"shift_{window}"] = 1.0 - cosine(
            previous[None, :], current[None, :]
        )[0]
    return output


class AttentionResponseCapture:
    def __init__(self, model: AutoModelForCausalLM, windows: list[int]) -> None:
        self.windows = windows
        self.start = 0
        self.end = 0
        self.enabled = False
        self.inputs: dict[int, torch.Tensor] = {}
        self.layer_stats: dict[int, dict[str, torch.Tensor]] = {}
        self.handles = []
        for layer_id, layer in enumerate(model.model.layers):
            attention = layer.self_attn
            self.handles.append(
                attention.register_forward_pre_hook(
                    self._pre_hook(layer_id), with_kwargs=True
                )
            )
            self.handles.append(
                attention.register_forward_hook(
                    self._post_hook(layer_id), with_kwargs=True
                )
            )

    def configure(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.enabled = True
        self.inputs.clear()
        self.layer_stats.clear()

    def _pre_hook(self, layer_id: int) -> Any:
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del module
            if not self.enabled:
                return
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if hidden is None:
                raise RuntimeError("attention pre-hook could not find hidden_states")
            self.inputs[layer_id] = hidden[0, self.start : self.end].detach().float()

        return hook

    def _post_hook(self, layer_id: int) -> Any:
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs
            if not self.enabled:
                return
            response = output[0] if isinstance(output, tuple) else output
            response = response[0, self.start : self.end].detach().float()
            hidden = self.inputs.pop(layer_id)
            hidden_norm = torch.linalg.vector_norm(hidden, dim=-1).clamp_min(1.0e-8)
            response_norm = torch.linalg.vector_norm(response, dim=-1)
            stats = vector_sequence_stats(response, self.windows)
            stats.update(
                {
                    "norm_ratio_mean": (response_norm / hidden_norm).mean(),
                    "residual_alignment_mean": cosine(response, hidden).mean(),
                }
            )
            self.layer_stats[layer_id] = stats

        return hook

    def finish(self) -> tuple[dict[str, float], dict[str, list[float]]]:
        self.enabled = False
        if self.inputs:
            raise RuntimeError("attention hook retained unmatched inputs")
        layer_ids = sorted(self.layer_stats)
        if not layer_ids:
            raise RuntimeError("attention hooks captured no layers")
        metric_names = sorted(
            set().union(*(self.layer_stats[layer_id] for layer_id in layer_ids))
        )
        by_layer: dict[str, list[float]] = {}
        compact: dict[str, float] = {}
        split = max(len(layer_ids) // 2, 1)
        for metric in metric_names:
            values = [
                float(self.layer_stats[layer_id][metric].item())
                if metric in self.layer_stats[layer_id]
                else math.nan
                for layer_id in layer_ids
            ]
            by_layer[metric] = values
            finite = np.asarray([value for value in values if math.isfinite(value)])
            late = np.asarray(
                [value for value in values[split:] if math.isfinite(value)]
            )
            compact[f"attention_{metric}_mean"] = float(finite.mean())
            compact[f"attention_{metric}_max"] = float(finite.max())
            compact[f"attention_{metric}_late_mean"] = float(late.mean())
            if len(finite) > 1:
                compact[f"attention_{metric}_layer_slope"] = float(
                    np.polyfit(np.arange(len(finite)), finite, 1)[0]
                )
        return compact, by_layer

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def scalar_sequence_features(
    name: str, values: torch.Tensor, windows: list[int]
) -> dict[str, float]:
    array = values.detach().float().cpu().numpy().astype(np.float64)
    output: dict[str, float] = {}
    for window in windows:
        current = array[-window:]
        output[f"{name}_mean_{window}"] = float(current.mean())
        output[f"{name}_std_{window}"] = float(current.std())
        if 2 * window <= len(array):
            previous = array[-2 * window : -window]
            output[f"{name}_change_{window}"] = float(
                current.mean() - previous.mean()
            )
    if len(array) > 1:
        output[f"{name}_slope"] = float(
            np.polyfit(np.arange(len(array)), array, 1)[0]
        )
    return output


@torch.inference_mode()
def normal_forward_signals(
    model: AutoModelForCausalLM,
    capture: AttentionResponseCapture,
    context_ids: np.ndarray,
    state_ids: np.ndarray,
    *,
    probe_tokens: int,
    windows: list[int],
    device: torch.device,
) -> tuple[dict[str, Any], float, int]:
    state = np.asarray(state_ids, dtype=np.int64)
    if probe_tokens >= len(state):
        raise ValueError("probe_tokens must be smaller than state length")
    context = torch.from_numpy(np.asarray(context_ids, dtype=np.int64))
    prefix = torch.from_numpy(state[:-probe_tokens])
    probe = torch.from_numpy(state[-probe_tokens:])
    prompt = torch.cat([context, prefix], dim=0)
    input_ids = torch.cat([prompt, probe], dim=0)[None, :].to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    prompt_tokens = int(prompt.numel())
    position_start = prompt_tokens - 1
    position_end = prompt_tokens + probe_tokens - 1
    capture.configure(position_start, position_end)

    synchronize(device)
    started = time.perf_counter()
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    positions = torch.arange(
        position_start, position_end, device=device, dtype=torch.long
    )
    final_hidden = outputs.last_hidden_state[0].index_select(0, positions)
    logits = model.lm_head(final_hidden).float()
    targets = input_ids[0, prompt_tokens : prompt_tokens + probe_tokens]
    log_probabilities = F.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    losses = F.nll_loss(log_probabilities, targets, reduction="none")
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    top2 = torch.topk(logits, 2, dim=-1).values
    top1_margin = top2[:, 0] - top2[:, 1]
    max_probability = probabilities.max(dim=-1).values
    target_probability = probabilities.gather(1, targets[:, None])[:, 0]

    signal_features: dict[str, float] = {}
    for name, values in (
        ("observed_surprisal", losses),
        ("predictive_entropy", entropy),
        ("top1_margin", top1_margin),
        ("max_probability", max_probability),
        ("observed_token_probability", target_probability),
    ):
        signal_features.update(scalar_sequence_features(name, values, windows))

    hidden_by_checkpoint: dict[str, dict[str, float]] = {}
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model did not return hidden states")
    checkpoints = sorted(
        {0, len(hidden_states) // 4, len(hidden_states) // 2, 3 * len(hidden_states) // 4, len(hidden_states) - 1}
    )
    for checkpoint in checkpoints:
        values = hidden_states[checkpoint][0].index_select(0, positions).float()
        stats = vector_sequence_stats(values, windows)
        converted = {name: float(value.item()) for name, value in stats.items()}
        hidden_by_checkpoint[str(checkpoint)] = converted
        for name, value in converted.items():
            signal_features[f"hidden_checkpoint{checkpoint}_{name}"] = value

    attention_compact, attention_by_layer = capture.finish()
    signal_features.update(attention_compact)
    synchronize(device)
    elapsed = time.perf_counter() - started
    output = {
        "signal_features": signal_features,
        "token_signals": {
            "observed_surprisal": losses.cpu().tolist(),
            "predictive_entropy": entropy.cpu().tolist(),
            "top1_margin": top1_margin.cpu().tolist(),
            "max_probability": max_probability.cpu().tolist(),
            "observed_token_probability": target_probability.cpu().tolist(),
        },
        "hidden_by_checkpoint": hidden_by_checkpoint,
        "attention_response_by_layer": attention_by_layer,
    }
    return output, elapsed, int(input_ids.shape[1])


def main() -> None:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size)

    suffixes = parse_ints(args.state_suffix_tokens)
    depths = parse_ints(args.current_scope_depths)
    windows = parse_ints(args.windows)
    if max(windows) > args.probe_tokens:
        raise ValueError("window exceeds probe_tokens")
    if min(suffixes) <= args.probe_tokens:
        raise ValueError("every state must leave a non-empty retrieval prefix")

    data_dir = Path(args.data_dir)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    if not data_summary.get("past_only") or data_summary.get("source_blocks") != 0:
        raise ValueError("requires past-only data without predefined source blocks")
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    query_count = len(queries) if args.max_queries <= 0 else min(len(queries), args.max_queries)
    retrieval_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in read_jsonl(args.retrieval_rows):
        if int(row["memory_tokens"]) != args.memory_tokens:
            continue
        method = str(row["method"])
        if not method.startswith("hier_bm25_scope"):
            continue
        depth = int(method.removeprefix("hier_bm25_scope"))
        suffix = int(row["prefix_tokens"])
        if depth in depths and suffix in suffixes:
            retrieval_lookup[(int(row["query_id"]), suffix, depth)] = row

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    capture = AttentionResponseCapture(model, windows)

    local_query_ids = [
        query_id for query_id in range(query_count) if query_id % world_size == rank
    ]
    rows = []
    for query_id in local_query_ids:
        for suffix in suffixes:
            state = np.asarray(queries[query_id, -suffix:], dtype=np.int32)
            for depth in depths:
                retrieval = retrieval_lookup[(query_id, suffix, depth)]
                selection = [
                    int(item) for item in retrieval["top_block_ids"][: args.retrieval_blocks]
                ]
                context = selected_context(selection, base_blocks)
                signals, seconds, model_input_tokens = normal_forward_signals(
                    model,
                    capture,
                    context,
                    state,
                    probe_tokens=args.probe_tokens,
                    windows=windows,
                    device=device,
                )
                rows.append(
                    {
                        "query_id": query_id,
                        "memory_tokens": args.memory_tokens,
                        "state_suffix_tokens": suffix,
                        "current_scope_depth": depth,
                        "method": f"hier_bm25_scope{depth}",
                        "selected_block_ids": selection,
                        "retrieved_tokens": len(selection) * int(data_summary["block_tokens"]),
                        "probe_tokens": args.probe_tokens,
                        "forward_seconds": seconds,
                        "model_input_tokens": model_input_tokens,
                        "retrieval_query_end_offset_tokens": int(
                            retrieval.get("query_end_offset_tokens", 0)
                        ),
                        "retrieval_query_uses_observed_probe_tokens": (
                            int(retrieval.get("query_end_offset_tokens", 0))
                            < args.probe_tokens
                        ),
                        "one_normal_forward_for_current_workset": True,
                        "expanded_workset_forward_used": False,
                        "future_target_used": False,
                        "selection_uses_target": False,
                        **signals,
                    }
                )
        print(f"rank={rank} completed query={query_id}", flush=True)

    capture.close()
    rank_path = output_dir / f"rows_rank{rank}.jsonl"
    with rank_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    barrier(world_size)
    if rank == 0:
        all_rows = []
        for worker_rank in range(world_size):
            all_rows.extend(read_jsonl(output_dir / f"rows_rank{worker_rank}.jsonl"))
        all_rows.sort(
            key=lambda row: (
                int(row["query_id"]),
                int(row["state_suffix_tokens"]),
                int(row["current_scope_depth"]),
            )
        )
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        summary = {
            "source": "zero-extra-candidate-forward online signal collection",
            "protocol": {
                "queries": query_count,
                "memory_tokens": args.memory_tokens,
                "state_suffix_tokens": suffixes,
                "current_scope_depths": depths,
                "probe_tokens": args.probe_tokens,
                "windows": windows,
                "retrieval_blocks": args.retrieval_blocks,
                "retrieval_query_uses_observed_probe_tokens": any(
                    bool(row["retrieval_query_uses_observed_probe_tokens"])
                    for row in all_rows
                ),
                "one_normal_forward_for_current_workset": True,
                "expanded_workset_forward_used": False,
                "future_target_used": False,
                "selection_uses_target": False,
                "world_size": world_size,
            },
            "rows": len(all_rows),
            "mean_forward_seconds": statistics.fmean(
                float(row["forward_seconds"]) for row in all_rows
            ),
            "mean_forward_seconds_by_state": {
                str(suffix): statistics.fmean(
                    float(row["forward_seconds"])
                    for row in all_rows
                    if int(row["state_suffix_tokens"]) == suffix
                )
                for suffix in suffixes
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    barrier(world_size)


if __name__ == "__main__":
    main()
