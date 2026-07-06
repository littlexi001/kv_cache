from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.replace(",", " ").split():
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def time_stage(self, name: str):
        timer = self

        class _Context:
            def __enter__(self) -> None:
                if not timer.enabled:
                    self.start = 0.0
                    return
                synchronize(timer.device)
                self.start = time.perf_counter()

            def __exit__(self, exc_type, exc, tb) -> None:
                if not timer.enabled:
                    return
                synchronize(timer.device)
                timer.seconds[name] += time.perf_counter() - self.start
                timer.calls[name] += 1

        return _Context()


def timed_average(fn, device: torch.device, iterations: int) -> float:
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / float(iterations)


def indices_from_keep_mask(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected_counts = keep.sum(dim=-1)
    max_selected = int(selected_counts.max().item()) if selected_counts.numel() else 0
    if max_selected <= 0:
        empty_indices = torch.zeros((*keep.shape[:-1], 0), dtype=torch.long, device=keep.device)
        empty_valid = torch.zeros_like(empty_indices, dtype=torch.bool)
        return empty_indices, empty_valid
    positions = torch.arange(keep.shape[-1], device=keep.device, dtype=torch.long).view(1, 1, -1)
    positions = positions.expand_as(keep)
    masked_positions = torch.where(keep, positions, torch.full_like(positions, -1))
    indices = torch.topk(masked_positions, k=max_selected, dim=-1, largest=True).values
    valid = indices >= 0
    return indices.clamp_min(0), valid


def final_indices_from_selected(
    selected_history_indices: torch.Tensor,
    selected_valid: torch.Tensor,
    history_count: int,
    key_count: int,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
    always_keep_self: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count, head_count, _ = selected_history_indices.shape
    filtered_valid = selected_valid.clone()
    sink_end = min(protect_sink_tokens, history_count) if protect_sink_tokens > 0 else 0
    recent_start = history_count
    if protect_recent_tokens > 0:
        recent_start = max(0, history_count - protect_recent_tokens)
        filtered_valid &= ~((selected_history_indices >= recent_start) & (selected_history_indices < history_count))
    if sink_end > 0:
        filtered_valid &= selected_history_indices >= sink_end

    index_parts = [selected_history_indices]
    valid_parts = [filtered_valid]
    if sink_end > 0:
        sink_indices = torch.arange(sink_end, device=selected_history_indices.device, dtype=torch.long).view(1, 1, -1)
        index_parts.append(sink_indices.expand(batch_count, head_count, -1))
        valid_parts.append(torch.ones((batch_count, head_count, sink_end), dtype=torch.bool, device=selected_history_indices.device))
    if protect_recent_tokens > 0 and recent_start < history_count:
        recent_start_no_overlap = max(recent_start, sink_end)
        if recent_start_no_overlap < history_count:
            recent_indices = torch.arange(
                recent_start_no_overlap,
                history_count,
                device=selected_history_indices.device,
                dtype=torch.long,
            ).view(1, 1, -1)
            index_parts.append(recent_indices.expand(batch_count, head_count, -1))
            valid_parts.append(
                torch.ones(
                    (batch_count, head_count, history_count - recent_start_no_overlap),
                    dtype=torch.bool,
                    device=selected_history_indices.device,
                )
            )
    if always_keep_self:
        self_indices = torch.full((batch_count, head_count, 1), key_count - 1, dtype=torch.long, device=selected_history_indices.device)
        index_parts.append(self_indices)
        valid_parts.append(torch.ones_like(self_indices, dtype=torch.bool))
    return torch.cat(index_parts, dim=-1), torch.cat(valid_parts, dim=-1)


def full_attention_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    scores = torch.matmul(query[:, :, None, :], key.transpose(2, 3)).squeeze(2) * scaling
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.sum(weights[:, :, :, None] * value, dim=2)


def sparse_final_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_indices: torch.Tensor,
    final_valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    head_dim = query.shape[-1]
    gather_index = final_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
    selected_keys = torch.gather(key, dim=2, index=gather_index)
    selected_values = torch.gather(value, dim=2, index=gather_index)
    selected_scores = (selected_keys.float() * query[:, :, None, :].float()).sum(dim=-1) * scaling
    selected_scores = selected_scores.masked_fill(~final_valid, torch.finfo(selected_scores.dtype).min)
    weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.sum(weights[:, :, :, None] * selected_values, dim=2)


def load_qabs_kernels(use_kernels: bool) -> Any | None:
    if not use_kernels:
        return None
    try:
        import qabs_cuda_kernels

        return qabs_cuda_kernels
    except Exception as exc:
        print(f"warning: qabs CUDA kernels unavailable; using torch fallback ({exc})", flush=True)
        return None


def make_random_mask(
    batch_count: int,
    head_count: int,
    history_count: int,
    keep_fraction: float,
    device: torch.device,
) -> torch.Tensor:
    keep_count = min(history_count, max(1, math.ceil(keep_fraction * history_count)))
    scores = torch.rand((batch_count, head_count, history_count), device=device)
    indices = torch.topk(scores, k=keep_count, dim=-1, largest=True).indices
    mask = torch.zeros((batch_count, head_count, history_count), dtype=torch.bool, device=device)
    mask.scatter_(dim=-1, index=indices, value=True)
    return mask


def qabs_attention_once(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    previous_candidate: torch.Tensor | None,
    previous_final: torch.Tensor | None,
    key_dim_major: torch.Tensor | None,
    args: argparse.Namespace,
    kernels: Any | None,
    timer: StageTimer,
) -> torch.Tensor:
    batch_count, head_count, key_count, head_dim = key.shape
    history_count = key_count - 1
    key_history = key[:, :, :history_count, :]
    selected_dim_count = min(max(1, args.qabs_dim_count), head_dim)
    requested = min(history_count, max(1, math.ceil(args.qabs_candidate_fraction * history_count)))
    keep_count = min(history_count, max(1, math.ceil(args.top_fraction * history_count)))

    with timer.time_stage("qdim_topk"):
        qdim_indices = torch.topk(query.float().abs(), k=selected_dim_count, dim=-1, largest=True).indices

    with timer.time_stage("partial_scores"):
        if kernels is not None and args.partial_impl == "cuda_dim_major" and key_dim_major is not None:
            partial_scores = kernels.partial_scores_dim_major(query.contiguous(), key_dim_major, qdim_indices.contiguous())
        elif kernels is not None and args.partial_impl == "cuda":
            partial_scores = kernels.partial_scores(query.contiguous(), key_history, selected_dim_count)
        else:
            q_selected = torch.gather(query.float(), dim=-1, index=qdim_indices)
            k_indices = qdim_indices[:, :, None, :].expand(batch_count, head_count, history_count, selected_dim_count)
            k_selected = torch.gather(key_history.float(), dim=-1, index=k_indices)
            partial_scores = (k_selected * q_selected[:, :, None, :]).sum(dim=-1)

    with timer.time_stage("candidate_select"):
        threshold = torch.topk(partial_scores, k=requested, dim=-1, largest=True).values[:, :, -1:]
        current_candidate = partial_scores >= threshold

    with timer.time_stage("candidate_union"):
        candidate_union = current_candidate.clone()
        if previous_candidate is not None and args.mode == "qabs8cand3reuse":
            candidate_union |= previous_candidate
        if previous_final is not None:
            candidate_union |= previous_final

    candidate_scores_are_history = False
    with timer.time_stage("candidate_full_scores"):
        if kernels is not None and args.use_cuda_full_scores:
            candidate_scores = kernels.candidate_full_scores(
                query.contiguous(),
                key_history,
                candidate_union.contiguous(),
                None,
                None,
                int(args.protect_sink_tokens),
                int(args.protect_recent_tokens),
                float(args.scaling),
            )
            candidate_scores_are_history = True
        else:
            candidate_indices, candidate_valid = indices_from_keep_mask(candidate_union)
            if candidate_indices.shape[-1] == 0:
                candidate_indices = torch.zeros((batch_count, head_count, 1), dtype=torch.long, device=query.device)
                candidate_valid = torch.zeros_like(candidate_indices, dtype=torch.bool)
            gather_index = candidate_indices[:, :, :, None].expand(-1, -1, -1, head_dim)
            candidate_keys = torch.gather(key_history, dim=2, index=gather_index)
            candidate_scores = (candidate_keys.float() * query[:, :, None, :].float()).sum(dim=-1) * float(args.scaling)
            candidate_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)

    with timer.time_stage("final_topk"):
        if candidate_scores_are_history:
            selected_scores, selected_history_indices = torch.topk(candidate_scores, k=keep_count, dim=-1, largest=True)
            selected_valid = torch.isfinite(selected_scores)
        else:
            _, selected_candidate_positions = torch.topk(candidate_scores, k=min(keep_count, candidate_scores.shape[-1]), dim=-1, largest=True)
            selected_history_indices = torch.gather(candidate_indices, dim=-1, index=selected_candidate_positions)
            selected_valid = torch.gather(candidate_valid, dim=-1, index=selected_candidate_positions)

    with timer.time_stage("final_mask_and_indices"):
        final_indices, final_valid = final_indices_from_selected(
            selected_history_indices,
            selected_valid,
            history_count,
            key_count,
            int(args.protect_sink_tokens),
            int(args.protect_recent_tokens),
            bool(args.always_keep_self),
        )

    with timer.time_stage("final_sparse_attention"):
        if kernels is not None and args.use_cuda_final_attention:
            output = kernels.final_attention(
                query.contiguous(),
                key,
                value,
                final_indices.contiguous(),
                final_valid.contiguous(),
                float(args.scaling),
            )
            return output[:, 0, :, :]
        return sparse_final_attention_torch(query, key, value, final_indices, final_valid, float(args.scaling))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_history(history_count: int, args: argparse.Namespace, kernels: Any | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed + history_count)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + history_count)

    batch_count = int(args.batch_count)
    head_count = int(args.head_count)
    head_dim = int(args.head_dim)
    key_count = history_count + 1
    scaling = float(args.scaling)

    query = torch.randn((batch_count, head_count, head_dim), device=device, dtype=dtype)
    key = torch.randn((batch_count, head_count, key_count, head_dim), device=device, dtype=dtype)
    value = torch.randn((batch_count, head_count, key_count, head_dim), device=device, dtype=dtype)
    previous_candidate = make_random_mask(
        batch_count,
        head_count,
        history_count,
        float(args.qabs_candidate_fraction),
        device,
    )
    previous_final = make_random_mask(
        batch_count,
        head_count,
        history_count,
        float(args.top_fraction),
        device,
    )
    key_dim_major = None
    if args.partial_impl == "cuda_dim_major":
        key_dim_major = key[:, :, :history_count, :].transpose(-1, -2).contiguous()

    baseline_fn = lambda: full_attention_eager(query, key, value, scaling)
    qabs_fn = lambda: qabs_attention_once(
        query,
        key,
        value,
        previous_candidate,
        previous_final,
        key_dim_major,
        args,
        kernels,
        StageTimer(device, enabled=False),
    )

    for _ in range(args.warmup_iterations):
        baseline_fn()
        qabs_fn()
    synchronize(device)

    baseline_seconds = timed_average(baseline_fn, device, int(args.iterations))
    qabs_seconds = timed_average(qabs_fn, device, int(args.iterations))

    profile_timer = StageTimer(device, enabled=True)
    for _ in range(args.profile_iterations):
        qabs_attention_once(
            query,
            key,
            value,
            previous_candidate,
            previous_final,
            key_dim_major,
            args,
            kernels,
            profile_timer,
        )
    synchronize(device)

    stage_rows: list[dict[str, Any]] = []
    profiled_total = sum(profile_timer.seconds.values())
    for stage in [
        "qdim_topk",
        "partial_scores",
        "candidate_select",
        "candidate_union",
        "candidate_full_scores",
        "final_topk",
        "final_mask_and_indices",
        "final_sparse_attention",
    ]:
        seconds = profile_timer.seconds.get(stage, 0.0)
        calls = profile_timer.calls.get(stage, 0)
        stage_rows.append(
            {
                "mode": args.mode,
                "history_tokens": history_count,
                "stage": stage,
                "seconds": seconds,
                "calls": calls,
                "ms_per_call": 1000.0 * seconds / calls if calls else 0.0,
                "fraction_of_profiled_qabs": seconds / profiled_total if profiled_total else 0.0,
            }
        )

    overhead_stages = [
        "qdim_topk",
        "partial_scores",
        "candidate_select",
        "candidate_union",
        "candidate_full_scores",
        "final_topk",
        "final_mask_and_indices",
    ]
    profiled_overhead = sum(profile_timer.seconds.get(stage, 0.0) for stage in overhead_stages)
    profiled_final_attention = profile_timer.seconds.get("final_sparse_attention", 0.0)
    summary = {
        "mode": args.mode,
        "history_tokens": history_count,
        "batch_count": batch_count,
        "head_count": head_count,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "partial_impl": args.partial_impl,
        "use_cuda_full_scores": bool(args.use_cuda_full_scores and kernels is not None),
        "use_cuda_final_attention": bool(args.use_cuda_final_attention and kernels is not None),
        "iterations": int(args.iterations),
        "profile_iterations": int(args.profile_iterations),
        "baseline_attention_ms": baseline_seconds * 1000.0,
        "qabs_attention_path_ms": qabs_seconds * 1000.0,
        "qabs_vs_baseline_attention": qabs_seconds / baseline_seconds if baseline_seconds else 0.0,
        "projected_baseline_all_layers_ms": baseline_seconds * 1000.0 * int(args.layer_count),
        "projected_qabs_all_layers_ms": qabs_seconds * 1000.0 * int(args.layer_count),
        "profiled_qabs_stage_sum_ms": 1000.0 * profiled_total / max(1, int(args.profile_iterations)),
        "profiled_new_overhead_ms": 1000.0 * profiled_overhead / max(1, int(args.profile_iterations)),
        "profiled_final_sparse_attention_ms": 1000.0 * profiled_final_attention / max(1, int(args.profile_iterations)),
    }
    return summary, stage_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark decode attention-only QABS overhead without MLP/lm-head/model forward time."
    )
    parser.add_argument("--output_dir", default="outputs/qabs_attention_only_benchmark")
    parser.add_argument("--histories", default="1024,4096,8192,16384,32768")
    parser.add_argument("--mode", choices=["qabs8cand3reusefinal", "qabs8cand3reuse"], default="qabs8cand3reusefinal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch_count", type=int, default=1)
    parser.add_argument("--head_count", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--layer_count", type=int, default=28)
    parser.add_argument("--qabs_dim_count", type=int, default=8)
    parser.add_argument("--qabs_candidate_fraction", type=float, default=0.03)
    parser.add_argument("--top_fraction", type=float, default=0.02)
    parser.add_argument("--protect_sink_tokens", type=int, default=10)
    parser.add_argument("--protect_recent_tokens", type=int, default=10)
    parser.add_argument("--always_keep_self", type=str2bool, default=True)
    parser.add_argument("--scaling", type=float, default=1.0 / math.sqrt(128.0))
    parser.add_argument("--partial_impl", choices=["torch", "cuda", "cuda_dim_major"], default="cuda_dim_major")
    parser.add_argument("--use_cuda_kernels", type=str2bool, default=True)
    parser.add_argument("--use_cuda_full_scores", type=str2bool, default=True)
    parser.add_argument("--use_cuda_final_attention", type=str2bool, default=True)
    parser.add_argument("--warmup_iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile_iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if args.head_dim <= 0 or args.head_count <= 0 or args.batch_count <= 0:
        raise ValueError("batch_count, head_count, and head_dim must be positive")
    if args.scaling == 1.0 / math.sqrt(128.0) and args.head_dim != 128:
        args.scaling = 1.0 / math.sqrt(float(args.head_dim))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = load_qabs_kernels(bool(args.use_cuda_kernels))
    if args.partial_impl.startswith("cuda") and kernels is None:
        args.partial_impl = "torch"

    summaries: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for history_count in parse_int_list(args.histories):
        print(f"benchmark history={history_count} mode={args.mode}", flush=True)
        summary, stage_rows = run_history(history_count, args, kernels)
        summaries.append(summary)
        stages.extend(stage_rows)
        print(json.dumps(summary, indent=2), flush=True)

    write_csv(output_dir / "attention_only_summary.csv", summaries)
    write_csv(output_dir / "attention_only_stage_profile.csv", stages)
    with (output_dir / "attention_only_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "summary": summaries, "stages": stages}, handle, indent=2)
    print(json.dumps({"output_dir": str(output_dir), "summary_csv": str(output_dir / "attention_only_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
