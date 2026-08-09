from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

import qabs_cuda_kernels as kernels
from run_head_top2_targeted_ppl_20260714 import (
    qabs_sampled_head_adaptive_attention,
)


def measure_ms(callback: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        callback()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_tokens", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260720)
    history = args.history_tokens
    query = torch.randn((1, 32, 1, 128), device="cuda", dtype=torch.float16)
    key = torch.randn((1, 8, history + 1, 128), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    state: dict[str, object] = {"rope_theta": 5_000_000.0}
    _, packed = qabs_sampled_head_adaptive_attention(
        query,
        key,
        value,
        attention_mask=None,
        scaling=128**-0.5,
        mass_threshold=0.75,
        budget_fractions=(0.02,),
        sample_fraction=0.0025,
        qabs_dim_count=8,
        candidate_fraction=0.08,
        use_cuda_kernels=True,
        skip_candidate_rerank=False,
        score_mode="pca_int4_chunked_logscale16",
        projection_dim=64,
        pca_state=state,
    )
    indices = packed.squeeze(2).contiguous()
    scores = kernels.candidate_compact_scores(
        query[:, :, 0].contiguous(), key, indices, 128**-0.5
    ).float()
    counts = torch.full(
        indices.shape[:2], indices.shape[-1], dtype=torch.long, device="cuda"
    )
    sorted_indices, order = torch.sort(indices, dim=-1)
    sorted_scores = torch.gather(scores, -1, order)

    def raw() -> torch.Tensor:
        return kernels.final_attention_from_scores_ragged(
            value, indices, scores, counts
        )

    def presorted() -> torch.Tensor:
        return kernels.final_attention_from_scores_ragged(
            value, sorted_indices, sorted_scores, counts
        )

    def online_sorted() -> torch.Tensor:
        current_indices, current_order = torch.sort(indices, dim=-1)
        current_scores = torch.gather(scores, -1, current_order)
        return kernels.final_attention_from_scores_ragged(
            value, current_indices, current_scores, counts
        )

    reference = raw()
    sorted_output = presorted()
    split_outputs = {
        split: kernels.final_attention_from_scores_split(
            value, indices, scores, counts, split
        )
        for split in (2, 4, 8, 16)
    }
    report = {
        "hardware": torch.cuda.get_device_name(),
        "history_tokens": history,
        "selected_tokens": indices.shape[-1],
        "raw_ms": measure_ms(raw, args.warmup, args.repeats),
        "presorted_ms": measure_ms(presorted, args.warmup, args.repeats),
        "online_sort_plus_attention_ms": measure_ms(
            online_sorted, args.warmup, args.repeats
        ),
        "split_attention_ms": {
            str(split): measure_ms(
                lambda split=split: kernels.final_attention_from_scores_split(
                    value, indices, scores, counts, split
                ),
                args.warmup,
                args.repeats,
            )
            for split in (2, 4, 8, 16)
        },
        "split_output_max_abs_error": {
            str(split): float(
                (reference.float() - output.float()).abs().max()
            )
            for split, output in split_outputs.items()
        },
        "sorted_output_max_abs_error": float(
            (reference.float() - sorted_output.float()).abs().max()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
