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
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, captured_qk, read_jsonl, resolve_dtype
from profile_step_state_q import step_state_text


METHODS = (
    "pre_last_raw",
    "pre_last_cos",
    "pre_qmean_cos",
    "pre_qmax_raw",
    "pre_qmax_cos",
    "pre_qmax_centered_cos",
    "pre_seg4_qmax_cos",
    "pre_seg4_qmax_centered_cos",
    "pre_centered_p0_cos",
    "pre_centered_p1_cos",
    "pre_centered_p2_cos",
    "pre_centered_p3_cos",
    "post_local_qmax_cos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve all real blocks by multiplying query Q with block K centroids."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument(
        "--step_types", default="resolve_bridge,resolve_answer_from_bridge"
    )
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--candidate_blocks", type=int, default=512)
    parser.add_argument("--score_batch", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def target_rank(candidates: list[int], target: int) -> int:
    return candidates.index(target) + 1 if target in candidates else 0


def top_candidates(scores: torch.Tensor, budget: int) -> list[list[int]]:
    count = min(budget, int(scores.shape[1]))
    return [
        [int(item) for item in row]
        for row in torch.topk(scores, k=count, dim=1, largest=True, sorted=True)
        .indices.cpu()
        .tolist()
    ]


def max_query_profile(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    similarities = torch.einsum("nqpd,bpd->nqbp", query, keys)
    return similarities.amax(dim=(1, 3))


def max_query_segment_profile(
    query: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    similarities = torch.einsum("nqpd,bspd->nqbsp", query, keys)
    return similarities.amax(dim=(1, 3, 4))


def score_methods(
    pre_query: torch.Tensor,
    post_query: torch.Tensor,
    pre_mean: torch.Tensor,
    pre_mean_cos: torch.Tensor,
    pre_mean_centered_cos: torch.Tensor,
    pre_segments: torch.Tensor,
    pre_segments_cos: torch.Tensor,
    pre_segments_centered_cos: torch.Tensor,
    post_mean: torch.Tensor,
    post_mean_cos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scale = math.sqrt(float(pre_query.shape[-1]))
    pre_query_cos = F.normalize(pre_query.float(), dim=-1).to(torch.float16)
    post_query_cos = F.normalize(post_query.float(), dim=-1).to(torch.float16)

    last = pre_query[:, -1:, :, :]
    last_cos = pre_query_cos[:, -1:, :, :]
    query_mean_cos = F.normalize(pre_query.float().mean(dim=1), dim=-1).to(
        torch.float16
    )
    centered_similarities = torch.einsum(
        "nqpd,bpd->nqbp", pre_query_cos, pre_mean_centered_cos
    )
    return {
        "pre_last_raw": max_query_profile(last, pre_mean) / scale,
        "pre_last_cos": max_query_profile(last_cos, pre_mean_cos),
        "pre_qmean_cos": torch.einsum(
            "npd,bpd->nbp", query_mean_cos, pre_mean_cos
        ).amax(dim=2),
        "pre_qmax_raw": max_query_profile(pre_query, pre_mean) / scale,
        "pre_qmax_cos": max_query_profile(pre_query_cos, pre_mean_cos),
        "pre_qmax_centered_cos": centered_similarities.amax(dim=(1, 3)),
        "pre_seg4_qmax_cos": max_query_segment_profile(
            pre_query_cos, pre_segments_cos
        ),
        "pre_seg4_qmax_centered_cos": max_query_segment_profile(
            pre_query_cos, pre_segments_centered_cos
        ),
        **{
            f"pre_centered_p{profile}_cos": centered_similarities[..., profile].amax(
                dim=1
            )
            for profile in range(centered_similarities.shape[-1])
        },
        "post_local_qmax_cos": max_query_profile(post_query_cos, post_mean_cos),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for split, step_type in sorted(
        {(str(row["split"]), str(row["step_type"])) for row in rows}
    ):
        group = [
            row
            for row in rows
            if str(row["split"]) == split and str(row["step_type"]) == step_type
        ]
        item: dict[str, Any] = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
        }
        for method in METHODS:
            for budget in (1, 3, 16, 512):
                item[f"{method}_recall_at_{budget}"] = statistics.fmean(
                    0 < int(row[f"{method}_rank"]) <= budget for row in group
                )
        summaries.append(item)
    return summaries


def main() -> None:
    args = parse_args()
    if min(args.query_tokens, args.candidate_blocks, args.score_batch) <= 0:
        raise ValueError("query_tokens, candidate_blocks and score_batch must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_summary = json.loads((index_dir / "summary.json").read_text(encoding="utf-8"))
    pair_specs = [dict(item) for item in index_summary["pair_specs"]]
    if int(index_summary["segments"]) != 4:
        raise ValueError("this experiment expects four segment centroids")

    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_types = {
        item.strip() for item in args.step_types.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) in allowed_splits
        and str(row["step_type"]) in allowed_types
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=(
            resolve_dtype(args.dtype) if device.type == "cuda" else torch.float32
        ),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))

    pre_queries = []
    post_queries = []
    query_started = time.perf_counter()
    for index, step in enumerate(steps, start=1):
        input_ids = tokenizer(
            step_state_text(step), add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        capture.clear()
        with torch.inference_mode():
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
        pre_q, _ = captured_qk(model, capture, pair_specs, "pre_rope_block_qk")
        post_q, _ = captured_qk(model, capture, pair_specs, "post_rope_record_qk")
        keep = min(args.query_tokens, int(pre_q.shape[1]))
        if keep != args.query_tokens:
            raise ValueError("query text is shorter than query_tokens")
        pre_queries.append(pre_q[0, -keep:].to(torch.float16).cpu())
        post_queries.append(post_q[0, -keep:].to(torch.float16).cpu())
        if index % 100 == 0 or index == len(steps):
            print(json.dumps({"captured_queries": index, "total": len(steps)}), flush=True)
    query_capture_seconds = time.perf_counter() - query_started
    capture.close()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pre_mean = torch.from_numpy(
        np.asarray(np.load(index_dir / "pre_k_mean.npy", mmap_mode="r"))
    ).to(device)
    pre_segments = torch.from_numpy(
        np.asarray(np.load(index_dir / "pre_k_segment_mean.npy", mmap_mode="r"))
    ).to(device)
    post_mean = torch.from_numpy(
        np.asarray(np.load(index_dir / "post_local_k_mean.npy", mmap_mode="r"))
    ).to(device)
    normalization_started = time.perf_counter()
    pre_mean_cos = F.normalize(pre_mean.float(), dim=-1).to(torch.float16)
    pre_mean_centered_cos = F.normalize(
        pre_mean.float() - pre_mean.float().mean(dim=0, keepdim=True), dim=-1
    ).to(torch.float16)
    pre_segments_cos = F.normalize(pre_segments.float(), dim=-1).to(torch.float16)
    pre_segments_centered_cos = F.normalize(
        pre_segments.float()
        - pre_segments.float().mean(dim=(0, 1), keepdim=True),
        dim=-1,
    ).to(torch.float16)
    post_mean_cos = F.normalize(post_mean.float(), dim=-1).to(torch.float16)
    normalization_seconds = time.perf_counter() - normalization_started

    rows = []
    method_seconds = {method: 0.0 for method in METHODS}
    score_started = time.perf_counter()
    for start in range(0, len(steps), args.score_batch):
        end = min(len(steps), start + args.score_batch)
        pre_batch = torch.stack(pre_queries[start:end]).to(device)
        post_batch = torch.stack(post_queries[start:end]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        method_started = time.perf_counter()
        scores = score_methods(
            pre_batch,
            post_batch,
            pre_mean,
            pre_mean_cos,
            pre_mean_centered_cos,
            pre_segments,
            pre_segments_cos,
            pre_segments_centered_cos,
            post_mean,
            post_mean_cos,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        all_method_seconds = time.perf_counter() - method_started
        for method in METHODS:
            method_seconds[method] += all_method_seconds / len(METHODS)
        ranked = {
            method: top_candidates(values, args.candidate_blocks)
            for method, values in scores.items()
        }
        for local_index, step in enumerate(steps[start:end]):
            target = int(step["target_block_ids"][0])
            row: dict[str, Any] = {
                "query_id": int(step["query_id"]),
                "step_index": int(step["step_index"]),
                "split": str(step["split"]),
                "step_type": str(step["step_type"]),
                "target_block_id": target,
                "selection_uses_gold": False,
            }
            for method in METHODS:
                candidates = ranked[method][local_index]
                row[f"{method}_candidates"] = candidates
                row[f"{method}_rank"] = target_rank(candidates, target)
            rows.append(row)
        print(json.dumps({"scored_queries": end, "total": len(steps)}), flush=True)
    score_seconds = time.perf_counter() - score_started

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "full-matrix real Q to block K-mean retrieval",
        "contains_synthetic_vectors": False,
        "selection_uses_gold": False,
        "index_dir": str(index_dir),
        "blocks": int(index_summary["blocks"]),
        "steps": len(steps),
        "query_tokens": args.query_tokens,
        "candidate_blocks": args.candidate_blocks,
        "methods": list(METHODS),
        "query_capture_seconds": query_capture_seconds,
        "index_normalization_seconds": normalization_seconds,
        "score_seconds": score_seconds,
        "mean_score_seconds_per_query": score_seconds / max(len(steps), 1),
        "approximate_method_seconds": method_seconds,
        "summaries": summarize(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
