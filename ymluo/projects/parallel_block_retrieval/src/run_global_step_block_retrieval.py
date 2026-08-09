from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from profile_real_qk import QKCapture, read_jsonl, resolve_dtype
from profile_step_state_q import step_state_text
from run_global_dynamic_svd_kv_single import capture_query_ids, distributed_retrieve
from run_real_qk_retrieval import load_index, setup_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-free step-state retrieval over the full distributed block index."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--task_types", default="multihop")
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--svd_rank", type=int, default=32)
    parser.add_argument("--candidate_blocks", type=int, default=512)
    parser.add_argument("--target_blocks", type=int, default=16)
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--block_chunk", type=int, default=256)
    parser.add_argument("--exclude_block_prefix_tokens", type=int, default=0)
    parser.add_argument(
        "--resolve_bridge_profiles",
        default="",
        help="Comma-separated profile indices used by resolve_bridge; empty uses all.",
    )
    parser.add_argument(
        "--resolve_answer_profiles",
        default="",
        help="Comma-separated profile indices used by resolve_answer_from_bridge; empty uses all.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def parse_profile_indices(text: str, profile_count: int) -> list[int]:
    if not text.strip():
        return list(range(profile_count))
    indices = list(dict.fromkeys(int(item.strip()) for item in text.split(",") if item.strip()))
    if not indices or any(item < 0 or item >= profile_count for item in indices):
        raise ValueError(f"profile indices must be within [0, {profile_count})")
    return indices


def select_profiles(tensor: torch.Tensor, indices: list[int], axis: int) -> torch.Tensor:
    normalized_axis = axis if axis >= 0 else tensor.ndim + axis
    contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
    if contiguous:
        return tensor.narrow(normalized_axis, indices[0], len(indices))
    index = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(normalized_axis, index)


def rank_or_zero(values: list[int], target: int) -> int:
    return values.index(target) + 1 if target in values else 0


def recall_at(rank: int, budget: int) -> bool:
    return 0 < rank <= budget


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["split"], row["step_type"]) for row in rows})
    output = []
    for split, step_type in keys:
        group = [
            row for row in rows if row["split"] == split and row["step_type"] == step_type
        ]
        summary = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
            "mean_q_capture_seconds": statistics.fmean(
                row["q_capture_seconds"] for row in group
            ),
            "mean_retrieval_seconds": statistics.fmean(
                row["retrieval_seconds"] for row in group
            ),
        }
        for stage in ("coarse", "exact"):
            for budget in (1, 3, 16, 512):
                summary[f"{stage}_recall_at_{budget}"] = statistics.fmean(
                    recall_at(int(row[f"{stage}_rank"]), budget) for row in group
                )
        output.append(summary)
    return output


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed()
    if world_size <= 1:
        raise ValueError("global step retrieval requires at least two distributed ranks")
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    if profile_summary["profile_space"] != "pre_rope_record_qk":
        raise ValueError("expected a pre-RoPE record K index")
    if args.svd_rank > int(profile_summary["svd_rank"]):
        raise ValueError("requested SVD rank exceeds stored index")
    raw_keys, svd_keys, local_block_ids, _ = load_index(
        profile_dir, profile_summary, rank, world_size, device
    )
    basis_payload = torch.load(
        profile_dir / "basis.pt", map_location="cpu", weights_only=False
    )
    basis = basis_payload["basis"].to(device=device, dtype=torch.float16)
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    profile_count = len(pair_specs)
    profile_routes = {
        "resolve_bridge": parse_profile_indices(
            args.resolve_bridge_profiles, profile_count
        ),
        "resolve_answer_from_bridge": parse_profile_indices(
            args.resolve_answer_profiles, profile_count
        ),
    }

    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    allowed_tasks = {item.strip() for item in args.task_types.split(",") if item.strip()}
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    steps = [
        row
        for row in read_jsonl(Path(args.step_queries_path))
        if str(row["split"]) in allowed_splits
        and str(row["task_type"]) in allowed_tasks
        and int(row["query_id"]) not in excluded_ids
    ]
    steps.sort(key=lambda row: (int(row["query_id"]), int(row["step_index"])))
    if args.max_steps > 0:
        steps = steps[: args.max_steps]

    model = None
    tokenizer = None
    capture = None
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=resolve_dtype(args.dtype),
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))

    # Fields consumed by distributed_retrieve but not exposed by this batch protocol.
    args.anchor_initial_query = False
    args.dynamic_query_tokens = args.query_tokens
    args.coarse_reserve_blocks = 0
    args.hop1_block = -1

    rows = []
    for step_offset, step in enumerate(steps):
        q_capture_seconds = 0.0
        step_query = None
        if rank == 0:
            state = step_state_text(step)
            state_ids = tokenizer(state, add_special_tokens=False)["input_ids"]
            started = time.perf_counter()
            step_query = capture_query_ids(
                model,
                capture,
                pair_specs,
                state_ids,
                args.query_tokens,
                device,
            )
            torch.cuda.synchronize(device)
            q_capture_seconds = time.perf_counter() - started
        target_block = int(step["target_block_ids"][0])
        route = profile_routes.get(str(step["step_type"]), list(range(profile_count)))
        selected, event = distributed_retrieve(
            query=select_profiles(step_query, route, axis=1) if rank == 0 else None,
            basis=select_profiles(basis, route, axis=0),
            raw_keys=select_profiles(raw_keys, route, axis=2),
            svd_keys=select_profiles(svd_keys, route, axis=2),
            local_block_ids=local_block_ids,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
            gold_block_id=target_block,
        )
        if rank == 0:
            coarse = [int(item) for item in event["coarse_candidate_ids"]]
            exact = [int(item) for item in event["exact_ranked_candidate_ids"]]
            row = {
                "query_id": int(step["query_id"]),
                "step_index": int(step["step_index"]),
                "split": str(step["split"]),
                "step_type": str(step["step_type"]),
                "target_block_id": target_block,
                "selection_uses_gold": False,
                "profile_indices": route,
                "q_capture_seconds": q_capture_seconds,
                "retrieval_seconds": float(event["retrieval_seconds"]),
                "coarse_rank": rank_or_zero(coarse, target_block),
                "exact_rank": rank_or_zero(exact, target_block),
                "coarse_candidates": coarse,
                "exact_candidates": exact,
                "coarse_top16": coarse[:16],
                "exact_top16": exact[:16],
                "selected_top16": [int(item) for item in selected],
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "step": step_offset + 1,
                        "steps": len(steps),
                        "query_id": row["query_id"],
                        "step_index": row["step_index"],
                        "coarse_rank": row["coarse_rank"],
                        "exact_rank": row["exact_rank"],
                        "retrieval_seconds": row["retrieval_seconds"],
                    }
                ),
                flush=True,
            )
            with (output_dir / "rows.partial.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        dist.barrier()

    if rank == 0:
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        payload = {
            "source": "leakage-free typed step Q over full distributed 10M K index",
            "contains_synthetic_vectors": False,
            "selection_uses_gold": False,
            "num_blocks": int(profile_summary["num_blocks"]),
            "num_tokens": int(profile_summary["num_tokens"]),
            "world_size": world_size,
            "steps": len(rows),
            "svd_rank": args.svd_rank,
            "candidate_blocks": args.candidate_blocks,
            "target_blocks": args.target_blocks,
            "profile_routes": profile_routes,
            "pair_specs": pair_specs,
            "summaries": summarize(rows),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        capture.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
