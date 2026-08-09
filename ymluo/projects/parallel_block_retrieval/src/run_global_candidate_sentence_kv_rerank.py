from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_branch_transition_verifier import contains_phrase
from evaluate_global_step_hybrid_candidates import step_anchor_text
from evaluate_global_step_hybrid_candidates import normalized_tokens
from profile_real_qk import QKCapture, read_jsonl, resolve_dtype
from profile_step_state_q import step_state_text
from run_global_dynamic_svd_kv_single import broadcast_query, capture_query_ids
from run_global_step_block_retrieval import parse_profile_indices, select_profiles
from run_real_qk_retrieval import load_index, setup_distributed
from run_step_state_kv_span_retrieval import find_text_subsequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed raw-K sentence reranking over global candidate blocks."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--profile_dir", required=True)
    parser.add_argument("--sidecar_dir", required=True)
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--candidate_rows_path", required=True)
    parser.add_argument("--candidate_field", default="hybrid_anchor_qk_candidates")
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=0,
        help="Optional prefix limit after the upstream candidate ranking.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--task_types", default="multihop")
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument("--query_tokens", type=int, default=16)
    parser.add_argument("--score_chunk", type=int, default=32)
    parser.add_argument("--branch_blocks", type=int, default=3)
    parser.add_argument("--spans_per_block", type=int, default=3)
    parser.add_argument(
        "--branch_block_order",
        choices=["candidate", "kv"],
        default="candidate",
        help="Choose branch blocks from the incoming candidate order or KV sentence ranking.",
    )
    parser.add_argument(
        "--span_anchor_gate",
        action="store_true",
        help="Rank only sentence spans containing the current state anchor when available.",
    )
    parser.add_argument(
        "--span_anchor_alias_fallback",
        action="store_true",
        help="Allow surname-only anchor spans for answer steps while preserving full-name candidates.",
    )
    parser.add_argument("--span_anchor_neighbor_radius", type=int, default=0)
    parser.add_argument(
        "--index_backend",
        choices=["gpu", "mmap_cpu", "sparse_cpu"],
        default="gpu",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resolve_bridge_profiles", default="0,1")
    parser.add_argument("--resolve_answer_profiles", default="2,3")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


class MMapRawKIndex:
    def __init__(self, profile_dir: Path, profile_summary: dict[str, Any]) -> None:
        self.shards = sorted(
            profile_summary["shards"], key=lambda item: int(item["block_start"])
        )
        self.arrays = [
            np.load(
                profile_dir / Path(str(shard["raw_k_path"])).name,
                mmap_mode="r",
            )
            for shard in self.shards
        ]

    def block(self, block_id: int) -> np.ndarray:
        for shard, array in zip(self.shards, self.arrays, strict=True):
            start = int(shard["block_start"])
            end = int(shard["block_end"])
            if start <= block_id < end:
                return np.asarray(array[block_id - start])
        raise IndexError(f"block {block_id} is outside all K shards")


class SparseRawKIndex:
    def __init__(self, profile_dir: Path, profile_summary: dict[str, Any]) -> None:
        block_ids = np.load(profile_dir / "block_ids.npy", mmap_mode="r")
        self.offsets = {
            int(block_id): offset for offset, block_id in enumerate(block_ids.tolist())
        }
        self.array = np.load(profile_dir / "raw_k.npy", mmap_mode="r")

    def block(self, block_id: int) -> np.ndarray:
        if block_id not in self.offsets:
            raise KeyError(f"block {block_id} was not included in the sparse K profile")
        return np.asarray(self.array[self.offsets[block_id]])


class SentenceSpanIndex:
    def __init__(self, sidecar_dir: Path) -> None:
        summary = json.loads((sidecar_dir / "summary.json").read_text(encoding="utf-8"))
        self.offsets = np.load(sidecar_dir / "block_span_offsets.npy", mmap_mode="r")
        self.spans = np.load(sidecar_dir / "sentence_spans.npy", mmap_mode="r")
        self.block_offsets = None
        if bool(summary.get("sparse", False)):
            block_ids = np.load(sidecar_dir / "block_ids.npy", mmap_mode="r")
            self.block_offsets = {
                int(block_id): offset for offset, block_id in enumerate(block_ids.tolist())
            }

    def block(self, block_id: int) -> np.ndarray:
        offset = (
            self.block_offsets[block_id]
            if self.block_offsets is not None
            else block_id
        )
        start = int(self.offsets[offset])
        end = int(self.offsets[offset + 1])
        return np.asarray(self.spans[start:end])


def overlap_fraction(span: Sequence[int], target: Sequence[int]) -> float:
    start, end = int(span[0]), int(span[1])
    target_start, target_end = int(target[0]), int(target[1])
    return max(0, min(end, target_end) - max(start, target_start)) / max(
        1, target_end - target_start
    )


def rank_sentence_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    ranked = sorted(
        records,
        key=lambda item: (
            -float(item["score"]),
            int(item["block_id"]),
            int(item["start"]),
        ),
    )
    block_order = []
    seen = set()
    for item in ranked:
        block_id = int(item["block_id"])
        if block_id not in seen:
            seen.add(block_id)
            block_order.append(block_id)
    return ranked, block_order


def rank_or_zero(values: Sequence[int], target: int) -> int:
    try:
        return values.index(target) + 1
    except ValueError:
        return 0


def score_local_spans(
    *,
    raw_keys: torch.Tensor,
    local_offsets: dict[int, int],
    candidate_ids: Sequence[int],
    query: torch.Tensor,
    route: list[int],
    span_index: SentenceSpanIndex,
    chunk_size: int,
) -> list[dict[str, Any]]:
    local_ids = [int(item) for item in candidate_ids if int(item) in local_offsets]
    if not local_ids:
        return []
    valid_query = select_profiles(query, route, axis=1)
    valid_query = valid_query / valid_query.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    routed_keys = select_profiles(raw_keys, route, axis=2)
    records = []
    for start in range(0, len(local_ids), chunk_size):
        block_ids = local_ids[start : start + chunk_size]
        offsets = torch.tensor(
            [local_offsets[item] for item in block_ids],
            dtype=torch.long,
            device=raw_keys.device,
        )
        keys = routed_keys.index_select(0, offsets).float()
        keys /= keys.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        similarities = torch.einsum(
            "qpd,btpd->bqpt", valid_query.float(), keys
        ).to(dtype=torch.float32)
        similarities_cpu = similarities.cpu().numpy()
        for local_index, block_id in enumerate(block_ids):
            for sentence_start, sentence_end in span_index.block(block_id):
                sentence_start = int(sentence_start)
                sentence_end = int(sentence_end)
                per_profile = similarities_cpu[
                    local_index, :, :, sentence_start:sentence_end
                ].max(axis=-1).mean(axis=0)
                records.append(
                    {
                        "block_id": block_id,
                        "start": sentence_start,
                        "end": sentence_end,
                        "score": float(per_profile.mean()),
                        "profile_scores": [float(item) for item in per_profile],
                    }
                )
    return records


def score_mmap_spans(
    *,
    index: MMapRawKIndex,
    candidate_ids: Sequence[int],
    query: torch.Tensor,
    route: list[int],
    span_index: SentenceSpanIndex,
) -> list[dict[str, Any]]:
    query_array = query.detach().float().cpu().numpy()[:, route]
    query_array /= np.maximum(
        np.linalg.norm(query_array, axis=-1, keepdims=True), 1.0e-6
    )
    records = []
    for block_id_value in candidate_ids:
        block_id = int(block_id_value)
        keys = index.block(block_id).astype(np.float32, copy=False)[:, route]
        keys = keys / np.maximum(np.linalg.norm(keys, axis=-1, keepdims=True), 1.0e-6)
        similarities = np.einsum("qpd,tpd->qpt", query_array, keys, optimize=True)
        for sentence_start, sentence_end in span_index.block(block_id):
            sentence_start = int(sentence_start)
            sentence_end = int(sentence_end)
            per_profile = similarities[:, :, sentence_start:sentence_end].max(
                axis=-1
            ).mean(axis=0)
            records.append(
                {
                    "block_id": block_id,
                    "start": sentence_start,
                    "end": sentence_end,
                    "score": float(per_profile.mean()),
                    "profile_scores": [float(item) for item in per_profile],
                }
            )
    return records


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["split"], row["step_type"]) for row in rows})
    for split, step_type in keys:
        group = [
            row for row in rows if row["split"] == split and row["step_type"] == step_type
        ]
        item: dict[str, Any] = {
            "split": split,
            "step_type": step_type,
            "steps": len(group),
            "candidate_recall": statistics.fmean(row["candidate_hit"] for row in group),
            "mean_candidate_blocks": statistics.fmean(
                row["candidate_blocks"] for row in group
            ),
            "mean_q_capture_seconds": statistics.fmean(
                row["q_capture_seconds"] for row in group
            ),
            "mean_sentence_score_seconds": statistics.fmean(
                row["sentence_score_seconds"] for row in group
            ),
            "mean_spans_before_anchor_gate": statistics.fmean(
                row["spans_before_anchor_gate"] for row in group
            ),
            "mean_spans_after_anchor_gate": statistics.fmean(
                row["spans_after_anchor_gate"] for row in group
            ),
            "anchor_gate_fallback_rate": statistics.fmean(
                row["anchor_gate_fallback"] for row in group
            ),
        }
        for budget in (1, 3, 16):
            item[f"candidate_target_recall_at_{budget}"] = statistics.fmean(
                0 < row["candidate_target_rank"] <= budget for row in group
            )
            item[f"target_block_recall_at_{budget}"] = statistics.fmean(
                0 < row["target_block_rank"] <= budget for row in group
            )
            item[f"target_span_recall_at_{budget}"] = statistics.fmean(
                0 < row["target_span_rank"] <= budget for row in group
            )
        for budget in (1, 2, 3):
            item[f"target_span_within_block_recall_at_{budget}"] = statistics.fmean(
                0 < row["target_span_within_block_rank"] <= budget for row in group
            )
        for budget in (3, 6, 9):
            item[f"branch_target_span_recall_at_{budget}"] = statistics.fmean(
                0 < row["branch_target_span_rank"] <= budget for row in group
            )
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        rank, world_size, device = 0, 1, torch.device("cpu")
    else:
        rank, world_size, _local_rank, device = setup_distributed()
    if args.index_backend in {"mmap_cpu", "sparse_cpu"} and world_size != 1:
        raise ValueError("CPU index backends currently require world size 1")
    corpus_dir = Path(args.corpus_dir)
    profile_dir = Path(args.profile_dir)
    sidecar_dir = Path(args.sidecar_dir)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    profile_summary = json.loads((profile_dir / "summary.json").read_text(encoding="utf-8"))
    raw_keys = None
    local_offsets: dict[int, int] = {}
    mmap_index = None
    if args.index_backend == "gpu":
        raw_keys, _svd_keys, local_block_ids, _ = load_index(
            profile_dir, profile_summary, rank, world_size, device
        )
        local_offsets = {
            int(block_id): offset for offset, block_id in enumerate(local_block_ids.tolist())
        }
    elif args.index_backend == "mmap_cpu":
        mmap_index = MMapRawKIndex(profile_dir, profile_summary)
    else:
        mmap_index = SparseRawKIndex(profile_dir, profile_summary)
    pair_specs = [dict(item) for item in profile_summary["pair_specs"]]
    profile_count = len(pair_specs)
    routes = {
        "resolve_bridge": parse_profile_indices(
            args.resolve_bridge_profiles, profile_count
        ),
        "resolve_answer_from_bridge": parse_profile_indices(
            args.resolve_answer_profiles, profile_count
        ),
    }
    span_index = SentenceSpanIndex(sidecar_dir)
    blocks = np.load(corpus_dir / "blocks.npy", mmap_mode="r")

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
    candidate_rows = {
        (int(row["query_id"]), int(row["step_index"])): row
        for row in read_jsonl(Path(args.candidate_rows_path))
    }

    model = None
    tokenizer = None
    capture = None
    if rank == 0:
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
        capture = QKCapture(model, sorted({int(item["layer"]) for item in pair_specs}))

    rows = []
    for step_offset, step in enumerate(steps):
        key = (int(step["query_id"]), int(step["step_index"]))
        if key not in candidate_rows:
            raise ValueError(f"missing candidate row for step {key}")
        candidate_ids = [
            int(item) for item in candidate_rows[key][args.candidate_field]
        ]
        if args.candidate_limit > 0:
            candidate_ids = candidate_ids[: args.candidate_limit]
        route = routes[str(step["step_type"])]
        step_query = None
        q_capture_seconds = 0.0
        if rank == 0:
            state_ids = tokenizer(
                step_state_text(step), add_special_tokens=False
            )["input_ids"]
            capture_started = time.perf_counter()
            step_query = capture_query_ids(
                model, capture, pair_specs, state_ids, args.query_tokens, device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            q_capture_seconds = time.perf_counter() - capture_started
        if world_size > 1:
            broadcasted, query_mask = broadcast_query(
                step_query,
                max_tokens=args.query_tokens,
                profile_count=profile_count,
                head_dim=int(profile_summary["head_dim"]),
                rank=rank,
                device=device,
            )
            query = broadcasted[0, query_mask[0]]
            dist.barrier()
        else:
            query = step_query.to(device=device, dtype=torch.float16)
        score_started = time.perf_counter()
        if args.index_backend == "gpu":
            local_records = score_local_spans(
                raw_keys=raw_keys,
                local_offsets=local_offsets,
                candidate_ids=candidate_ids,
                query=query,
                route=route,
                span_index=span_index,
                chunk_size=args.score_chunk,
            )
        else:
            local_records = score_mmap_spans(
                index=mmap_index,
                candidate_ids=candidate_ids,
                query=query,
                route=route,
                span_index=span_index,
            )
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(gathered, local_records)
            dist.barrier()
        else:
            gathered[0] = local_records
        sentence_score_seconds = time.perf_counter() - score_started
        if rank != 0:
            continue

        all_records = [item for group in gathered if group for item in group]
        spans_before_anchor_gate = len(all_records)
        anchor_gate_fallback = False
        if args.span_anchor_gate:
            anchor = step_anchor_text(step)
            aliases = [anchor]
            anchor_terms = normalized_tokens(anchor)
            if (
                args.span_anchor_alias_fallback
                and str(step["step_type"]) == "resolve_answer_from_bridge"
                and len(anchor_terms) >= 2
                and len(anchor_terms[-1]) >= 4
            ):
                aliases.append(anchor_terms[-1])
            direct_matches = []
            for item in all_records:
                token_ids = blocks[int(item["block_id"])][
                    int(item["start"]) : int(item["end"])
                ].tolist()
                sentence_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                if any(contains_phrase(sentence_text, alias) for alias in aliases):
                    direct_matches.append(item)
            gated_records = list(direct_matches)
            if direct_matches and args.span_anchor_neighbor_radius > 0:
                by_block: dict[int, list[dict[str, Any]]] = {}
                for item in all_records:
                    by_block.setdefault(int(item["block_id"]), []).append(item)
                expanded = []
                for block_id, block_records in by_block.items():
                    block_records.sort(key=lambda item: int(item["start"]))
                    matched_positions = {
                        index
                        for index, item in enumerate(block_records)
                        if item in direct_matches
                    }
                    selected_positions = {
                        neighbor
                        for index in matched_positions
                        for neighbor in range(
                            max(0, index - args.span_anchor_neighbor_radius),
                            min(
                                len(block_records),
                                index + args.span_anchor_neighbor_radius + 1,
                            ),
                        )
                    }
                    expanded.extend(block_records[index] for index in selected_positions)
                gated_records = expanded
            if gated_records:
                all_records = gated_records
            else:
                anchor_gate_fallback = True
        spans_after_anchor_gate = len(all_records)
        ranked_sentences, block_order = rank_sentence_records(all_records)
        target_block = int(step["target_block_ids"][0])
        target_span = find_text_subsequence(
            blocks[target_block].tolist(), str(step["target_fact"]), tokenizer
        )
        target_sentence_ranks = [
            index + 1
            for index, item in enumerate(ranked_sentences)
            if int(item["block_id"]) == target_block
            and overlap_fraction((item["start"], item["end"]), target_span) >= 0.8
        ]
        target_block_sentences = [
            item for item in ranked_sentences if int(item["block_id"]) == target_block
        ]
        target_within_ranks = [
            index + 1
            for index, item in enumerate(target_block_sentences)
            if overlap_fraction((item["start"], item["end"]), target_span) >= 0.8
        ]
        branch_candidates = []
        branch_blocks = (
            block_order[: args.branch_blocks]
            if args.branch_block_order == "kv"
            else candidate_ids[: args.branch_blocks]
        )
        sentences_by_block = {
            block_id: [
                item for item in ranked_sentences if int(item["block_id"]) == block_id
            ][: args.spans_per_block]
            for block_id in branch_blocks
        }
        for span_offset in range(args.spans_per_block):
            for block_rank, block_id in enumerate(branch_blocks, start=1):
                if span_offset >= len(sentences_by_block[block_id]):
                    continue
                item = sentences_by_block[block_id][span_offset]
                span_rank = span_offset + 1
                branch_candidates.append(
                    {
                        "rank": len(branch_candidates) + 1,
                        "block_rank": block_rank,
                        "span_rank": span_rank,
                        "block_id": block_id,
                        "start": int(item["start"]),
                        "end": int(item["end"]),
                        "score": float(item["score"]),
                        "target_overlap": (
                            overlap_fraction((item["start"], item["end"]), target_span)
                            if block_id == target_block
                            else 0.0
                        ),
                    }
                )
        branch_target_ranks = [
            index + 1
            for index, item in enumerate(branch_candidates)
            if float(item["target_overlap"]) >= 0.8
        ]
        row = {
            "query_id": key[0],
            "step_index": key[1],
            "split": str(step["split"]),
            "step_type": str(step["step_type"]),
            "profile_indices": route,
            "selection_uses_gold": False,
            "candidate_blocks": len(candidate_ids),
            "candidate_hit": target_block in candidate_ids,
            "candidate_target_rank": rank_or_zero(candidate_ids, target_block),
            "target_block_id": target_block,
            "target_block_rank": rank_or_zero(block_order, target_block),
            "target_span_rank": min(target_sentence_ranks) if target_sentence_ranks else 0,
            "target_span_within_block_rank": (
                min(target_within_ranks) if target_within_ranks else 0
            ),
            "branch_target_span_rank": (
                min(branch_target_ranks) if branch_target_ranks else 0
            ),
            "q_capture_seconds": q_capture_seconds,
            "sentence_score_seconds": sentence_score_seconds,
            "span_anchor_gate": args.span_anchor_gate,
            "span_anchor_alias_fallback": args.span_anchor_alias_fallback,
            "span_anchor_neighbor_radius": args.span_anchor_neighbor_radius,
            "spans_before_anchor_gate": spans_before_anchor_gate,
            "spans_after_anchor_gate": spans_after_anchor_gate,
            "anchor_gate_fallback": anchor_gate_fallback,
            "top_sentences": ranked_sentences[:16],
            "branch_candidates": branch_candidates,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "step": step_offset + 1,
                    "steps": len(steps),
                    "query_id": key[0],
                    "step_index": key[1],
                    "candidate_hit": row["candidate_hit"],
                    "target_block_rank": row["target_block_rank"],
                    "target_span_rank": row["target_span_rank"],
                    "sentence_score_seconds": sentence_score_seconds,
                }
            ),
            flush=True,
        )

    if rank == 0:
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "source": "distributed operator-routed raw128 K sentence reranking",
            "contains_synthetic_vectors": False,
            "selection_uses_gold": False,
            "world_size": world_size,
            "index_backend": args.index_backend,
            "device": str(device),
            "steps": len(rows),
            "candidate_field": args.candidate_field,
            "candidate_limit": args.candidate_limit,
            "branch_block_order": args.branch_block_order,
            "span_anchor_gate": args.span_anchor_gate,
            "span_anchor_alias_fallback": args.span_anchor_alias_fallback,
            "span_anchor_neighbor_radius": args.span_anchor_neighbor_radius,
            "branch_blocks": args.branch_blocks,
            "spans_per_block": args.spans_per_block,
            "profile_routes": routes,
            "pair_specs": pair_specs,
            "summaries": summarize(rows),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        capture.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
