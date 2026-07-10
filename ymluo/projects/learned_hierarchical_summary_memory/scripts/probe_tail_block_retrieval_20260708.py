from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen8b_paper_benchmarks import (  # noqa: E402
    Config,
    BenchCase,
    build_tail_block_index,
    content_words,
    exact_identifiers,
    load_longbench_cases,
    load_ruler_cases,
    score_blocks,
    score_blocks_tail,
    token_blocks,
)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def make_config(args: argparse.Namespace, block_tokens: int) -> Config:
    return Config(
        output_dir=str(args.output_dir),
        model_name_or_path=args.model_name_or_path,
        adapter_path="",
        longbench_data_dir=args.longbench_data_dir,
        ruler_data_dir=args.ruler_data_dir,
        longbench_tasks=parse_csv(args.longbench_tasks),
        ruler_tasks=parse_csv(args.ruler_tasks),
        ruler_context_lengths=parse_ints(args.ruler_context_lengths),
        methods=(),
        max_examples_per_task=args.max_examples_per_task,
        case_ids=parse_csv(args.case_ids),
        block_tokens=block_tokens,
        recent_tokens=args.recent_tokens,
        max_input_tokens=24000,
        summary10_words=10,
        summary100_words=100,
        summary1000_words=900,
        max_new_tokens_exact=48,
        max_new_tokens_summary=120,
        dtype="float16",
        attn_implementation="sdpa",
        device_map="auto",
        cuda_visible_devices="",
        router_path="",
        seed=args.seed,
    )


def old_context_for_case(tokenizer: Any, case: BenchCase, recent_tokens: int) -> str:
    ids = tokenizer(case.context, add_special_tokens=False)["input_ids"]
    old_cut = max(0, len(ids) - recent_tokens)
    return tokenizer.decode(ids[:old_cut], skip_special_tokens=True)


def answer_hit(blocks: list[str], answers: tuple[str, ...]) -> int:
    lowered = "\n".join(blocks).lower()
    return int(any(answer and answer.lower() in lowered for answer in answers))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["group"], row["block_tokens"], row["top_k"], row["scorer"]) for row in rows})
    for group, block_tokens, top_k, scorer in keys:
        items = [row for row in rows if (row["group"], row["block_tokens"], row["top_k"], row["scorer"]) == (group, block_tokens, top_k, scorer)]
        out.append(
            {
                "group": group,
                "block_tokens": block_tokens,
                "top_k": top_k,
                "scorer": scorer,
                "samples": len(items),
                "answer_hit_rate": statistics.mean(float(row["answer_hit"]) for row in items),
                "avg_score_seconds": statistics.mean(float(row["score_seconds"]) for row in items),
                "avg_tail_build_seconds": statistics.mean(float(row.get("tail_build_seconds", 0.0)) for row in items),
                "avg_topk_overlap_with_full": statistics.mean(float(row["topk_overlap_with_full"]) for row in items),
                "avg_blocks": statistics.mean(float(row["num_blocks"]) for row in items),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--longbench_data_dir", default="/home/fdong/ymluo/external/KVCache-Factory/data/LongBench")
    parser.add_argument("--ruler_data_dir", default="/home/fdong/ymluo/external/KVCache-Factory/data/RULER")
    parser.add_argument("--longbench_tasks", default="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count")
    parser.add_argument("--ruler_tasks", default="niah_single_1,niah_single_2,niah_multikey_1,niah_multiquery,niah_multivalue,vt,cwe,fwe")
    parser.add_argument("--ruler_context_lengths", default="4096,8192,16384")
    parser.add_argument("--max_examples_per_task", type=int, default=20)
    parser.add_argument("--case_ids", default="")
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--block_tokens_list", default="128,256,512")
    parser.add_argument("--topk_list", default="3,8,12")
    parser.add_argument("--seed", type=int, default=2026070809)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    base_config = make_config(args, block_tokens=512)
    cases = load_longbench_cases(base_config) + load_ruler_cases(base_config)
    rows: list[dict[str, Any]] = []
    for block_tokens in parse_ints(args.block_tokens_list):
        config = replace(base_config, block_tokens=block_tokens)
        for case in cases:
            old_context = old_context_for_case(tokenizer, case, args.recent_tokens)
            blocks = token_blocks(tokenizer, old_context, block_tokens)
            if not blocks:
                continue
            started = time.perf_counter()
            full_scored = score_blocks(tokenizer, old_context, case.query, config)
            full_seconds = time.perf_counter() - started
            started = time.perf_counter()
            tail_scored = score_blocks_tail(tokenizer, old_context, case.query, config)
            tail_seconds = time.perf_counter() - started
            started = time.perf_counter()
            _, postings = build_tail_block_index(blocks)
            tail_build_seconds = time.perf_counter() - started
            query_terms = set(content_words(case.query)) | exact_identifiers(case.query)
            query_identifiers = exact_identifiers(case.query)
            started = time.perf_counter()
            score_by_idx: Counter[int] = Counter()
            for term in query_terms:
                weight = 4 if term in query_identifiers else 1
                for block_idx in postings.get(term, []):
                    score_by_idx[block_idx] += weight
            indexed_scored = [(int(score_by_idx.get(idx, 0)), idx, block) for idx, block in enumerate(blocks)]
            indexed_scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            tail_index_query_seconds = time.perf_counter() - started
            full_order = [idx for score, idx, block in full_scored]
            tail_order = [idx for score, idx, block in tail_scored]
            indexed_order = [idx for score, idx, block in indexed_scored]
            for top_k in parse_ints(args.topk_list):
                full_ids = set(full_order[:top_k])
                tail_ids = set(tail_order[:top_k])
                indexed_ids = set(indexed_order[:top_k])
                for scorer, order, seconds in (
                    ("full_words", full_order, full_seconds),
                    ("tail_words", tail_order, tail_seconds),
                    ("tail_index_query", indexed_order, tail_index_query_seconds),
                ):
                    scorer_ids = tail_ids if scorer == "tail_words" else indexed_ids if scorer == "tail_index_query" else full_ids
                    selected = [blocks[idx] for idx in order[:top_k] if 0 <= idx < len(blocks)]
                    rows.append(
                        {
                            "group": case.benchmark,
                            "task": case.task,
                            "case_id": case.case_id,
                            "block_tokens": block_tokens,
                            "top_k": top_k,
                            "scorer": scorer,
                            "answer_hit": answer_hit(selected, case.answers),
                            "score_seconds": seconds,
                            "tail_build_seconds": tail_build_seconds if scorer != "full_words" else 0.0,
                            "topk_overlap_with_full": len(full_ids & scorer_ids) / max(1, top_k),
                            "num_blocks": len(blocks),
                            "answers": json.dumps(case.answers, ensure_ascii=False),
                        }
                    )
    summary = summarize(rows)
    write_csv(args.output_dir / "tail_probe_trials.csv", rows)
    write_csv(args.output_dir / "tail_probe_summary.csv", summary)
    print("group,block,topk,scorer,samples,hit,score_seconds,tail_build_seconds,overlap,blocks")
    for row in summary:
        print(
            f"{row['group']},{row['block_tokens']},{row['top_k']},{row['scorer']},"
            f"{row['samples']},{row['answer_hit_rate']:.4f},{row['avg_score_seconds']:.6f},"
            f"{row['avg_tail_build_seconds']:.6f},"
            f"{row['avg_topk_overlap_with_full']:.4f},{row['avg_blocks']:.1f}"
        )


if __name__ == "__main__":
    main()
