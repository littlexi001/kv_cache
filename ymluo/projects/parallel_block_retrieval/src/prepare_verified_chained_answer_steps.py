from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from analyze_branch_transition_verifier import choose_branch
from run_single_query_dynamic_kv_generation import answer_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create answer-step states from frozen-verifier bridge generations."
    )
    parser.add_argument("--step_queries_path", required=True)
    parser.add_argument("--bridge_generation_rows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--exclude_query_ids", default="")
    parser.add_argument(
        "--selection_rows_path",
        default="",
        help="Optional frozen selector rows keyed by query_id.",
    )
    parser.add_argument(
        "--selection_field",
        default="heuristic_index",
        help="Branch-index field used when selection_rows_path is provided.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_generated_state(text: str) -> str:
    value = " ".join(text.strip().split())
    value = re.sub(
        r"^(?:final\s+answer|answer|bridge\s+entity)\s*[:=-]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(" \t\r\n\"'`.,;:")


def rewrite_answer_step_with_generated_state(
    answer_step: dict[str, Any], generated_state: str
) -> dict[str, Any]:
    rewritten = dict(answer_step)
    template = str(rewritten.get("step_question_template", "")).strip()
    raw_question = str(rewritten.get("official_raw_step_question", "")).strip()
    old_lookup = str(rewritten.get("lookup_key", "")).strip()
    old_question = str(rewritten["step_question"])

    if template:
        step_question = template.format(bridge=generated_state)
    elif "#1" in raw_question:
        step_question = raw_question.replace("#1", generated_state)
    elif old_lookup:
        step_question = re.sub(
            re.escape(old_lookup),
            lambda _match: generated_state,
            old_question,
            flags=re.IGNORECASE,
        )
    else:
        step_question = old_question

    state_entry = f"BRIDGE_ENTITY: {generated_state}"
    rewritten["lookup_key"] = generated_state
    rewritten["step_question"] = step_question
    rewritten["retrieval_state"] = f"{generated_state} {step_question}".strip()
    rewritten["compact_state_before"] = [state_entry]
    rewritten["full_state_before"] = [state_entry]
    return rewritten


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_ids = {
        int(item.strip()) for item in args.exclude_query_ids.split(",") if item.strip()
    }
    all_steps = read_jsonl(Path(args.step_queries_path))
    all_bridge_steps = {
        int(row["query_id"]): row
        for row in all_steps
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_bridge"
        and int(row["query_id"]) not in excluded_ids
    }
    all_answer_steps = {
        int(row["query_id"]): row
        for row in all_steps
        if str(row["split"]) == args.split
        and str(row["step_type"]) == "resolve_answer_from_bridge"
        and int(row["query_id"]) not in excluded_ids
    }
    generation_rows = {
        int(row["query_id"]): row
        for row in read_jsonl(Path(args.bridge_generation_rows_path))
        if str(row["split"]) == args.split and str(row["step_type"]) == "resolve_bridge"
        and int(row["query_id"]) not in excluded_ids
    }
    generation_ids = set(generation_rows)
    if not generation_ids:
        raise ValueError("no bridge generation rows found")
    if not generation_ids <= set(all_bridge_steps) or not generation_ids <= set(
        all_answer_steps
    ):
        raise ValueError("generation rows contain queries without paired chain steps")
    bridge_steps = {query_id: all_bridge_steps[query_id] for query_id in generation_ids}
    answer_steps = {query_id: all_answer_steps[query_id] for query_id in generation_ids}
    selection_rows = (
        {
            int(row["query_id"]): row
            for row in read_jsonl(Path(args.selection_rows_path))
        }
        if args.selection_rows_path
        else {}
    )
    if selection_rows and set(selection_rows) != generation_ids:
        raise ValueError("frozen selection rows do not exactly cover generation queries")

    output_steps = []
    traces = []
    for query_id in sorted(answer_steps):
        bridge_step = bridge_steps[query_id]
        generation = generation_rows[query_id]
        heuristic_index, score_trace = choose_branch(
            bridge_step, generation["branches"]
        )
        selected_index = (
            int(selection_rows[query_id][args.selection_field])
            if selection_rows
            else heuristic_index
        )
        if not 0 <= selected_index < len(generation["branches"]):
            raise ValueError(f"invalid selected branch {selected_index} for {query_id}")
        selected = generation["branches"][selected_index]
        generated_state = clean_generated_state(
            str(selected.get("state_text", selected["generated_text"]))
        )
        if not generated_state:
            generated_state = "(empty generation)"
        answer_step = rewrite_answer_step_with_generated_state(
            answer_steps[query_id], generated_state
        )
        answer_step["state_source"] = "frozen_transition_verifier_bridge_generation"
        answer_step["bridge_generation_branch_index"] = selected_index
        output_steps.append(answer_step)
        bridge_hit = answer_hit(generated_state, [str(bridge_step["target_output"])])
        traces.append(
            {
                "query_id": query_id,
                "selection_uses_gold": False,
                "selected_branch_index": selected_index,
                "heuristic_branch_index": heuristic_index,
                "selection_source": (
                    args.selection_field if selection_rows else "frozen_grounding_heuristic"
                ),
                "generated_bridge_state": generated_state,
                "bridge_target_hit": bridge_hit,
                "score_trace": score_trace,
            }
        )

    with (output_dir / "answer_steps.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_steps:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "bridge_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": (
            f"frozen selector {args.selection_field} selected bridge generation"
            if selection_rows
            else "frozen grounding verifier selected bridge generation"
        ),
        "selection_uses_gold": False,
        "selection_rows_path": args.selection_rows_path or None,
        "selection_field": args.selection_field if selection_rows else None,
        "split": args.split,
        "answer_steps": len(output_steps),
        "bridge_state_hit_rate": sum(row["bridge_target_hit"] for row in traces)
        / len(traces),
        "answer_steps_path": str(output_dir / "answer_steps.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
