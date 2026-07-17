from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import run_length_causal_mechanism_20260717 as causal
import run_local_rule_failure_boundary as base


STYLES = (
    "zero_answer",
    "answer_prefix",
    "fewshot_answer",
    "path",
    "path_prefix",
    "two_fields_prefix",
    "state_fields",
    "state_fields_prefix",
    "rule_state_fields",
    "thinking_answer",
)


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in csv_items(value)]


def task_text(style: str, body_text: str, start_code: str) -> str:
    common = (
        f"{body_text}\nTWO-STEP TASK\nStart code: {start_code}\n"
        "Use exactly two applicable VERIFIED RULE transitions. Ignore NOTE and DECOY RULE lines.\n"
        "The result must not be the start code or the one-step code.\n"
    )
    if style in {"zero_answer", "answer_prefix", "fewshot_answer", "thinking_answer"}:
        return common + "Return one complete final code only, with no explanation."
    if style in {"path", "path_prefix"}:
        return common + (
            "Return exactly: PATH: <start code> -> <one-step code> -> <final code>.\n"
            "Do not add any other text."
        )
    if style == "two_fields_prefix":
        return common + (
            "Return exactly two lines:\nSTEP1: <one-step code>\nFINAL: <final code>\n"
            "Do not add any other text."
        )
    if style in {"state_fields", "state_fields_prefix"}:
        return common + (
            "Treat this as deterministic state update, not free-form text generation.\n"
            "STATE0 is the given start code. STATE1 is the THEN-code of the VERIFIED RULE "
            "whose IF-code equals STATE0. STATE2 repeats the same operation from STATE1.\n"
            "Copy every code exactly. Return exactly three lines:\n"
            "STATE0: <start code>\nSTATE1: <one-step code>\nSTATE2: <final code>"
        )
    if style == "rule_state_fields":
        return common + (
            "First select the VERIFIED RULE whose IF-code exactly matches the current state; "
            "then copy its THEN-code. Repeat once.\n"
            "Return exactly four lines:\nRULE1: <rule label>\nSTATE1: <one-step code>\n"
            "RULE2: <rule label>\nFINAL: <final code>"
        )
    raise ValueError(f"unknown style: {style}")


def assistant_prefix(style: str, start_code: str) -> str:
    if style == "answer_prefix":
        return "FINAL CODE: "
    if style == "path_prefix":
        return f"PATH: {start_code} -> "
    if style == "two_fields_prefix":
        return "STEP1: "
    if style == "state_fields_prefix":
        return f"STATE0: {start_code}\nSTATE1: "
    return ""


def conversation(style: str, user_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if style == "fewshot_answer":
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        "VERIFIED RULE E0: IF QA10-101 IS ACTIVE THEN QB20-202 BECOMES ACTIVE.\n"
                        "VERIFIED RULE E1: IF QB20-202 IS ACTIVE THEN QC30-303 BECOMES ACTIVE.\n"
                        "TWO-STEP TASK\nStart code: QA10-101\n"
                        "Return one complete final code only, with no explanation."
                    ),
                },
                {"role": "assistant", "content": "QC30-303"},
            ]
        )
    messages.append({"role": "user", "content": user_text})
    return messages


def apply_chat(tokenizer: Any, messages: list[dict[str, str]], thinking: bool) -> list[int]:
    kwargs = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        return list(tokenizer.apply_chat_template(**kwargs, enable_thinking=thinking))
    except TypeError:
        return list(tokenizer.apply_chat_template(**kwargs))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate concise and structured Qwen prompts.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=8)
    parser.add_argument("--lengths", default="0")
    parser.add_argument("--conditions", default="clean,conflict1")
    parser.add_argument("--styles", default=",".join(STYLES))
    parser.add_argument("--max_new_tokens", type=int, default=96)
    args = parser.parse_args()

    styles = csv_items(args.styles)
    unknown = sorted(set(styles) - set(STYLES))
    if unknown:
        raise ValueError(f"unknown styles: {unknown}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    rows: list[dict[str, Any]] = []
    lengths = csv_ints(args.lengths)
    conditions = csv_items(args.conditions)
    total = args.num_seeds * len(lengths) * len(conditions) * len(styles)
    done = 0
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for length in lengths:
            for condition in conditions:
                body = causal.build_body(
                    tokenizer,
                    seed=seed,
                    target_context_tokens=length,
                    condition=condition,
                    placement="middle",
                )
                body_text = tokenizer.decode(body["body_ids"][0], skip_special_tokens=True)
                start_code = body["gold_codes"][0]
                gold_intermediate = body["gold_codes"][1]
                gold_final = body["gold_codes"][2]
                for style in styles:
                    user_text = task_text(style, body_text, start_code)
                    input_ids = apply_chat(
                        tokenizer,
                        conversation(style, user_text),
                        thinking=style == "thinking_answer",
                    )
                    prefix = assistant_prefix(style, start_code)
                    input_ids.extend(base.token_ids(tokenizer, prefix))
                    tensor = torch.tensor(input_ids, dtype=torch.long, device="cuda").view(1, -1)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(
                            tensor,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    elapsed = time.perf_counter() - started
                    generated = tokenizer.decode(
                        output[0, tensor.shape[1] :], skip_special_tokens=True
                    )
                    response = prefix + generated
                    mentions = base.extract_known_code_mentions(
                        response,
                        gold_final,
                        body["events"],
                        body["candidates"],
                    )
                    known_codes = [str(item["answer"]) for item in mentions]
                    final_known = known_codes[-1] if known_codes else ""
                    intermediate_positions = [
                        index for index, code in enumerate(known_codes) if code == gold_intermediate
                    ]
                    final_positions = [
                        index for index, code in enumerate(known_codes) if code == gold_final
                    ]
                    ordered_path = int(
                        bool(intermediate_positions)
                        and bool(final_positions)
                        and intermediate_positions[0] < final_positions[-1]
                    )
                    row = {
                        "seed": seed,
                        "target_context_tokens": length,
                        "condition": condition,
                        "style": style,
                        "prompt_tokens": int(tensor.shape[1]),
                        "generated_tokens": int(output.shape[1] - tensor.shape[1]),
                        "gold_intermediate": gold_intermediate,
                        "gold_final": gold_final,
                        "generated_text": generated.replace("\n", "\\n"),
                        "rendered_response": response.replace("\n", "\\n"),
                        "known_codes": " ".join(known_codes),
                        "final_known": final_known,
                        "final_correct": int(final_known == gold_final),
                        "contains_final": int(gold_final in known_codes),
                        "contains_intermediate": int(gold_intermediate in known_codes),
                        "ordered_gold_path": ordered_path,
                        "hit_token_limit": int(
                            int(output.shape[1] - tensor.shape[1]) >= args.max_new_tokens
                        ),
                        "generation_seconds": elapsed,
                    }
                    rows.append(row)
                    done += 1
                    print(
                        f"[{done}/{total}] seed={seed} length={length} condition={condition} "
                        f"style={style} final={row['final_correct']} path={ordered_path} "
                        f"response={response[:100]!r}",
                        flush=True,
                    )

    write_csv(output_dir / "results.csv", rows)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_context_tokens"], row["condition"], row["style"])].append(row)
    summary = []
    for (length, condition, style), selected in sorted(grouped.items()):
        summary.append(
            {
                "target_context_tokens": length,
                "condition": condition,
                "style": style,
                "sample_count": len(selected),
                "final_accuracy": statistics.mean(row["final_correct"] for row in selected),
                "contains_final_rate": statistics.mean(row["contains_final"] for row in selected),
                "ordered_gold_path_rate": statistics.mean(
                    row["ordered_gold_path"] for row in selected
                ),
                "token_limit_rate": statistics.mean(row["hit_token_limit"] for row in selected),
                "mean_generated_tokens": statistics.mean(
                    row["generated_tokens"] for row in selected
                ),
            }
        )
    write_csv(output_dir / "summary.csv", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
