from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_NEEDLE = (
    "The best thing to do in San Francisco is eat a sandwich "
    "and sit in Dolores Park on a sunny day."
)
DEFAULT_ANSWER = "eat a sandwich and sit in Dolores Park"
DEFAULT_QUESTION = "What is the best thing to do in San Francisco?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expanded Needle-In-A-Haystack JSONL samples from local essay text."
    )
    parser.add_argument(
        "--haystack_dir",
        default="external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays",
        help="Directory containing .txt haystack source files.",
    )
    parser.add_argument(
        "--output_dir",
        default="ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack",
    )
    parser.add_argument(
        "--context_lengths",
        default="1000,2000,4000,8000,16000",
        help="Comma-separated approximate word-token context lengths before needle insertion.",
    )
    parser.add_argument(
        "--depth_percents",
        default="0,10,25,50,75,90,100",
        help="Comma-separated needle insertion depths.",
    )
    parser.add_argument("--needle_text", default=DEFAULT_NEEDLE)
    parser.add_argument("--expected_answer", default=DEFAULT_ANSWER)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--include_answer_prompt",
        action="store_true",
        help='Append "Answer:" after the question.',
    )
    return parser.parse_args()


def parse_number_list(value: str, number_type: type[int] | type[float]) -> list[int] | list[float]:
    parsed = [number_type(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one number.")
    return parsed


def read_haystack_text(haystack_dir: Path) -> str:
    files = sorted(haystack_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under {haystack_dir}")
    parts: list[str] = []
    for path in files:
        parts.append(path.read_text(encoding="utf-8", errors="ignore").strip())
    return "\n\n".join(part for part in parts if part)


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def repeated_prefix(source: str, target_words: int) -> str:
    spans = word_spans(source)
    if len(spans) >= target_words:
        end = spans[target_words - 1][1]
        return source[:end]

    chunks: list[str] = []
    running_words = 0
    while running_words < target_words:
        chunks.append(source)
        running_words += len(spans)
    combined = "\n\n".join(chunks)
    combined_spans = word_spans(combined)
    end = combined_spans[target_words - 1][1]
    return combined[:end]


def snap_to_sentence_boundary(text: str, char_index: int) -> int:
    if char_index <= 0:
        return 0
    if char_index >= len(text):
        return len(text)
    window_start = max(0, char_index - 1200)
    prefix = text[window_start:char_index]
    matches = list(re.finditer(r"[.!?]\s+", prefix))
    if not matches:
        return char_index
    return window_start + matches[-1].end()


def insertion_char_for_depth(context: str, depth_percent: float) -> int:
    if depth_percent <= 0:
        return 0
    if depth_percent >= 100:
        return len(context)
    spans = word_spans(context)
    word_index = min(len(spans), max(0, int(len(spans) * depth_percent / 100.0)))
    if word_index >= len(spans):
        return len(context)
    return snap_to_sentence_boundary(context, spans[word_index][0])


def build_prompt(context: str, question: str, include_answer_prompt: bool) -> str:
    suffix = "\nAnswer:" if include_answer_prompt else ""
    return f"{context}\n\nQuestion: {question}{suffix}"


def insert_needle_text(context: str, insertion_char: int, needle_text: str) -> str:
    if insertion_char <= 0:
        return needle_text + "\n\n" + context.lstrip()
    if insertion_char >= len(context):
        return context.rstrip() + "\n\n" + needle_text
    return (
        context[:insertion_char].rstrip()
        + "\n\n"
        + needle_text
        + "\n\n"
        + context[insertion_char:].lstrip()
    )


def main() -> None:
    args = parse_args()
    haystack_dir = Path(args.haystack_dir)
    output_dir = Path(args.output_dir)
    prompt_dir = output_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    context_lengths = [int(value) for value in parse_number_list(args.context_lengths, int)]
    depth_percents = [float(value) for value in parse_number_list(args.depth_percents, float)]
    source = read_haystack_text(haystack_dir)

    jsonl_path = output_dir / "needle_in_haystack.jsonl"
    manifest_path = output_dir / "manifest.csv"
    rows: list[dict[str, object]] = []

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl:
        for context_length in context_lengths:
            base_context = repeated_prefix(source, context_length)
            base_word_count = len(word_spans(base_context))
            for depth in depth_percents:
                insertion_char = insertion_char_for_depth(base_context, depth)
                context_with_needle = insert_needle_text(
                    base_context, insertion_char, args.needle_text
                )
                needle_char_start = context_with_needle.index(args.needle_text)
                needle_char_end = needle_char_start + len(args.needle_text)
                actual_depth = 100.0 * insertion_char / max(1, len(base_context))
                sample_id = f"niah_len{context_length}_depth{depth:g}"
                prompt = build_prompt(context_with_needle, args.question, args.include_answer_prompt)
                prompt_path = prompt_dir / f"{sample_id}.txt"
                prompt_path.write_text(prompt, encoding="utf-8", newline="\n")

                record = {
                    "sample_id": sample_id,
                    "task_type": "single_needle",
                    "haystack_source": str(haystack_dir),
                    "context_length_words_target": context_length,
                    "context_length_words_actual_before_insert": base_word_count,
                    "target_depth_percent": depth,
                    "actual_depth_percent_by_char": actual_depth,
                    "needle_text": args.needle_text,
                    "expected_answer": args.expected_answer,
                    "question": args.question,
                    "needle_char_start_in_context": needle_char_start,
                    "needle_char_end_in_context": needle_char_end,
                    "context": context_with_needle,
                    "prompt": prompt,
                    "prompt_path": str(prompt_path),
                }
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "context_length_words_target": context_length,
                        "target_depth_percent": depth,
                        "actual_depth_percent_by_char": f"{actual_depth:.4f}",
                        "needle_char_start_in_context": needle_char_start,
                        "needle_char_end_in_context": needle_char_end,
                        "prompt_path": str(prompt_path),
                    }
                )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(manifest_path),
        "prompt_dir": str(prompt_dir),
        "sample_count": len(rows),
        "context_lengths": context_lengths,
        "depth_percents": depth_percents,
        "needle_text": args.needle_text,
        "expected_answer": args.expected_answer,
        "question": args.question,
        "length_unit": "approximate whitespace-delimited words before insertion",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
