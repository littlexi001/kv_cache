from __future__ import annotations

import json
from pathlib import Path


ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def number_word(value: int) -> str:
    if value < 20:
        return ONES[value]
    if value < 100:
        tens = value // 10 * 10
        rest = value % 10
        return TENS[tens] if rest == 0 else f"{TENS[tens]} {ONES[rest]}"
    raise ValueError(value)


def count_text(count: int, numeric: bool) -> str:
    return str(count) if numeric else number_word(count)


def apple_phrase(count: int, numeric: bool) -> str:
    unit = "apple" if count == 1 else "apples"
    return f"Ming ate {count_text(count, numeric)} {unit}"


def make_sample(sample_index: int, counts: list[int], offset: int, numeric: bool) -> dict[str, object]:
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "the next Monday",
        "the next Tuesday",
        "the next Wednesday",
        "the next Thursday",
        "the next Friday",
        "the next Saturday",
        "the next Sunday",
        "the final Monday",
        "the final Tuesday",
        "the final Wednesday",
        "the final Thursday",
    ]
    fillers = [
        "He wrote a short note about the weather and packed his school bag.",
        "He walked past the library, watered a small plant, and practiced handwriting.",
        "He helped tidy the kitchen, read a comic, and listened to a song before dinner.",
        "He described a quiet street, a blue notebook, and a bus ride home.",
        "He mentioned homework, a cup of tea, and a phone call with his cousin.",
    ]
    zero_lines = [
        "Ming did not write down any apple snack that day.",
        "There was no apple entry in the diary for that day.",
        "The diary talked about lunch but gave no apple count.",
        "Ming wrote about oranges and bread, not apples.",
    ]
    evidence_lines: list[str] = []
    lines: list[str] = []
    count_index = 0
    total_days = max(len(days), offset + (len(counts) - 1) * 2 + 4)
    for day_index in range(total_days):
        day = days[day_index % len(days)]
        filler = fillers[(sample_index + day_index) % len(fillers)]
        day_label = str(day_index + 1) if numeric else number_word(day_index + 1)
        line = f"Day {day_label} ({day}): {filler}"
        should_place_count = count_index < len(counts) and (day_index - offset) >= 0 and (day_index - offset) % 2 == 0
        if should_place_count:
            phrase = apple_phrase(counts[count_index], numeric)
            line = f"{line} {phrase} after dinner."
            evidence_lines.append(f"{phrase} after dinner.")
            count_index += 1
        else:
            line = f"{line} {zero_lines[(sample_index + day_index) % len(zero_lines)]}"
        lines.append(line)

    total = sum(counts)
    answer = str(total) if numeric else number_word(total)
    instruction = (
        "Use the diary context to answer the question. Add every apple count that Ming recorded. Answer with digits only."
        if numeric
        else "Use the diary context to answer the question. Add every apple count that Ming recorded. Answer with English number words only."
    )
    context = (
        "Ming kept this diary over several days. Some entries record apple snacks, while other entries only record daily life.\n"
        + "\n".join(lines)
    )
    return {
        "sample_id": f"ming_apple_diary_{sample_index:02d}",
        "question": "How many apples did Ming eat in total?",
        "answer": answer,
        "instruction": instruction,
        "context": context,
        "answer_evidence_texts": evidence_lines,
        "metadata": {
            "apple_counts": [number_word(count) for count in counts],
            "apple_counts_numeric": [str(count) for count in counts],
            "apple_count_values": counts,
            "total_value": total,
            "total_word": number_word(total),
            "total_numeric": str(total),
            "evidence_count": len(evidence_lines),
        },
    }


def write_dataset(numeric: bool) -> None:
    specs = [
        ([1, 2, 3, 1, 4, 2, 3, 1], 0),
        ([2, 1, 2, 3, 2, 1, 4, 2, 3], 1),
        ([3, 2, 1, 4, 1, 3, 2, 2, 1, 3], 2),
        ([1, 1, 2, 2, 3, 3, 1, 4, 2, 1, 2], 3),
        ([4, 1, 3, 2, 4, 1, 2, 3, 1, 2, 4, 1], 0),
        ([2, 2, 2, 1, 3, 4, 1, 1, 2, 3, 2, 1, 4], 1),
        ([1, 3, 1, 3, 2, 4, 2, 1, 3, 2, 1, 4, 2, 3], 2),
        ([3, 1, 4, 2, 1, 3, 2, 4, 1, 2, 3, 1, 2, 4, 1], 3),
    ]
    samples = [make_sample(index, counts, offset, numeric) for index, (counts, offset) in enumerate(specs, start=1)]
    suffix = "numeric" if numeric else "words"
    out = Path(f"ymluo/projects/qwen3_top1_category_ablation/data/ming_apple_diary_sum_8_{suffix}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(out)
    for sample in samples:
        meta = sample["metadata"]
        total = meta["total_numeric"] if numeric else meta["total_word"]
        print(sample["sample_id"], total, meta["evidence_count"])


def main() -> None:
    write_dataset(numeric=False)
    write_dataset(numeric=True)


if __name__ == "__main__":
    main()
