#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_controlled_public_kv_benchmark_v1 import LONG_BENCH_PROMPTS, make_pages, token_ids  # noqa: E402
from evaluate_qwen3_top2_head_limit3_ppl import AutoTokenizer  # noqa: E402


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_hit(text: str, answers: list[str]) -> bool:
    norm_text = normalize(text)
    for answer in answers:
        norm_answer = normalize(answer)
        if len(norm_answer) >= 2 and norm_answer in norm_text:
            return True
    return False


def load_longbench_rows(zip_path: Path, tasks: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for task in tasks:
            name = f"data/{task}.jsonl"
            if name not in archive.namelist():
                continue
            for idx, line in enumerate(archive.open(name).read().decode("utf-8").splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("_id", idx))
                out[(task, sample_id)] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_results", required=True)
    parser.add_argument("--longbench_zip", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--page_tokens", type=int, required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    task_results = Path(args.task_results)
    rows = list(csv.DictReader(task_results.open("r", encoding="utf-8", newline="")))
    tasks = {row["task"] for row in rows if row.get("benchmark") == "longbench"}
    data = load_longbench_rows(Path(args.longbench_zip), tasks)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("benchmark") != "longbench":
            continue
        task = row["task"]
        sample_id = row["sample_id"]
        data_row = data.get((task, sample_id))
        if data_row is None:
            continue
        info = LONG_BENCH_PROMPTS[task]
        prefix_ids = token_ids(tokenizer, str(info["prefix"]))
        pages = make_pages(tokenizer, str(data_row["context"]), len(prefix_ids), args.page_tokens)
        selected = {
            int(item)
            for item in str(row.get("selected_pages", "")).split(",")
            if item.strip().isdigit()
        }
        answers = [str(item) for item in data_row.get("answers", [])]
        selected_text = "\n".join(page.text for page in pages if page.page_id in selected)
        full_text = "\n".join(page.text for page in pages)
        out_rows.append(
            {
                "task": task,
                "sample_id": sample_id,
                "score": row.get("score", ""),
                "selected_page_count": len(selected),
                "page_count": len(pages),
                "selected_answer_hit": int(answer_hit(selected_text, answers)),
                "full_answer_hit": int(answer_hit(full_text, answers)),
                "selected_pages": row.get("selected_pages", ""),
                "answers": json.dumps(answers, ensure_ascii=False),
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "sample_id",
        "score",
        "selected_page_count",
        "page_count",
        "selected_answer_hit",
        "full_answer_hit",
        "selected_pages",
        "answers",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in out_rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    print("task,n,selected_answer_hit,full_answer_hit,mean_score")
    for task, items in sorted(by_task.items()):
        n = len(items)
        sel = sum(int(item["selected_answer_hit"]) for item in items) / max(1, n)
        full = sum(int(item["full_answer_hit"]) for item in items) / max(1, n)
        score = sum(float(item["score"] or 0.0) for item in items) / max(1, n)
        print(f"{task},{n},{sel:.4f},{full:.4f},{score:.4f}")


if __name__ == "__main__":
    main()

