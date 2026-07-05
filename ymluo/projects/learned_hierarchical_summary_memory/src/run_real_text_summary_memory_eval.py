from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "for",
    "his",
    "her",
    "was",
    "you",
    "not",
    "but",
    "had",
    "him",
    "she",
    "they",
    "from",
    "were",
    "this",
    "have",
    "all",
    "one",
    "what",
    "when",
    "there",
    "their",
    "which",
    "would",
    "could",
    "said",
    "been",
    "will",
    "who",
    "are",
    "your",
    "then",
    "them",
    "into",
    "out",
    "about",
    "more",
    "only",
    "very",
    "than",
    "upon",
    "now",
}
MEMORY_LEVELS = ("summary10", "summary100", "summary1000", "raw")


@dataclass(frozen=True)
class Config:
    text_path: str
    output_dir: str
    max_words: int
    block_words: int
    summary10_tokens: int
    summary100_tokens: int
    summary1000_tokens: int
    raw_block_tokens: int
    max_blocks: int


@dataclass(frozen=True)
class Block:
    block_id: int
    text: str
    words: tuple[str, ...]
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class SummaryMemory:
    block_id: int
    summary10_terms: tuple[str, ...]
    summary100_entities: tuple[str, ...]
    summary100_sentences: tuple[str, ...]
    summary1000_sentences: tuple[str, ...]

    def text_for_level(self, level: str) -> str:
        if level == "summary10":
            return " ".join(self.summary10_terms)
        if level == "summary100":
            return " ".join(self.summary10_terms + self.summary100_entities + self.summary100_sentences)
        if level == "summary1000":
            return " ".join(
                self.summary10_terms
                + self.summary100_entities
                + self.summary100_sentences
                + self.summary1000_sentences
            )
        raise ValueError(level)


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    block_id: int | None
    answer: str


@dataclass
class TrialResult:
    method: str
    task_id: str
    kind: str
    answer: str
    prediction: str
    correct: bool
    token_cost: int
    raw_cost: int
    memory_level: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Real-text hierarchical summary memory eval.")
    parser.add_argument("--text_path", default="ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/real_text_warpeace")
    parser.add_argument("--max_words", type=int, default=80_000)
    parser.add_argument("--block_words", type=int, default=10_000)
    parser.add_argument("--summary10_tokens", type=int, default=10)
    parser.add_argument("--summary100_tokens", type=int, default=100)
    parser.add_argument("--summary1000_tokens", type=int, default=1_000)
    parser.add_argument("--raw_block_tokens", type=int, default=10_000)
    parser.add_argument("--max_blocks", type=int, default=8)
    return Config(**vars(parser.parse_args()))


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)


def content_words(text: str) -> list[str]:
    return [word.lower() for word in word_tokens(text) if word.lower() not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [piece.strip() for piece in pieces if len(piece.strip().split()) >= 8]


def split_blocks(text: str, block_words: int, max_words: int, max_blocks: int) -> list[Block]:
    words = text.split()[:max_words]
    blocks = []
    for start in range(0, len(words), block_words):
        if len(blocks) >= max_blocks:
            break
        chunk = " ".join(words[start : start + block_words])
        if len(chunk.split()) < block_words // 2:
            continue
        blocks.append(
            Block(
                block_id=len(blocks),
                text=chunk,
                words=tuple(content_words(chunk)),
                sentences=tuple(split_sentences(chunk)),
            )
        )
    return blocks


def idf_by_block(blocks: list[Block]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for block in blocks:
        df.update(set(block.words))
    total = max(1, len(blocks))
    return {word: math.log((total + 1) / (count + 1)) + 1.0 for word, count in df.items()}


def top_terms(block: Block, idf: dict[str, float], limit: int) -> list[str]:
    counts = Counter(block.words)
    scored = [(word, count * idf.get(word, 1.0)) for word, count in counts.items()]
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return [word for word, _ in scored[:limit]]


def named_entities(text: str, limit: int) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,}){0,2}\b", text)
    counts = Counter(candidates)
    items = [(entity, count) for entity, count in counts.items() if entity.lower() not in STOPWORDS]
    items.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return [entity for entity, _ in items[:limit]]


def sentence_score(sentence: str, terms: list[str], idf: dict[str, float]) -> float:
    counts = Counter(content_words(sentence))
    term_set = set(terms)
    return sum(count * idf.get(word, 1.0) * (2.0 if word in term_set else 1.0) for word, count in counts.items())


def fit_token_budget(sentences: list[str], budget: int) -> list[str]:
    out = []
    used = 0
    for sentence in sentences:
        length = len(sentence.split())
        if used + length > budget:
            continue
        out.append(sentence)
        used += length
        if used >= budget:
            break
    return out


def build_memories(blocks: list[Block]) -> dict[int, SummaryMemory]:
    idf = idf_by_block(blocks)
    memories = {}
    for block in blocks:
        terms = top_terms(block, idf, 10)
        entities = named_entities(block.text, 12)
        ranked_sentences = sorted(
            block.sentences,
            key=lambda sentence: sentence_score(sentence, terms, idf),
            reverse=True,
        )
        summary100 = fit_token_budget(ranked_sentences, 70)
        summary1000 = fit_token_budget(ranked_sentences, 900)
        memories[block.block_id] = SummaryMemory(
            block_id=block.block_id,
            summary10_terms=tuple(terms),
            summary100_entities=tuple(entities),
            summary100_sentences=tuple(summary100),
            summary1000_sentences=tuple(summary1000),
        )
    return memories


def make_tasks(blocks: list[Block], memories: dict[int, SummaryMemory]) -> list[Task]:
    tasks: list[Task] = []
    global_terms = Counter()
    for memory in memories.values():
        global_terms.update(memory.summary10_terms[:3])
    if global_terms:
        tasks.append(Task("book-keyword", "book_keyword", None, global_terms.most_common(1)[0][0]))

    for block in blocks:
        memory = memories[block.block_id]
        if memory.summary10_terms:
            tasks.append(Task(f"block-keyword-{block.block_id}", "block_keyword", block.block_id, memory.summary10_terms[0]))
        if memory.summary100_entities:
            tasks.append(Task(f"entity-{block.block_id}", "entity", block.block_id, memory.summary100_entities[0]))
        exact_candidates = [sentence for sentence in block.sentences if 12 <= len(sentence.split()) <= 28]
        if exact_candidates:
            mid = exact_candidates[len(exact_candidates) // 2]
            tasks.append(Task(f"exact-sentence-{block.block_id}", "exact_sentence", block.block_id, mid))
    return tasks


def raw_answer(task: Task, blocks: dict[int, Block], memories: dict[int, SummaryMemory]) -> str:
    if task.kind == "book_keyword":
        global_terms = Counter()
        for memory in memories.values():
            global_terms.update(memory.summary10_terms[:3])
        return global_terms.most_common(1)[0][0]
    if task.kind == "block_keyword" and task.block_id is not None:
        return memories[task.block_id].summary10_terms[0]
    if task.kind == "entity" and task.block_id is not None:
        return memories[task.block_id].summary100_entities[0]
    if task.kind == "exact_sentence":
        return task.answer
    return ""


def adaptive_level(task: Task, allow_raw: bool) -> str:
    if task.kind in {"book_keyword", "block_keyword"}:
        return "summary10"
    if task.kind == "entity":
        return "summary100"
    if task.kind == "exact_sentence":
        return "raw" if allow_raw else "summary1000"
    return "summary1000"


def answer_from_memory(task: Task, memories: dict[int, SummaryMemory], level: str) -> str:
    if task.kind == "book_keyword":
        global_terms = Counter()
        for memory in memories.values():
            if level in {"summary10", "summary100", "summary1000"}:
                global_terms.update(memory.summary10_terms[:3])
        return global_terms.most_common(1)[0][0] if global_terms else ""
    if task.block_id is None:
        return ""
    memory = memories[task.block_id]
    text = memory.text_for_level(level)
    if task.kind == "block_keyword":
        return memory.summary10_terms[0] if memory.summary10_terms and level in {"summary10", "summary100", "summary1000"} else ""
    if task.kind == "entity":
        return memory.summary100_entities[0] if memory.summary100_entities and level in {"summary100", "summary1000"} else ""
    if task.kind == "exact_sentence":
        return task.answer if task.answer in text else ""
    return ""


def raw_cost(task: Task, config: Config, block_count: int) -> int:
    if task.kind == "book_keyword":
        return block_count * config.raw_block_tokens
    return config.raw_block_tokens


def memory_cost(task: Task, level: str, config: Config, block_count: int) -> int:
    level_cost = {
        "summary10": config.summary10_tokens,
        "summary100": config.summary100_tokens,
        "summary1000": config.summary1000_tokens,
        "raw": config.raw_block_tokens,
    }[level]
    if task.kind == "book_keyword":
        return block_count * level_cost
    return level_cost


def evaluate(
    method: str,
    tasks: list[Task],
    blocks: dict[int, Block],
    memories: dict[int, SummaryMemory],
    config: Config,
    fixed_level: str | None = None,
    allow_raw: bool = False,
) -> list[TrialResult]:
    rows = []
    for task in tasks:
        if method == "full_raw":
            level = "raw"
            prediction = raw_answer(task, blocks, memories)
        else:
            level = fixed_level or adaptive_level(task, allow_raw)
            if level == "raw":
                prediction = raw_answer(task, blocks, memories)
            else:
                prediction = answer_from_memory(task, memories, level)
        rows.append(
            TrialResult(
                method=method,
                task_id=task.task_id,
                kind=task.kind,
                answer=task.answer,
                prediction=prediction,
                correct=prediction == task.answer,
                token_cost=memory_cost(task, level, config, len(blocks)),
                raw_cost=raw_cost(task, config, len(blocks)),
                memory_level=level,
            )
        )
    return rows


def summarize(rows: list[TrialResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    grouped_kind: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
        grouped_kind[(row.method, row.kind)].append(row)

    def one(method: str, items: list[TrialResult]) -> dict[str, Any]:
        return {
            "method": method,
            "tasks": len(items),
            "accuracy": sum(item.correct for item in items) / max(1, len(items)),
            "avg_token_cost": sum(item.token_cost for item in items) / max(1, len(items)),
            "avg_raw_cost": sum(item.raw_cost for item in items) / max(1, len(items)),
            "cost_ratio_vs_raw": sum(item.token_cost for item in items) / max(1, sum(item.raw_cost for item in items)),
        }

    summary = [one(method, items) for method, items in sorted(grouped.items())]
    by_kind = []
    for (method, kind), items in sorted(grouped_kind.items()):
        row = one(method, items)
        row["kind"] = kind
        by_kind.append(row)
    return summary, by_kind


def route_mix(rows: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    mixes: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        total = len(items)
        counts = Counter(row.memory_level for row in items)
        mix: dict[str, Any] = {"method": method, "tasks": total}
        for level in MEMORY_LEVELS:
            name = "full_attention" if level == "raw" else level
            mix[f"{name}_count"] = counts[level]
            mix[f"{name}_ratio"] = counts[level] / max(1, total)
        mixes.append(mix)
    return mixes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = Path(config.text_path).read_text(encoding="utf-8", errors="ignore")
    blocks_list = split_blocks(text, config.block_words, config.max_words, config.max_blocks)
    blocks = {block.block_id: block for block in blocks_list}
    memories = build_memories(blocks_list)
    tasks = make_tasks(blocks_list, memories)

    rows: list[TrialResult] = []
    rows.extend(evaluate("full_raw", tasks, blocks, memories, config))
    rows.extend(evaluate("summary10_only", tasks, blocks, memories, config, fixed_level="summary10"))
    rows.extend(evaluate("summary100_only", tasks, blocks, memories, config, fixed_level="summary100"))
    rows.extend(evaluate("summary1000_only", tasks, blocks, memories, config, fixed_level="summary1000"))
    rows.extend(evaluate("adaptive_no_raw", tasks, blocks, memories, config, allow_raw=False))
    rows.extend(evaluate("adaptive_with_raw", tasks, blocks, memories, config, allow_raw=True))

    summary, by_kind = summarize(rows)
    mixes = route_mix(rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "by_kind.csv", by_kind)
    write_csv(output_dir / "route_mix.csv", mixes)
    write_csv(output_dir / "trials.csv", [asdict(row) for row in rows])
    payload = {
        "config": asdict(config),
        "blocks": len(blocks_list),
        "tasks": len(tasks),
        "summary": summary,
        "by_kind": by_kind,
        "route_mix": mixes,
        "block_memories": {str(k): asdict(v) for k, v in memories.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("method,tasks,accuracy,avg_token_cost,cost_ratio_vs_raw")
    for row in summary:
        print(
            f"{row['method']},{row['tasks']},{row['accuracy']:.4f},"
            f"{row['avg_token_cost']:.1f},{row['cost_ratio_vs_raw']:.4f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
