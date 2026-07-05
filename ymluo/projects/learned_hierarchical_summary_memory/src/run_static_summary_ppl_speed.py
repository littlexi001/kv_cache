from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
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


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    text_paths: tuple[str, ...]
    dataset_names: tuple[str, ...]
    methods: tuple[str, ...]
    samples_per_dataset: int
    sample_stride_tokens: int
    prefill_tokens: int
    eval_tokens: int
    block_tokens: int
    recent_tokens: int
    summary10_words: int
    summary100_words: int
    summary1000_words: int
    max_text_tokens: int
    device: str
    dtype: str
    attn_implementation: str
    summary_backend: str = "heuristic"
    learned_summary_train_tokens: int = 60_000
    learned_summary_epochs: int = 8
    learned_summary_hidden_dim: int = 32
    learned_summary_lr: float = 3e-3
    learned_summary_max_sentences: int = 20_000
    learned_summary_seed: int = 2026070307


@dataclass(frozen=True)
class Block:
    block_id: int
    text: str
    words: tuple[str, ...]
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class StaticSummary:
    block_id: int
    summary10: str
    summary100: str
    summary1000: str

    def text_for_level(self, level: str) -> str:
        if level == "summary10":
            return self.summary10
        if level == "summary100":
            return self.summary100
        if level == "summary1000":
            return self.summary1000
        raise ValueError(level)


@dataclass
class LearnedSummaryScorer:
    model: Any
    mean: tuple[float, ...]
    std: tuple[float, ...]
    metadata: dict[str, Any]

    def score(self, features: list[list[float]]) -> list[float]:
        import torch

        if not features:
            return []
        normalized = [
            [(value - self.mean[idx]) / max(self.std[idx], 1e-6) for idx, value in enumerate(row)]
            for row in features
        ]
        with torch.inference_mode():
            tensor = torch.tensor(normalized, dtype=torch.float32)
            logits = self.model(tensor).view(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
        return [float(score) for score in scores]


@dataclass
class TrialResult:
    dataset: str
    sample_id: int
    method: str
    prompt_tokens: int
    eval_tokens: int
    total_input_tokens: int
    nll: float
    ppl: float
    forward_seconds: float
    tokens_per_second: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Static summary memory PPL/speed for ordinary text generation.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_ppl_speed")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument(
        "--text_paths",
        default=(
            "ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt,"
            "ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt"
        ),
    )
    parser.add_argument("--dataset_names", default="warpeace,montecristo")
    parser.add_argument(
        "--methods",
        default="full_raw,recent_only,static_sum10,static_sum100,static_sum1000,static_hier",
    )
    parser.add_argument("--samples_per_dataset", type=int, default=4)
    parser.add_argument("--sample_stride_tokens", type=int, default=2048)
    parser.add_argument("--prefill_tokens", type=int, default=8192)
    parser.add_argument("--eval_tokens", type=int, default=128)
    parser.add_argument("--block_tokens", type=int, default=2048)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--summary10_words", type=int, default=10)
    parser.add_argument("--summary100_words", type=int, default=100)
    parser.add_argument("--summary1000_words", type=int, default=900)
    parser.add_argument("--max_text_tokens", type=int, default=80_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--summary_backend", choices=["heuristic", "learned"], default="heuristic")
    parser.add_argument("--learned_summary_train_tokens", type=int, default=60_000)
    parser.add_argument("--learned_summary_epochs", type=int, default=8)
    parser.add_argument("--learned_summary_hidden_dim", type=int, default=32)
    parser.add_argument("--learned_summary_lr", type=float, default=3e-3)
    parser.add_argument("--learned_summary_max_sentences", type=int, default=20_000)
    parser.add_argument("--learned_summary_seed", type=int, default=2026070307)
    args = parser.parse_args()
    text_paths = tuple(item.strip() for item in args.text_paths.split(",") if item.strip())
    dataset_names = tuple(item.strip() for item in args.dataset_names.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if len(dataset_names) != len(text_paths):
        raise ValueError("--dataset_names must have the same count as --text_paths")
    return Config(**{**vars(args), "text_paths": text_paths, "dataset_names": dataset_names, "methods": methods})


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)


def content_words(text: str) -> list[str]:
    return [word.lower() for word in word_tokens(text) if word.lower() not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [piece.strip() for piece in pieces if len(piece.strip().split()) >= 8]


def sentence_score(sentence: str, terms: list[str], idf: dict[str, float]) -> float:
    counts = Counter(content_words(sentence))
    term_set = set(terms)
    return sum(count * idf.get(word, 1.0) * (2.0 if word in term_set else 1.0) for word, count in counts.items())


def sentence_features(sentence: str, index: int, total: int, terms: list[str], idf: dict[str, float]) -> list[float]:
    words = content_words(sentence)
    counts = Counter(words)
    term_set = set(terms)
    word_count = max(1, len(sentence.split()))
    content_count = max(1, len(words))
    tfidf = sum(count * idf.get(word, 1.0) for word, count in counts.items())
    boosted = sum(count * idf.get(word, 1.0) * (2.0 if word in term_set else 1.0) for word, count in counts.items())
    overlap = sum(1 for word in set(words) if word in term_set) / max(1, len(term_set))
    entities = named_entities(sentence, 20)
    capitalized = len(re.findall(r"\b[A-Z][a-z]{3,}\b", sentence))
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)*\b", sentence))
    relative_position = index / max(1, total - 1)
    avg_idf = tfidf / content_count
    return [
        math.log1p(word_count),
        math.log1p(content_count),
        relative_position,
        1.0 - relative_position,
        math.log1p(tfidf),
        math.log1p(boosted),
        overlap,
        len(set(words)) / content_count,
        math.log1p(len(entities)),
        math.log1p(capitalized),
        math.log1p(numbers),
        avg_idf,
        1.0 if ":" in sentence or ";" in sentence else 0.0,
    ]


def fit_word_budget(parts: list[str], budget: int) -> str:
    out: list[str] = []
    used = 0
    for part in parts:
        words = part.split()
        if not words:
            continue
        if used + len(words) > budget:
            remaining = budget - used
            if remaining > 0:
                out.append(" ".join(words[:remaining]))
            break
        out.append(part)
        used += len(words)
        if used >= budget:
            break
    return " ".join(out)


def fit_token_budget(tokenizer: Any, parts: list[str], budget: int) -> str:
    if budget <= 0:
        return ""
    out_ids: list[int] = []
    for part in parts:
        ids = tokenizer(part, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        remaining = budget - len(out_ids)
        if remaining <= 0:
            break
        out_ids.extend(ids[:remaining])
        if len(out_ids) >= budget:
            break
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def ratio_for_method(method: str) -> float | None:
    return {
        "summary1_8": 1.0 / 8.0,
        "summary1_4": 1.0 / 4.0,
        "summary1_2": 1.0 / 2.0,
        "static_ratio_1_8": 1.0 / 8.0,
        "static_ratio_1_4": 1.0 / 4.0,
        "static_ratio_1_2": 1.0 / 2.0,
    }.get(method)


def build_blocks(tokenizer: Any, token_ids: list[int], block_tokens: int) -> list[Block]:
    blocks: list[Block] = []
    for start in range(0, len(token_ids), block_tokens):
        ids = token_ids[start : start + block_tokens]
        if len(ids) < max(32, block_tokens // 4):
            continue
        text = tokenizer.decode(ids, skip_special_tokens=True)
        blocks.append(
            Block(
                block_id=len(blocks),
                text=text,
                words=tuple(content_words(text)),
                sentences=tuple(split_sentences(text)),
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


def rank_sentences_with_learned_scorer(
    block: Block,
    terms: list[str],
    idf: dict[str, float],
    summary_scorer: LearnedSummaryScorer,
) -> list[str]:
    if not block.sentences:
        return []
    features = [
        sentence_features(sentence, idx, len(block.sentences), terms, idf)
        for idx, sentence in enumerate(block.sentences)
    ]
    scored = list(zip(block.sentences, summary_scorer.score(features)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [sentence for sentence, _ in scored]


def build_static_summaries(
    config: Config,
    blocks: list[Block],
    summary_scorer: LearnedSummaryScorer | None = None,
) -> list[StaticSummary]:
    idf = idf_by_block(blocks)
    summaries: list[StaticSummary] = []
    for block in blocks:
        terms = top_terms(block, idf, config.summary10_words)
        entities = named_entities(block.text, 12)
        if config.summary_backend == "learned":
            if summary_scorer is None:
                raise ValueError("summary_backend=learned requires a trained summary_scorer")
            ranked_sentences = rank_sentences_with_learned_scorer(block, terms, idf, summary_scorer)
            summary10 = fit_word_budget(ranked_sentences, config.summary10_words) or " ".join(terms)
            summary100 = fit_word_budget(ranked_sentences, config.summary100_words)
        elif config.summary_backend == "heuristic":
            ranked_sentences = sorted(
                block.sentences,
                key=lambda sentence: sentence_score(sentence, terms, idf),
                reverse=True,
            )
            summary10 = " ".join(terms)
            summary100 = fit_word_budget([" ".join(terms), " ".join(entities)] + ranked_sentences, config.summary100_words)
        else:
            raise ValueError(config.summary_backend)
        summary1000 = fit_word_budget(ranked_sentences, config.summary1000_words)
        summaries.append(
            StaticSummary(
                block_id=block.block_id,
                summary10=summary10,
                summary100=summary100,
                summary1000=summary1000,
            )
        )
    return summaries


def context_for_method(
    config: Config,
    tokenizer: Any,
    prefix_ids: list[int],
    method: str,
    summary_scorer: LearnedSummaryScorer | None = None,
) -> str:
    if method == "full_raw":
        return tokenizer.decode(prefix_ids, skip_special_tokens=True)

    recent_ids = prefix_ids[-config.recent_tokens :] if config.recent_tokens > 0 else []
    older_ids = prefix_ids[: max(0, len(prefix_ids) - len(recent_ids))]
    recent_text = tokenizer.decode(recent_ids, skip_special_tokens=True)
    if method == "recent_only":
        return recent_text

    blocks = build_blocks(tokenizer, older_ids, config.block_tokens)
    summaries = build_static_summaries(config, blocks, summary_scorer=summary_scorer)

    if method == "static_sum10":
        summary_text = "\n".join(item.summary10 for item in summaries)
    elif method == "static_sum100":
        summary_text = "\n".join(item.summary100 for item in summaries)
    elif method == "static_sum1000":
        summary_text = "\n".join(item.summary1000 for item in summaries)
    elif ratio_for_method(method) is not None:
        ratio = ratio_for_method(method)
        if ratio is None:
            raise ValueError(method)
        idf = idf_by_block(blocks)
        parts = []
        for block in blocks:
            terms = top_terms(block, idf, config.summary10_words)
            if config.summary_backend == "learned":
                if summary_scorer is None:
                    raise ValueError("summary_backend=learned requires a trained summary_scorer")
                ranked_sentences = rank_sentences_with_learned_scorer(block, terms, idf, summary_scorer)
            else:
                ranked_sentences = sorted(
                    block.sentences,
                    key=lambda sentence: sentence_score(sentence, terms, idf),
                    reverse=True,
                )
            block_ids = tokenizer(block.text, add_special_tokens=False)["input_ids"]
            token_budget = max(8, int(round(len(block_ids) * ratio)))
            summary = fit_token_budget(tokenizer, [" ".join(terms)] + ranked_sentences, token_budget)
            if summary:
                parts.append(f"[block {block.block_id}] {summary}")
        summary_text = "\n".join(parts)
    elif method == "static_hier":
        parts = []
        for idx, item in enumerate(summaries):
            distance_from_recent = len(summaries) - idx
            if distance_from_recent == 1:
                parts.append(item.summary1000)
            elif distance_from_recent <= 3:
                parts.append(item.summary100)
            else:
                parts.append(item.summary10)
        summary_text = "\n".join(parts)
    else:
        raise ValueError(method)

    return f"Static memory summaries:\n{summary_text}\n\nRecent raw text:\n{recent_text}"


def build_learned_summary_examples(
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    config: Config,
) -> tuple[list[list[float]], list[float], dict[str, Any]]:
    blocks: list[Block] = []
    for ids in token_ids_by_dataset.values():
        train_ids = ids[: min(len(ids), config.learned_summary_train_tokens)]
        blocks.extend(build_blocks(tokenizer, train_ids, config.block_tokens))
    idf = idf_by_block(blocks)
    features: list[list[float]] = []
    labels: list[float] = []
    positives = 0
    for block in blocks:
        if not block.sentences:
            continue
        terms = top_terms(block, idf, config.summary10_words)
        ranked = sorted(
            range(len(block.sentences)),
            key=lambda idx: sentence_score(block.sentences[idx], terms, idf),
            reverse=True,
        )
        positive_count = max(1, min(6, math.ceil(0.18 * len(ranked))))
        positive_indices = set(ranked[:positive_count])
        for idx, sentence in enumerate(block.sentences):
            features.append(sentence_features(sentence, idx, len(block.sentences), terms, idf))
            label = 1.0 if idx in positive_indices else 0.0
            labels.append(label)
            positives += int(label)

    if len(features) > config.learned_summary_max_sentences:
        rng = random.Random(config.learned_summary_seed)
        selected = rng.sample(range(len(features)), config.learned_summary_max_sentences)
        features = [features[idx] for idx in selected]
        labels = [labels[idx] for idx in selected]

    metadata = {
        "blocks": len(blocks),
        "sentences": len(features),
        "positive_rate": sum(labels) / max(1, len(labels)),
        "pseudo_label": "top heuristic-ranked sentences per block",
    }
    return features, labels, metadata


def train_learned_summary_scorer(
    tokenizer: Any,
    token_ids_by_dataset: dict[str, list[int]],
    config: Config,
) -> LearnedSummaryScorer | None:
    if config.summary_backend != "learned":
        return None
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    features, labels, metadata = build_learned_summary_examples(tokenizer, token_ids_by_dataset, config)
    if not features:
        raise ValueError("no sentence examples were found for learned summarizer training")

    random.seed(config.learned_summary_seed)
    torch.manual_seed(config.learned_summary_seed)
    dim = len(features[0])
    columns = list(zip(*features))
    mean = tuple(float(statistics.mean(column)) for column in columns)
    std = tuple(float(statistics.pstdev(column) or 1.0) for column in columns)
    normalized = [
        [(value - mean[idx]) / max(std[idx], 1e-6) for idx, value in enumerate(row)]
        for row in features
    ]

    x = torch.tensor(normalized, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    model = nn.Sequential(
        nn.Linear(dim, config.learned_summary_hidden_dim),
        nn.ReLU(),
        nn.Linear(config.learned_summary_hidden_dim, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learned_summary_lr, weight_decay=1e-4)
    positive_rate = float(y.mean().item())
    pos_weight = torch.tensor([(1.0 - positive_rate) / max(positive_rate, 1e-4)], dtype=torch.float32)
    started = time.perf_counter()
    for _ in range(config.learned_summary_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        probs = torch.sigmoid(model(x))
        pred = (probs >= 0.5).float()
        accuracy = float((pred == y).float().mean().item())
        final_loss = float(F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight).item())
    metadata.update(
        {
            "feature_dim": dim,
            "hidden_dim": config.learned_summary_hidden_dim,
            "epochs": config.learned_summary_epochs,
            "lr": config.learned_summary_lr,
            "train_loss": final_loss,
            "train_accuracy": accuracy,
            "train_seconds": time.perf_counter() - started,
        }
    )
    model.eval()
    print(
        "learned summarizer: "
        f"sentences={metadata['sentences']} positive_rate={metadata['positive_rate']:.3f} "
        f"loss={metadata['train_loss']:.4f} acc={metadata['train_accuracy']:.3f} "
        f"seconds={metadata['train_seconds']:.1f}",
        flush=True,
    )
    return LearnedSummaryScorer(model=model, mean=mean, std=std, metadata=metadata)


def resolve_dtype(dtype_name: str, torch_module: Any) -> Any:
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[dtype_name]


def synchronize(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)


def score_target(model: Any, input_ids: Any, prompt_len: int, target_len: int) -> tuple[float, float]:
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits
    target_ids = input_ids[:, prompt_len : prompt_len + target_len]
    pred_logits = logits[:, prompt_len - 1 : prompt_len + target_len - 1, :]
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]).float(),
        target_ids.reshape(-1),
        reduction="mean",
    )
    nll = float(loss.detach().cpu())
    ppl = math.exp(min(nll, 80.0))
    return nll, ppl


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[(row.dataset, row.method)].append(row)

    summary: list[dict[str, Any]] = []
    full_by_dataset: dict[str, dict[str, float]] = {}
    for (dataset, method), items in grouped.items():
        if method != "full_raw":
            continue
        full_by_dataset[dataset] = {
            "tokens": statistics.mean(item.total_input_tokens for item in items),
            "seconds": statistics.mean(item.forward_seconds for item in items),
        }

    for (dataset, method), items in sorted(grouped.items()):
        total_eval = sum(item.eval_tokens for item in items)
        mean_nll = sum(item.nll * item.eval_tokens for item in items) / max(1, total_eval)
        avg_tokens = statistics.mean(item.total_input_tokens for item in items)
        avg_seconds = statistics.mean(item.forward_seconds for item in items)
        full = full_by_dataset.get(dataset, {"tokens": avg_tokens, "seconds": avg_seconds})
        summary.append(
            {
                "dataset": dataset,
                "method": method,
                "samples": len(items),
                "eval_tokens": total_eval,
                "mean_nll": mean_nll,
                "ppl": math.exp(min(mean_nll, 80.0)),
                "avg_prompt_tokens": statistics.mean(item.prompt_tokens for item in items),
                "avg_total_input_tokens": avg_tokens,
                "token_ratio_vs_full_raw": avg_tokens / full["tokens"] if full["tokens"] else 0.0,
                "avg_forward_seconds": avg_seconds,
                "time_ratio_vs_full_raw": avg_seconds / full["seconds"] if full["seconds"] else 0.0,
                "speedup_vs_full_raw": full["seconds"] / avg_seconds if avg_seconds > 0 else 0.0,
                "avg_tokens_per_second": statistics.mean(item.tokens_per_second for item in items),
            }
        )
    return summary


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": resolve_dtype(config.dtype, torch)}
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    token_ids_by_dataset: dict[str, list[int]] = {}
    for dataset_name, text_path in zip(config.dataset_names, config.text_paths):
        text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
        token_ids_by_dataset[dataset_name] = tokenizer(text, add_special_tokens=False)["input_ids"][: config.max_text_tokens]
    summary_scorer = train_learned_summary_scorer(tokenizer, token_ids_by_dataset, config)
    if summary_scorer is not None:
        torch.save(
            {
                "state_dict": summary_scorer.model.state_dict(),
                "mean": summary_scorer.mean,
                "std": summary_scorer.std,
                "metadata": summary_scorer.metadata,
            },
            output_dir / "learned_summary_scorer.pt",
        )
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.to(requested_device)
    model.eval()
    input_device = next(model.parameters()).device

    rows: list[TrialResult] = []
    for dataset_name, all_ids in token_ids_by_dataset.items():
        needed = config.prefill_tokens + config.eval_tokens
        max_start = max(0, len(all_ids) - needed)
        starts = [min(max_start, idx * config.sample_stride_tokens) for idx in range(config.samples_per_dataset)]
        for sample_id, start in enumerate(starts):
            prefix_ids = all_ids[start : start + config.prefill_tokens]
            target_ids = all_ids[start + config.prefill_tokens : start + config.prefill_tokens + config.eval_tokens]
            if len(prefix_ids) < config.prefill_tokens or len(target_ids) < config.eval_tokens:
                continue
            for method in config.methods:
                prompt_text = context_for_method(config, tokenizer, prefix_ids, method, summary_scorer=summary_scorer)
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                if not prompt_ids:
                    prompt_ids = [tokenizer.eos_token_id or 0]
                input_list = prompt_ids + target_ids
                input_tensor = torch.tensor(input_list, dtype=torch.long, device=input_device).view(1, -1)
                synchronize(torch, input_device)
                started = time.perf_counter()
                with torch.inference_mode():
                    nll, ppl = score_target(model, input_tensor, len(prompt_ids), len(target_ids))
                synchronize(torch, input_device)
                elapsed = time.perf_counter() - started
                total_tokens = int(input_tensor.shape[1])
                rows.append(
                    TrialResult(
                        dataset=dataset_name,
                        sample_id=sample_id,
                        method=method,
                        prompt_tokens=len(prompt_ids),
                        eval_tokens=len(target_ids),
                        total_input_tokens=total_tokens,
                        nll=nll,
                        ppl=ppl,
                        forward_seconds=elapsed,
                        tokens_per_second=total_tokens / elapsed if elapsed > 0 else 0.0,
                    )
                )
                del input_tensor

    summary = summarize(rows)
    write_csv(output_dir / "trials.csv", [asdict(row) for row in rows])
    write_csv(output_dir / "summary.csv", summary)
    payload = {
        "config": asdict(config),
        "summary_scorer": summary_scorer.metadata if summary_scorer is not None else None,
        "summary": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("dataset,method,samples,ppl,avg_total_input_tokens,avg_forward_seconds,speedup_vs_full_raw")
    for row in summary:
        print(
            f"{row['dataset']},{row['method']},{row['samples']},{row['ppl']:.4f},"
            f"{row['avg_total_input_tokens']:.1f},{row['avg_forward_seconds']:.4f},{row['speedup_vs_full_raw']:.3f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
