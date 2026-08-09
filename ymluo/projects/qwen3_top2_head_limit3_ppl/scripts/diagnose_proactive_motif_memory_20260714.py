from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_multitopic_lpcm_ppl_20260714 import (  # noqa: E402
    TOPICS,
    AutoTokenizer,
    encode_topic_stream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank known recurrence sources with history-only features.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--results_json", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--dataset_cache_dir", default="/home/fdong/ymluo/datasets/sklearn")
    parser.add_argument("--history_tokens", type=int, default=32_000)
    parser.add_argument("--query_tokens", type=int, default=256)
    parser.add_argument("--eval_tokens", type=int, default=256)
    parser.add_argument("--windows_per_topic", type=int, default=3)
    parser.add_argument("--window_stride_tokens", type=int, default=32_512)
    parser.add_argument("--span_tokens", type=int, default=480)
    parser.add_argument("--span_stride_tokens", type=int, default=16)
    parser.add_argument("--ngram_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def rank_desc(values: list[float], target_index: int) -> int:
    target = values[target_index]
    return 1 + sum(value > target for value in values)


def history_span_features(
    remote_ids: list[int],
    tokenizer: Any,
    span_tokens: int,
    stride_tokens: int,
    ngram_tokens: int,
) -> list[dict[str, Any]]:
    token_counts = Counter(remote_ids)
    ngrams = [tuple(remote_ids[i : i + ngram_tokens]) for i in range(len(remote_ids) - ngram_tokens + 1)]
    ngram_counts = Counter(ngrams)
    spans: list[dict[str, Any]] = []
    max_start = max(0, len(remote_ids) - span_tokens)
    for start in range(0, max_start + 1, stride_tokens):
        end = min(len(remote_ids), start + span_tokens)
        ids = remote_ids[start:end]
        rarity = sum(math.log((len(remote_ids) + 1) / token_counts[token]) for token in ids) / len(ids)
        local_ngrams = [
            tuple(remote_ids[i : i + ngram_tokens])
            for i in range(start, max(start, end - ngram_tokens + 1))
        ]
        recurrence = (
            sum(math.log1p(ngram_counts[gram] - 1) for gram in local_ngrams) / len(local_ngrams)
            if local_ngrams
            else 0.0
        )
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        template_markers = len(re.findall(r"(?:[-_=*]{3,}|\b(?:subject|writes|article)\b|@)", text.lower()))
        spans.append(
            {
                "start": start,
                "end": end,
                "rarity": rarity,
                "recurrence": recurrence,
                "template_markers": float(template_markers),
            }
        )
    for feature in ("rarity", "recurrence", "template_markers"):
        normalized = minmax([span[feature] for span in spans])
        for span, value in zip(spans, normalized):
            span[f"{feature}_normalized"] = value
    for span in spans:
        span["motif_score"] = (
            0.45 * span["rarity_normalized"]
            + 0.35 * span["recurrence_normalized"]
            + 0.20 * span["template_markers_normalized"]
        )
    return spans


def source_span_index(spans: list[dict[str, Any]], source_start: int) -> int:
    containing = [
        (index, abs(span["start"] - source_start))
        for index, span in enumerate(spans)
        if span["start"] <= source_start < span["end"]
    ]
    if not containing:
        raise RuntimeError(f"No candidate span contains source token {source_start}")
    return min(containing, key=lambda item: item[1])[0]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    rows = json.loads(args.results_json.read_text(encoding="utf-8"))
    controller_rows = [row for row in rows if row["method"] == "universal_controller"]
    required_tokens = (
        (args.windows_per_topic - 1) * args.window_stride_tokens
        + args.history_tokens
        + args.eval_tokens
    )
    streams: dict[str, list[int]] = {}
    output: list[dict[str, Any]] = []
    for row in controller_rows:
        rebuilds = [trace for trace in row.get("echo_matches", []) if trace.get("cache_rebuilt")]
        if not rebuilds:
            continue
        topic = row["topic"]
        if topic not in streams:
            streams[topic] = encode_topic_stream(
                tokenizer,
                TOPICS[topic],
                required_tokens,
                args.dataset_cache_dir,
                args.seed,
            )
        start = row["window"] * args.window_stride_tokens
        history = streams[topic][start : start + args.history_tokens]
        remote_ids = history[: -args.query_tokens]
        spans = history_span_features(
            remote_ids,
            tokenizer,
            args.span_tokens,
            args.span_stride_tokens,
            args.ngram_tokens,
        )
        for trace in rebuilds:
            source_start = int(trace["remote_match_start"])
            source_index = source_span_index(spans, source_start)
            source = spans[source_index]
            feature_ranks = {
                feature: rank_desc([span[feature] for span in spans], source_index)
                for feature in ("rarity", "recurrence", "template_markers", "motif_score")
            }
            output.append(
                {
                    "topic": topic,
                    "window": row["window"],
                    "source_start": source_start,
                    "trigger_target_start": trace["target_start"],
                    "candidate_spans": len(spans),
                    "source_span": source,
                    "feature_ranks": feature_ranks,
                    "top_motif_spans": sorted(spans, key=lambda span: span["motif_score"], reverse=True)[:10],
                }
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
