from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import torch

import run_attention_confidence_sweep_8b as attention_runner
import run_local_rule_failure_boundary as base


# The bins are defined by an external English corpus proxy (wordfreq Zipf
# frequency).  WordNet synset IDs fix the intended sense.  Definitions are
# length-matched paraphrases of the WordNet glosses and never contain the query
# lemma, so success requires semantic matching rather than exact token lookup.
CONCEPTS: tuple[dict[str, Any], ...] = (
    {
        "concept_id": "horse.n.01",
        "bin": "common",
        "pair": "animal",
        "lemma": "horse",
        "definition": "a domesticated hoofed herbivore commonly used for riding and work",
        "query_clue": "a domesticated equine used for riding or work",
    },
    {
        "concept_id": "mongoose.n.01",
        "bin": "tail",
        "pair": "animal",
        "lemma": "mongoose",
        "definition": "an agile Old World mammal that preys on snakes and rodents",
        "query_clue": "a small carnivore famous for fighting venomous snakes",
    },
    {
        "concept_id": "sword.n.01",
        "bin": "common",
        "pair": "tool",
        "lemma": "sword",
        "definition": "a cutting or thrusting weapon with a long blade fixed to a hilt",
        "query_clue": "a long bladed weapon used for slashing or thrusting",
    },
    {
        "concept_id": "scraper.n.01",
        "bin": "tail",
        "pair": "tool",
        "lemma": "scraper",
        "definition": "a hand tool used to remove material by rubbing a sharp edge",
        "query_clue": "an implement for stripping paint or residue from a surface",
    },
    {
        "concept_id": "bottle.n.01",
        "bin": "common",
        "pair": "container",
        "lemma": "bottle",
        "definition": "a narrow-necked glass or plastic vessel used to store liquids",
        "query_clue": "a capped liquid holder with a narrow neck",
    },
    {
        "concept_id": "toolbox.n.01",
        "bin": "tail",
        "pair": "container",
        "lemma": "toolbox",
        "definition": "a box or chest designed to store and organize hand tools",
        "query_clue": "a portable case for keeping repair implements together",
    },
    {
        "concept_id": "salt.n.01",
        "bin": "common",
        "pair": "substance",
        "lemma": "salt",
        "definition": "an ionic compound formed when acid hydrogen is replaced by a metal",
        "query_clue": "the familiar ionic seasoning compound made of positive and negative ions",
    },
    {
        "concept_id": "gypsum.n.01",
        "bin": "tail",
        "pair": "substance",
        "lemma": "gypsum",
        "definition": "a soft hydrated calcium sulfate mineral used in plaster and cement",
        "query_clue": "a white mineral widely used to make plasterboard",
    },
)

LABELS = tuple("ABCDEFGH")

SEMANTIC_FILLER_PARAGRAPHS = (
    "Museum staff compared animal specimens, manufactured instruments, storage vessels, and mineral samples.",
    "Each catalog record described visible shape, material, habitat, dimensions, handling, and preservation status.",
    "A zoology note discussed body plans, feeding behavior, movement, breeding, and environmental adaptation.",
    "A workshop inventory grouped handheld implements by measurement, impact, cutting, fastening, and repair use.",
    "The collections guide classified household vessels by opening, capacity, material, closure, and ceremonial use.",
    "A geology worksheet compared clays, crystals, compounds, sediments, hardness, absorption, and chemical origin.",
    "Curators checked anonymous item numbers, shelf locations, condition reports, photographs, and acquisition dates.",
    "The educational text used broad category descriptions while omitting the names of individual catalog entries.",
)

SHORT_QUERY_MODES = ("lemma", "lemma_alt", "paraphrase")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def find_subsequence(values: Sequence[int], target: Sequence[int]) -> int:
    for start in range(len(values) - len(target) + 1):
        if list(values[start : start + len(target)]) == list(target):
            return start
    raise ValueError("marker token sequence not found")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chat_wrapper(tokenizer: Any) -> tuple[list[int], list[int]]:
    marker = "CODEX_SEMANTIC_NEEDLE_MARKER_20260722"
    kwargs = {
        "conversation": [{"role": "user", "content": marker}],
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        rendered = list(tokenizer.apply_chat_template(**kwargs, enable_thinking=False))
    except TypeError:
        rendered = list(tokenizer.apply_chat_template(**kwargs))
    marker_ids = encode(tokenizer, marker)
    start = find_subsequence(rendered, marker_ids)
    return rendered[:start], rendered[start + len(marker_ids) :]


def query_text(concept: dict[str, Any], mode: str) -> str:
    if mode == "lemma":
        target = f'the concept named "{concept["lemma"]}"'
        instruction = f"Find {target} in the catalog by meaning."
    elif mode == "lemma_alt":
        instruction = f'Which catalog description denotes "{concept["lemma"]}"?'
    elif mode == "paraphrase":
        instruction = f'Find the catalog entry matching this independent clue: "{concept["query_clue"]}".'
    else:
        raise ValueError(f"unknown query mode: {mode}")
    return (
        "\n\nRETRIEVAL QUESTION\n"
        + instruction
        + "\nReturn only its arbitrary label, with no explanation."
    )


def build_catalog(tokenizer: Any, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    order = list(CONCEPTS)
    rng.shuffle(order)
    rotated = list(LABELS)
    rng.shuffle(rotated)
    mapping = {concept["concept_id"]: rotated[index] for index, concept in enumerate(order)}

    token_ids = encode(
        tokenizer,
        "REFERENCE CATALOG\n"
        "Each entry maps a semantic description to an arbitrary one-letter label. "
        "Labels have no meaning and must be looked up.\n",
    )
    spans: dict[str, dict[str, Any]] = {}
    for index, concept in enumerate(order, start=1):
        entry_start = len(token_ids)
        token_ids.extend(encode(tokenizer, f"\nENTRY {index}\nDESCRIPTION:"))
        definition_start = len(token_ids)
        token_ids.extend(encode(tokenizer, " " + concept["definition"]))
        definition_end = len(token_ids)
        token_ids.extend(encode(tokenizer, "\nARBITRARY LABEL:"))
        label_start = len(token_ids)
        label_ids = encode(tokenizer, " " + mapping[concept["concept_id"]])
        if len(label_ids) != 1:
            raise AssertionError(f"label is not one token: {mapping[concept['concept_id']]} -> {label_ids}")
        token_ids.extend(label_ids)
        label_end = len(token_ids)
        token_ids.extend(encode(tokenizer, "\n"))
        spans[concept["concept_id"]] = {
            "entry": (entry_start, len(token_ids)),
            "definition": (definition_start, definition_end),
            "label_span": (label_start, label_end),
            "label_token_id": label_ids[0],
            "label": mapping[concept["concept_id"]],
        }
    return {"token_ids": token_ids, "spans": spans, "mapping": mapping, "order": [c["concept_id"] for c in order]}


def build_filler_ids(tokenizer: Any, target_length: int, seed: int, filler_type: str) -> list[int]:
    if filler_type == "plain":
        return base.build_filler_ids(tokenizer, target_length, seed)
    if filler_type != "semantic":
        raise ValueError(f"unknown filler type: {filler_type}")
    rng = random.Random(seed)
    paragraphs = list(SEMANTIC_FILLER_PARAGRAPHS)
    result: list[int] = []
    while len(result) < target_length + 512:
        rng.shuffle(paragraphs)
        result.extend(encode(tokenizer, "\n".join(paragraphs) + "\n"))
    return result[:target_length]


def validate_design(tokenizer: Any) -> dict[str, Any]:
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency

    rows: list[dict[str, Any]] = []
    seen = set()
    for concept in CONCEPTS:
        if concept["concept_id"] in seen:
            raise AssertionError(f"duplicate concept: {concept['concept_id']}")
        seen.add(concept["concept_id"])
        synset = wn.synset(concept["concept_id"])
        if concept["lemma"] not in {lemma.lower() for lemma in synset.lemma_names()}:
            raise AssertionError(f"lemma/synset mismatch: {concept['concept_id']}")
        if re.search(rf"\b{re.escape(concept['lemma'])}\b", concept["definition"], flags=re.I):
            raise AssertionError(f"definition leaks lemma: {concept['concept_id']}")
        query_token_ids = encode(tokenizer, " " + concept["lemma"])
        rows.append(
            {
                **concept,
                "zipf_frequency": float(zipf_frequency(concept["lemma"], "en")),
                "wordnet_gloss": synset.definition(),
                "wordnet_lexname": synset.lexname(),
                "lemma_token_ids": query_token_ids,
                "lemma_token_count": len(query_token_ids),
                "definition_token_count": len(encode(tokenizer, " " + concept["definition"])),
                "query_clue_token_count": len(encode(tokenizer, " " + concept["query_clue"])),
            }
        )
    for label in LABELS:
        if len(encode(tokenizer, " " + label)) != 1:
            raise AssertionError(f"label {label} is not a single leading-space token")
    all_filler = " ".join(base.FILLER_PARAGRAPHS + list(SEMANTIC_FILLER_PARAGRAPHS)).lower()
    leaked = [concept["lemma"] for concept in CONCEPTS if re.search(rf"\b{re.escape(concept['lemma'])}\b", all_filler)]
    if leaked:
        raise AssertionError(f"filler leaks query lemmas: {leaked}")
    return {"concepts": rows, "labels": list(LABELS), "filler_lemma_leak": leaked}


def place_catalog(
    tokenizer: Any,
    wrapper_prefix: Sequence[int],
    body_length: int,
    gap: int,
    filler_type: str,
    filler_seed: int,
    catalog: dict[str, Any],
) -> tuple[list[int], dict[str, dict[str, Any]], int]:
    filler = build_filler_ids(tokenizer, body_length, filler_seed, filler_type)
    block = list(catalog["token_ids"])
    start = body_length - gap - len(block)
    if start < 0:
        raise ValueError(f"catalog ({len(block)}) plus gap ({gap}) does not fit body ({body_length})")
    filler[start : start + len(block)] = block
    absolute_start = len(wrapper_prefix) + start
    spans: dict[str, dict[str, Any]] = {}
    for concept_id, relative in catalog["spans"].items():
        entry = tuple(absolute_start + value for value in relative["entry"])
        definition = tuple(absolute_start + value for value in relative["definition"])
        label_token_id = int(relative["label_token_id"])
        label_position = absolute_start + int(relative["label_span"][0])
        if filler[label_position - len(wrapper_prefix)] != label_token_id:
            raise AssertionError(f"cannot locate label token for {concept_id}")
        spans[concept_id] = {
            "entry": entry,
            "definition": definition,
            "label_span": (label_position, label_position + 1),
            "label_token_id": label_token_id,
            "label": relative["label"],
        }
    return list(wrapper_prefix) + filler, spans, absolute_start


def build_short_prefix(
    wrapper_prefix: Sequence[int], catalog: dict[str, Any]
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    absolute_start = len(wrapper_prefix)
    prefix = list(wrapper_prefix) + list(catalog["token_ids"])
    spans: dict[str, dict[str, Any]] = {}
    for concept_id, relative in catalog["spans"].items():
        entry = tuple(absolute_start + value for value in relative["entry"])
        definition = tuple(absolute_start + value for value in relative["definition"])
        label_token_id = int(relative["label_token_id"])
        label_position = absolute_start + int(relative["label_span"][0])
        if prefix[label_position] != label_token_id:
            raise AssertionError(f"cannot locate short label token for {concept_id}")
        spans[concept_id] = {
            "entry": entry,
            "definition": definition,
            "label_span": (label_position, label_position + 1),
            "label_token_id": label_token_id,
            "label": relative["label"],
        }
    return prefix, spans


def model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    max_position = args.body_length + 2048
    factor = base.rope_factor_for_length(max_position, args.original_max_position_embeddings)
    if factor > 1.0:
        config.max_position_embeddings = max_position
        config.rope_scaling = {
            "type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(args.original_max_position_embeddings),
        }
    load_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "attn_implementation": args.attn_implementation,
        "low_cpu_mem_usage": True,
        "device_map": {"": 0},
    }
    if args.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    metadata = {
        "load_in_8bit": bool(args.load_in_8bit),
        "torch_dtype": str(next(model.parameters()).dtype),
        "device_map": getattr(model, "hf_device_map", None),
        "rope_factor": factor,
        "max_position_embeddings": max_position,
    }
    return model, tokenizer, metadata


@torch.inference_mode()
def extend_cache(model: Any, cache: Any, token_ids: Sequence[int], start_position: int, chunk_size: int) -> float:
    device = model_input_device(model)
    started = time.perf_counter()
    past_len = start_position
    for start in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[start : start + chunk_size]
        chunk = torch.tensor(chunk_ids, dtype=torch.long, device=device).view(1, -1)
        output = base.forward_with_cache(model, chunk, cache, past_len)
        cache = output.past_key_values
        past_len += len(chunk_ids)
        del output, chunk
    base.synchronize()
    return time.perf_counter() - started


@torch.inference_mode()
def prefill_mutable(model: Any, token_ids: Sequence[int], chunk_size: int) -> tuple[Any, float]:
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    seconds = extend_cache(model, cache, token_ids, 0, chunk_size)
    if int(cache.get_seq_length()) != len(token_ids):
        raise AssertionError(f"cache length {cache.get_seq_length()} != prefix length {len(token_ids)}")
    return cache, seconds


def score_logits(tokenizer: Any, logits: torch.Tensor, gold_token_id: int, candidate_ids: Sequence[int]) -> dict[str, Any]:
    values = logits[0, -1].float()
    log_probs = torch.log_softmax(values, dim=-1)
    candidate = torch.tensor(candidate_ids, dtype=torch.long, device=values.device)
    candidate_values = log_probs[candidate]
    candidate_index = int(torch.argmax(candidate_values).item())
    gold_index = list(candidate_ids).index(gold_token_id)
    competing = torch.cat((candidate_values[:gold_index], candidate_values[gold_index + 1 :]))
    gold_logprob = float(log_probs[gold_token_id].item())
    greedy_id = int(torch.argmax(values).item())
    return {
        "gold_logprob": gold_logprob,
        "gold_probability": math.exp(gold_logprob),
        "gold_ppl": math.exp(-gold_logprob),
        "candidate_correct": candidate_index == gold_index,
        "candidate_prediction_token_id": int(candidate_ids[candidate_index]),
        "candidate_prediction": tokenizer.decode([int(candidate_ids[candidate_index])]).strip(),
        "candidate_margin": float(candidate_values[gold_index].item() - competing.max().item()),
        "greedy_correct": greedy_id == gold_token_id,
        "greedy_token_id": greedy_id,
        "greedy_token": tokenizer.decode([greedy_id], clean_up_tokenization_spaces=False),
    }


def span_lse(logits: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
    return torch.logsumexp(logits[:, span[0] : span[1]], dim=1)


@torch.inference_mode()
def summarize_attention(
    model: Any,
    output: Any,
    captured_queries: dict[int, torch.Tensor],
    relevant: dict[str, Any],
    catalog_span: tuple[int, int],
) -> dict[str, Any]:
    cache = base.legacy_cache(output.past_key_values)
    key_length = int(cache[0][0].shape[2])
    entry = tuple(relevant["entry"])
    definition = tuple(relevant["definition"])
    label_position = int(relevant["label_span"][0])
    top20_count = min(20, key_length)
    top2_count = max(1, int(math.ceil(0.02 * key_length)))
    head_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for layer_index, layer_cache in enumerate(cache):
        keys = layer_cache[0][0]
        queries = captured_queries[layer_index][0]
        q_heads = int(queries.shape[0])
        kv_heads = int(keys.shape[0])
        group_size = q_heads // kv_heads
        scale = float(model.model.layers[layer_index].self_attn.scaling)
        layer_heads: list[dict[str, Any]] = []
        for kv_index in range(kv_heads):
            first_head = kv_index * group_size
            q = queries[first_head : first_head + group_size].float()
            k = keys[kv_index].float()
            logits = torch.matmul(q, k.transpose(0, 1)) * scale
            probabilities = torch.softmax(logits, dim=1)
            evidence_lse = span_lse(logits, entry)
            masked = logits.clone()
            masked[:, entry[0] : entry[1]] = -torch.inf
            background_lse = torch.logsumexp(masked, dim=1)
            entry_mass = probabilities[:, entry[0] : entry[1]].sum(dim=1)
            definition_mass = probabilities[:, definition[0] : definition[1]].sum(dim=1)
            label_mass = probabilities[:, label_position]
            catalog_mass = probabilities[:, catalog_span[0] : catalog_span[1]].sum(dim=1)
            label_logits = logits[:, label_position]
            rank = 1 + (logits > label_logits.unsqueeze(1)).sum(dim=1)
            top20_indices = torch.topk(logits, k=top20_count, dim=1).indices
            top2_indices = torch.topk(logits, k=top2_count, dim=1).indices
            entry_top20 = ((top20_indices >= entry[0]) & (top20_indices < entry[1])).any(dim=1)
            entry_top2 = ((top2_indices >= entry[0]) & (top2_indices < entry[1])).any(dim=1)
            top20_mass = torch.topk(probabilities, k=top20_count, dim=1).values.sum(dim=1)
            for local_index in range(group_size):
                row = {
                    "layer": layer_index,
                    "head": first_head + local_index,
                    "entry_mass": float(entry_mass[local_index].item()),
                    "definition_mass": float(definition_mass[local_index].item()),
                    "label_mass": float(label_mass[local_index].item()),
                    "catalog_mass": float(catalog_mass[local_index].item()),
                    "evidence_logsumexp": float(evidence_lse[local_index].item()),
                    "background_logsumexp": float(background_lse[local_index].item()),
                    "needle_log_odds": float((evidence_lse[local_index] - background_lse[local_index]).item()),
                    "label_logit": float(label_logits[local_index].item()),
                    "label_rank_fraction": float(rank[local_index].item() / key_length),
                    "entry_hit_top20": float(entry_top20[local_index].item()),
                    "entry_hit_top2pct": float(entry_top2[local_index].item()),
                    "outside_top20_mass": float((1.0 - top20_mass[local_index]).item()),
                }
                layer_heads.append(row)
                head_rows.append(row)
        numeric = [key for key in layer_heads[0] if key not in {"layer", "head"}]
        layer_rows.append({"layer": layer_index, **{key: mean(row[key] for row in layer_heads) for key in numeric}})
    numeric = [key for key in head_rows[0] if key not in {"layer", "head"}]
    return {
        "key_length": key_length,
        "top20_count": top20_count,
        "top2_count": top2_count,
        "model_mean": {key: mean(row[key] for row in head_rows) for key in numeric},
        "layer_mean": layer_rows,
        "head_rows": head_rows,
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_branch(
    model: Any,
    tokenizer: Any,
    cache: Any,
    base_length: int,
    suffix_ids: Sequence[int],
    gold_token_id: int,
    candidate_ids: Sequence[int],
    chunk_size: int,
    relevant: dict[str, Any] | None,
    catalog_span: tuple[int, int] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, float]]:
    if len(suffix_ids) < 2:
        raise ValueError("query suffix must have at least two tokens")
    extend_seconds = extend_cache(model, cache, suffix_ids[:-1], base_length, chunk_size)
    prompt_len_minus_one = base_length + len(suffix_ids) - 1
    last = torch.tensor([suffix_ids[-1]], dtype=torch.long).view(1, 1)
    output, captured, query_seconds = attention_runner.capture_query_states(
        model, cache, last, prompt_len_minus_one
    )
    scores = score_logits(tokenizer, output.logits, gold_token_id, candidate_ids)
    attention = (
        summarize_attention(model, output, captured, relevant, catalog_span)
        if relevant is not None and catalog_span is not None
        else None
    )
    # DynamicCache is intentionally reused.  Cropping removes this query branch
    # without duplicating the 64K base cache, so all eight concepts share one
    # expensive prefill on a single 24 GB GPU.
    output.past_key_values.crop(base_length)
    if int(output.past_key_values.get_seq_length()) != base_length:
        raise AssertionError("failed to restore shared prefix cache")
    del output, captured, last
    release_cuda()
    return scores, attention, {"extend_seconds": extend_seconds, "query_seconds": query_seconds}


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    result = {
        "n": len(rows),
        "candidate_accuracy": mean(float(row["scores"]["candidate_correct"]) for row in rows),
        "greedy_accuracy": mean(float(row["scores"]["greedy_correct"]) for row in rows),
        "geometric_mean_ppl": math.exp(mean(math.log(float(row["scores"]["gold_ppl"])) for row in rows)),
    }
    with_attention = [row for row in rows if row.get("attention")]
    if with_attention:
        for key in (
            "entry_mass",
            "definition_mass",
            "label_mass",
            "catalog_mass",
            "needle_log_odds",
            "entry_hit_top20",
            "entry_hit_top2pct",
            "outside_top20_mass",
        ):
            result[key] = mean(float(row["attention"]["model_mean"][key]) for row in with_attention)
    return result


def write_report(output_dir: Path, audit: dict[str, Any], status: str) -> None:
    short_rows = read_jsonl(output_dir / "short_rows.jsonl")
    long_rows = read_jsonl(output_dir / "long_rows.jsonl")
    lines = [
        "# Semantic common-vs-tail needle pilot",
        "",
        f"Status: **{status}**",
        "",
        "## Concept audit",
        "",
        "| Pair | Bin | Synset | Lemma | Zipf | Lemma tokens | Definition tokens |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in audit["concepts"]:
        lines.append(
            f"| {row['pair']} | {row['bin']} | {row['concept_id']} | {row['lemma']} | "
            f"{row['zipf_frequency']:.2f} | {row['lemma_token_count']} | {row['definition_token_count']} |"
        )
    lines.extend(["", "## Short-context familiarity gate", ""])
    if short_rows:
        lines.extend([
            "| Bin | N | Candidate accuracy | Greedy accuracy | Geomean PPL |",
            "|---|---:|---:|---:|---:|",
        ])
        for bin_name in ("common", "tail"):
            summary = aggregate_rows([row for row in short_rows if row["bin"] == bin_name])
            lines.append(
                f"| {bin_name} | {summary['n']} | {summary['candidate_accuracy']:.2%} | "
                f"{summary['greedy_accuracy']:.2%} | {summary['geometric_mean_ppl']:.3f} |"
            )
        lines.append("")
        lines.append("Per-concept familiarity uses three independent query phrasings; the long pilot starts only if every concept reaches at least 90% candidate accuracy.")
    else:
        lines.append("Not started.")
    lines.extend(["", "## 64K pilot", ""])
    if long_rows:
        lines.extend([
            "| Filler | Gap | Query | Bin | N | Candidate acc. | Greedy acc. | PPL | Evidence mass | Top-20 head recall | Log-odds |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        keys = sorted({(row["filler_type"], row["gap"], row["query_mode"], row["bin"]) for row in long_rows})
        for filler_type, gap, query_mode, bin_name in keys:
            summary = aggregate_rows(
                [
                    row
                    for row in long_rows
                    if (row["filler_type"], row["gap"], row["query_mode"], row["bin"])
                    == (filler_type, gap, query_mode, bin_name)
                ]
            )
            lines.append(
                f"| {filler_type} | {gap} | {query_mode} | {bin_name} | {summary['n']} | "
                f"{summary['candidate_accuracy']:.2%} | {summary['greedy_accuracy']:.2%} | "
                f"{summary['geometric_mean_ppl']:.3f} | {summary['entry_mass']:.6f} | "
                f"{summary['entry_hit_top20']:.2%} | {summary['needle_log_odds']:.3f} |"
            )
    else:
        lines.append("Waiting for the familiarity gate.")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This is a calibration pilot with four concepts per bin, not the final significance test.",
            "- Bin assignment uses public-corpus lemma frequency as a proxy for training exposure; the Qwen training mixture is unknown.",
            "- The answer labels are arbitrary one-token A-H codes, and the evidence definitions do not contain the queried lemmas.",
            "- Candidate accuracy asks whether the gold label is best among A-H. Greedy accuracy asks whether it is best in the full vocabulary.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efficient semantic common-vs-tail 64K pilot on one GPU.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--body_length", type=int, default=65536)
    parser.add_argument("--gaps", default="4096,16384")
    parser.add_argument("--filler_types", default="plain,semantic")
    parser.add_argument("--query_modes", default="lemma,paraphrase")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    audit = validate_design(tokenizer)
    wrapper_prefix, wrapper_suffix = chat_wrapper(tokenizer)
    gaps = parse_csv_ints(args.gaps)
    filler_types = parse_csv_strs(args.filler_types)
    query_modes = parse_csv_strs(args.query_modes)
    catalog = build_catalog(tokenizer, args.seed)
    design = {
        "definition": "WordNet-sense-controlled semantic lookup with public-corpus lemma Zipf bins and arbitrary A-H answer labels",
        "audit": audit,
        "catalog_mapping": catalog["mapping"],
        "catalog_order": catalog["order"],
        "body_length": args.body_length,
        "gaps": gaps,
        "filler_types": filler_types,
        "query_modes": query_modes,
        "short_query_modes": list(SHORT_QUERY_MODES),
        "short_gate": "every concept candidate accuracy >= 0.90 across three query phrasings",
        "long_cases": len(CONCEPTS) * len(gaps) * len(filler_types) * len(query_modes),
        "shared_prefills": len(gaps) * len(filler_types),
        "seed": args.seed,
    }
    (output_dir / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(design, ensure_ascii=False, indent=2))
        return
    if args.summarize_only:
        write_report(output_dir, audit, "partial summary")
        return

    (output_dir / "running").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")
    model, tokenizer, model_metadata = load_model(args)
    (output_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    candidate_ids = [encode(tokenizer, " " + label)[0] for label in LABELS]

    short_path = output_dir / "short_rows.jsonl"
    short_completed = {row["case_id"] for row in read_jsonl(short_path)}
    short_prefix, short_spans = build_short_prefix(wrapper_prefix, catalog)
    short_cache, short_prefill_seconds = prefill_mutable(model, short_prefix, args.prefill_chunk_size)
    short_base_length = len(short_prefix)
    for concept in CONCEPTS:
        relevant = short_spans[concept["concept_id"]]
        for mode in SHORT_QUERY_MODES:
            case_id = f"short_{concept['concept_id']}_{mode}"
            if case_id in short_completed:
                continue
            suffix = encode(tokenizer, query_text(concept, mode)) + list(wrapper_suffix) + encode(tokenizer, "LABEL:")
            scores, _, timing = run_branch(
                model,
                tokenizer,
                short_cache,
                short_base_length,
                suffix,
                int(relevant["label_token_id"]),
                candidate_ids,
                args.prefill_chunk_size,
                None,
                None,
            )
            row = {
                "case_id": case_id,
                "stage": "short_familiarity",
                "concept_id": concept["concept_id"],
                "lemma": concept["lemma"],
                "bin": concept["bin"],
                "pair": concept["pair"],
                "query_mode": mode,
                "gold_label": relevant["label"],
                "scores": scores,
                "timing": {**timing, "shared_prefill_seconds": short_prefill_seconds},
            }
            append_jsonl(short_path, row)
            print(json.dumps({"stage": "short", "case_id": case_id, "candidate_correct": scores["candidate_correct"], "greedy_correct": scores["greedy_correct"]}), flush=True)
    del short_cache
    release_cuda()

    short_rows = read_jsonl(short_path)
    familiarity: dict[str, float] = {}
    for concept in CONCEPTS:
        rows = [row for row in short_rows if row["concept_id"] == concept["concept_id"]]
        familiarity[concept["concept_id"]] = mean(float(row["scores"]["candidate_correct"]) for row in rows)
    (output_dir / "familiarity.json").write_text(json.dumps(familiarity, indent=2), encoding="utf-8")
    write_report(output_dir, audit, "short gate complete")
    failed = {concept_id: value for concept_id, value in familiarity.items() if value < 0.90}
    if failed:
        (output_dir / "calibration_failed").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        (output_dir / "running").unlink(missing_ok=True)
        raise RuntimeError(f"short-context familiarity gate failed: {failed}")
    (output_dir / "short_gate_passed").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")

    long_path = output_dir / "long_rows.jsonl"
    completed = {row["case_id"] for row in read_jsonl(long_path)}
    for filler_type in filler_types:
        for gap in gaps:
            prefix, spans, catalog_start = place_catalog(
                tokenizer,
                wrapper_prefix,
                args.body_length,
                gap,
                filler_type,
                args.seed + (0 if filler_type == "plain" else 1000),
                catalog,
            )
            catalog_span = (catalog_start, catalog_start + len(catalog["token_ids"]))
            cache, prefill_seconds = prefill_mutable(model, prefix, args.prefill_chunk_size)
            base_length = len(prefix)
            print(json.dumps({"stage": "prefill", "filler_type": filler_type, "gap": gap, "seconds": round(prefill_seconds, 2), "tokens": base_length}), flush=True)
            for query_mode in query_modes:
                for concept in CONCEPTS:
                    case_id = f"n{args.body_length}_{filler_type}_g{gap}_{query_mode}_{concept['concept_id']}"
                    if case_id in completed:
                        continue
                    relevant = spans[concept["concept_id"]]
                    suffix = (
                        encode(tokenizer, query_text(concept, query_mode))
                        + list(wrapper_suffix)
                        + encode(tokenizer, "LABEL:")
                    )
                    started = time.perf_counter()
                    scores, attention, timing = run_branch(
                        model,
                        tokenizer,
                        cache,
                        base_length,
                        suffix,
                        int(relevant["label_token_id"]),
                        candidate_ids,
                        args.prefill_chunk_size,
                        relevant,
                        catalog_span,
                    )
                    row = {
                        "case_id": case_id,
                        "stage": "long_pilot",
                        "concept_id": concept["concept_id"],
                        "lemma": concept["lemma"],
                        "bin": concept["bin"],
                        "pair": concept["pair"],
                        "filler_type": filler_type,
                        "gap": gap,
                        "query_mode": query_mode,
                        "body_length": args.body_length,
                        "prompt_tokens": base_length + len(suffix),
                        "catalog_start": catalog_start,
                        "catalog_tokens": len(catalog["token_ids"]),
                        "gold_label": relevant["label"],
                        "scores": scores,
                        "attention": attention,
                        "timing": {
                            **timing,
                            "shared_prefill_seconds": prefill_seconds,
                            "case_seconds": time.perf_counter() - started,
                        },
                    }
                    append_jsonl(long_path, row)
                    print(
                        json.dumps(
                            {
                                "stage": "long",
                                "case_id": case_id,
                                "candidate_correct": scores["candidate_correct"],
                                "greedy_correct": scores["greedy_correct"],
                                "gold_ppl": round(scores["gold_ppl"], 4),
                                "entry_mass": round(attention["model_mean"]["entry_mass"], 8),
                                "seconds": round(row["timing"]["case_seconds"], 2),
                            }
                        ),
                        flush=True,
                    )
            del cache
            release_cuda()
            write_report(output_dir, audit, "running long pilot")

    write_report(output_dir, audit, "complete")
    (output_dir / "running").unlink(missing_ok=True)
    (output_dir / "done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")


if __name__ == "__main__":
    main()
