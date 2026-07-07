from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_seq_autoencoder_search_smoke import (  # noqa: E402
    LatentSearcher,
    SeqAutoencoder,
    attention,
    block_recall,
    block_scores_from_token_scores,
    mean_block_k,
    relative_mse,
    topk_recall,
)

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except Exception:  # pragma: no cover - depends on installed transformers version.
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception as exc:  # pragma: no cover
        apply_rotary_pos_emb = None
        _ROTARY_IMPORT_ERROR = exc
    else:
        _ROTARY_IMPORT_ERROR = None
else:
    _ROTARY_IMPORT_ERROR = None


@dataclass(frozen=True)
class Config:
    output_dir: str
    model_name_or_path: str
    prompt_tokens: int
    page_tokens: int
    recent_tokens: int
    cases: tuple[str, ...]
    layers: str
    kv_heads: str
    max_query_tokens: int
    block_size: int
    latent_dim: int
    train_fraction: float
    batch_size: int
    ae_epochs: int
    search_epochs: int
    lr: float
    rare_recon_weight: float
    rare_token_fraction: float
    block_search_weight: float
    topk_score_weight: float
    score_topk: int
    score_temperature: float
    dtype: str
    attn_implementation: str
    device_map: str
    local_files_only: bool
    seed: int


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    context_ids: list[int]
    query: str
    answer: str
    evidence_pages: tuple[int, ...]


@dataclass(frozen=True)
class SampleMeta:
    case: str
    layer: int
    kv_head: int


@dataclass(frozen=True)
class TraceDataset:
    k: torch.Tensor
    v: torch.Tensor
    q: torch.Tensor
    y: torch.Tensor
    meta: tuple[SampleMeta, ...]


@dataclass
class EvalRow:
    split: str
    group: str
    samples: int
    seq_len: int
    block_size: int
    blocks: int
    dim: int
    latent_dim: int
    latent_storage_ratio_vs_kv: float
    recon_k_relative_mse: float
    recon_v_relative_mse: float
    recon_attention_relative_mse: float
    recon_attention_cosine: float
    recon_token_top1_recall: float
    recon_token_top5_recall: float
    mean_block_top1_recall: float
    mean_block_top3_recall: float
    recon_block_top1_recall: float
    recon_block_top3_recall: float
    latent_block_top1_recall: float
    latent_block_top3_recall: float


@dataclass
class TrainStats:
    total_loss: float
    recon_loss: float
    block_loss: float
    topk_loss: float
    rare_loss: float
    search_loss: float
    train_seconds: float


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Collect real Qwen Q/K/V traces and test seq-autoencoder reconstructed-K search."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--prompt_tokens", type=int, default=1024)
    parser.add_argument("--page_tokens", type=int, default=128)
    parser.add_argument("--recent_tokens", type=int, default=128)
    parser.add_argument("--cases", type=parse_csv_tuple, default=("old_single", "two_old", "decoy_exact"))
    parser.add_argument("--layers", default="all")
    parser.add_argument("--kv_heads", default="all")
    parser.add_argument("--max_query_tokens", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--train_fraction", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ae_epochs", type=int, default=8)
    parser.add_argument("--search_epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--rare_recon_weight", type=float, default=0.2)
    parser.add_argument("--rare_token_fraction", type=float, default=0.01)
    parser.add_argument("--block_search_weight", type=float, default=0.0)
    parser.add_argument("--topk_score_weight", type=float, default=0.0)
    parser.add_argument("--score_topk", type=int, default=4)
    parser.add_argument("--score_temperature", type=float, default=2.0)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--local_files_only", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=2026070701)
    args = parser.parse_args()
    return Config(**vars(args))


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


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def pick_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_index_spec(spec: str, max_count: int, name: str) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(max_count))
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid {name} range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    bad = sorted(idx for idx in selected if idx < 0 or idx >= max_count)
    if bad:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {bad}")
    return sorted(selected)


def build_base_ids(tokenizer: Any, prompt_tokens: int) -> list[int]:
    filler = (
        "This neutral background passage fills the long context. It includes ordinary "
        "narrative details, irrelevant names, and no requested secret answer. "
    )
    filler_ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    while len(ids) < prompt_tokens:
        ids.extend(filler_ids)
    return ids[:prompt_tokens]


def replace_at(ids: list[int], start: int, replacement: list[int]) -> None:
    if start >= len(ids):
        return
    end = min(len(ids), start + len(replacement))
    ids[start:end] = replacement[: end - start]


def record_offset(page_tokens: int) -> int:
    return min(16, max(0, page_tokens // 2))


def variant_index(suffix: str) -> int:
    if not suffix:
        return 0
    if suffix.isdigit():
        return int(suffix)
    value = 0
    for char in suffix.lower():
        if not ("a" <= char <= "z"):
            continue
        value = value * 26 + (ord(char) - ord("a") + 1)
    return max(0, value - 1)


def split_case_family(name: str) -> tuple[str, int]:
    for family in (
        "old_single",
        "two_old",
        "decoy_exact",
        "needle_fact",
        "multi_hop_bridge",
        "current_conflict",
    ):
        if name == family:
            return family, 0
        prefix = family + "_"
        if name.startswith(prefix):
            return family, variant_index(name[len(prefix) :])
    raise ValueError(f"Unknown case: {name}")


def safe_page(preferred: int, pages: int) -> int:
    if pages <= 1:
        return 0
    return max(0, min(preferred, max(0, pages - 2)))


def distinct_pages(preferred_pages: list[int], pages: int) -> tuple[int, ...]:
    pool = list(range(1, max(1, pages - 1))) or [0]
    out: list[int] = []
    for preferred in preferred_pages:
        page = safe_page(preferred, pages)
        if page in out:
            candidates = sorted(pool, key=lambda item: (abs(item - preferred), item))
            for candidate in candidates:
                if candidate not in out:
                    page = candidate
                    break
        out.append(page)
    return tuple(out)


ANSWER_WORDS = [
    "ORBITAL-COPPER-284",
    "LUNAR-SILVER-391",
    "CANYON-QUARTZ-762",
    "VECTOR-AMBER-518",
    "SUMMIT-COBALT-044",
    "RIVER-ONYX-673",
    "GLACIER-BRONZE-925",
    "FOREST-PLATINUM-317",
    "ANCHOR-MARBLE-806",
    "CIRCUIT-GARNET-159",
    "HORIZON-ZINC-482",
    "VALLEY-TOPAZ-640",
]
OLD_PAGE_PLAN = [5, 3, 6, 2, 4, 1, 7, 5, 3, 6, 2, 4]
TWO_PAGE_PLAN = [
    (2, 4),
    (1, 5),
    (3, 6),
    (2, 5),
    (1, 4),
    (3, 5),
    (4, 7),
    (2, 6),
    (5, 7),
    (1, 6),
    (3, 7),
    (2, 7),
]
DECOY_PAGE_PLAN = [
    (2, 5),
    (1, 4),
    (3, 6),
    (2, 4),
    (4, 6),
    (1, 5),
    (3, 7),
    (2, 6),
    (5, 7),
    (1, 6),
    (4, 7),
    (2, 7),
]


def make_case(tokenizer: Any, name: str, prompt_tokens: int, page_tokens: int) -> SyntheticCase:
    ids = build_base_ids(tokenizer, prompt_tokens)
    pages = max(1, prompt_tokens // page_tokens)
    family, variant = split_case_family(name)

    if family == "old_single":
        key = f"REAL-TRACE-KEY-{7319 + 137 * variant}"
        answer = ANSWER_WORDS[variant % len(ANSWER_WORDS)]
        page = safe_page(OLD_PAGE_PLAN[variant % len(OLD_PAGE_PLAN)], pages)
        if variant % 3 == 0:
            record = (
                f"Verified old memory record. The key {key} maps to answer {answer}. "
                f"If asked about {key}, answer exactly {answer}."
            )
        elif variant % 3 == 1:
            record = (
                f"Archive fact. Lookup code {key} has final value {answer}. "
                f"Use this archive fact when the question names {key}."
            )
        else:
            record = (
                f"Long-range note: {key} resolves to {answer}. "
                f"The answer token for {key} is exactly {answer}."
            )
        replace_at(
            ids,
            page * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + record + "\n\n", add_special_tokens=False)["input_ids"],
        )
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: What is the answer for key {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page,),
        )
    if family == "needle_fact":
        key = f"PUBLIC-NEEDLE-{5107 + 173 * variant}"
        answer = ANSWER_WORDS[variant % len(ANSWER_WORDS)]
        page = safe_page(OLD_PAGE_PLAN[variant % len(OLD_PAGE_PLAN)], pages)
        if variant % 3 == 0:
            record = (
                f"Hidden audit line. Ticket {key} has checksum {answer}. "
                f"When the question asks for ticket {key}, return checksum {answer}."
            )
        elif variant % 3 == 1:
            record = (
                f"Needle record. The only valid checksum for ticket {key} is {answer}. "
                f"Do not infer from nearby background lines."
            )
        else:
            record = (
                f"Single-hop public benchmark fact: ticket {key} resolves to checksum {answer}. "
                f"The requested answer is exactly {answer}."
            )
        replace_at(
            ids,
            page * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + record + "\n\n", add_special_tokens=False)["input_ids"],
        )
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: What checksum is assigned to ticket {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page,),
        )
    if family == "two_old":
        key = f"BRIDGE-TRACE-{42 + 11 * variant}"
        node = f"NODE-TULIP-{17 + 5 * variant}"
        answer = [
            "HARBOR-SILVER-902",
            "PRAIRIE-GOLD-118",
            "CASCADE-IRON-530",
            "EMBER-GLASS-267",
            "MEADOW-NICKEL-814",
            "ISLAND-TIN-406",
            "FJORD-STEEL-731",
            "CITADEL-ALLOY-258",
            "ORCHARD-BRASS-609",
            "TUNDRA-CHROME-144",
            "TEMPLE-NICKEL-987",
            "CAVERN-TITAN-320",
        ][variant % 12]
        page_a, page_b = distinct_pages(list(TWO_PAGE_PLAN[variant % len(TWO_PAGE_PLAN)]), pages)
        if variant % 2 == 0:
            rec_a = f"Bridge old memory record. The key {key} points to intermediate token {node}."
            rec_b = f"Answer old memory record. The intermediate token {node} maps to final answer {answer}."
        else:
            rec_a = f"First hop record. Query id {key} should route to bridge marker {node}."
            rec_b = f"Second hop record. Bridge marker {node} gives final response {answer}."
        replace_at(
            ids,
            page_a * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_a + "\n\n", add_special_tokens=False)["input_ids"],
        )
        replace_at(
            ids,
            page_b * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_b + "\n\n", add_special_tokens=False)["input_ids"],
        )
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: For key {key}, follow the bridge and give the final answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page_a, page_b),
        )
    if family == "multi_hop_bridge":
        key = f"PUBLIC-HOP-{820 + 29 * variant}"
        node = f"LOCATOR-DELTA-{31 + 7 * variant}"
        answer = [
            "HARBOR-SILVER-902",
            "PRAIRIE-GOLD-118",
            "CASCADE-IRON-530",
            "EMBER-GLASS-267",
            "MEADOW-NICKEL-814",
            "ISLAND-TIN-406",
            "FJORD-STEEL-731",
            "CITADEL-ALLOY-258",
            "ORCHARD-BRASS-609",
            "TUNDRA-CHROME-144",
            "TEMPLE-NICKEL-987",
            "CAVERN-TITAN-320",
        ][variant % 12]
        page_a, page_b = distinct_pages(list(TWO_PAGE_PLAN[variant % len(TWO_PAGE_PLAN)]), pages)
        if variant % 2 == 0:
            rec_a = f"Route table entry. Request id {key} should be looked up through locator {node}."
            rec_b = f"Locator table entry. Locator {node} gives final checksum {answer}."
        else:
            rec_a = f"First clue. The public query id {key} links to bridge locator {node}."
            rec_b = f"Second clue. Bridge locator {node} resolves to exact answer {answer}."
        replace_at(
            ids,
            page_a * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_a + "\n\n", add_special_tokens=False)["input_ids"],
        )
        replace_at(
            ids,
            page_b * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_b + "\n\n", add_special_tokens=False)["input_ids"],
        )
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: Follow the locator for request id {key} and give the final checksum exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page_a, page_b),
        )
    if family == "decoy_exact":
        key = f"STATUS-TRACE-{8801 + 19 * variant}"
        decoy_base = [
            "EMBER-447",
            "PEARL-102",
            "IVORY-663",
            "JADE-219",
            "RUBY-705",
            "OPAL-334",
            "SAPPHIRE-918",
            "AMETHYST-276",
            "ONYX-551",
            "CORAL-803",
            "TOPAZ-146",
            "QUARTZ-729",
        ][variant % 12]
        old_answer = f"OBSOLETE-{decoy_base}"
        answer = f"VERIFIED-{decoy_base}"
        page_old, page_new = distinct_pages(list(DECOY_PAGE_PLAN[variant % len(DECOY_PAGE_PLAN)]), pages)
        if variant % 2 == 0:
            rec_old = f"Obsolete record. The key {key} used to map to answer {old_answer}, but this record is outdated."
            rec_new = f"Verified current record. The key {key} now maps to answer {answer}. Use the verified current record."
        else:
            rec_old = f"Superseded status line. The key {key} previously returned {old_answer}; ignore this stale value."
            rec_new = f"Current status line. The key {key} currently returns {answer}. Prefer current status over stale status."
        replace_at(
            ids,
            page_old * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_old + "\n\n", add_special_tokens=False)["input_ids"],
        )
    if family == "current_conflict":
        key = f"PUBLIC-CONFLICT-{3301 + 23 * variant}"
        decoy_base = [
            "EMBER-447",
            "PEARL-102",
            "IVORY-663",
            "JADE-219",
            "RUBY-705",
            "OPAL-334",
            "SAPPHIRE-918",
            "AMETHYST-276",
            "ONYX-551",
            "CORAL-803",
            "TOPAZ-146",
            "QUARTZ-729",
        ][variant % 12]
        old_answer = f"STALE-{decoy_base}"
        answer = f"CURRENT-{decoy_base}"
        page_old, page_new = distinct_pages(list(DECOY_PAGE_PLAN[variant % len(DECOY_PAGE_PLAN)]), pages)
        if variant % 2 == 0:
            rec_old = f"Old revision. Item {key} previously had checksum {old_answer}; this old revision is obsolete."
            rec_new = f"Current revision. Item {key} now has verified checksum {answer}. Use the current revision."
        else:
            rec_old = f"Conflicting stale line. Item {key} returned {old_answer} before the update; ignore stale lines."
            rec_new = f"Verified update line. Item {key} currently returns checksum {answer}."
        replace_at(
            ids,
            page_old * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_old + "\n\n", add_special_tokens=False)["input_ids"],
        )
        replace_at(
            ids,
            page_new * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_new + "\n\n", add_special_tokens=False)["input_ids"],
        )
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: What is the current verified checksum for item {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page_new,),
        )
        replace_at(
            ids,
            page_new * page_tokens + record_offset(page_tokens),
            tokenizer("\n\n" + rec_new + "\n\n", add_special_tokens=False)["input_ids"],
        )
        return SyntheticCase(
            name=name,
            context_ids=ids,
            query=f"Question: What is the verified current answer for key {key}? Answer exactly.\nAnswer:",
            answer=answer,
            evidence_pages=(page_new,),
        )
    raise ValueError(f"Unknown case: {name}")


def fixed_query_ids(tokenizer: Any, query: str, max_query_tokens: int) -> list[int]:
    ids = tokenizer(query, add_special_tokens=False)["input_ids"]
    if len(ids) >= max_query_tokens:
        return ids[:max_query_tokens]
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    return ids + [int(pad_id)] * (max_query_tokens - len(ids))


def model_core(model: torch.nn.Module) -> torch.nn.Module:
    for attr_name in ("model", "transformer"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


def identity_norm(module: Any, x: torch.Tensor) -> torch.Tensor:
    return module(x) if module is not None else x


def compute_layer_qkv(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    layer_idx: int,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if apply_rotary_pos_emb is None:
        raise ImportError(f"Could not import Qwen rotary helper: {_ROTARY_IMPORT_ERROR}")
    core = model_core(model)
    layer = core.layers[layer_idx]
    attn = layer.self_attn
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype
    x = hidden_states.to(device=device, dtype=dtype)
    with torch.no_grad():
        x = layer.input_layernorm(x)
        q = attn.q_proj(x).view(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(x.shape[0], x.shape[1], num_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(x.shape[0], x.shape[1], num_kv_heads, head_dim).transpose(1, 2)
        q = identity_norm(getattr(attn, "q_norm", None), q)
        k = identity_norm(getattr(attn, "k_norm", None), k)
        cos, sin = position_embeddings
        cos = cos.to(device=device, dtype=q.dtype)
        sin = sin.to(device=device, dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q[0].detach().float().cpu(), k[0].detach().float().cpu(), v[0].detach().float().cpu()


@torch.no_grad()
def collect_trace_dataset(model: Any, tokenizer: Any, config: Config) -> TraceDataset:
    core = model_core(model)
    num_layers = int(getattr(model.config, "num_hidden_layers"))
    num_heads = int(getattr(model.config, "num_attention_heads"))
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(model.config, "head_dim", getattr(model.config, "hidden_size") // num_heads))
    layer_indices = parse_index_spec(config.layers, num_layers, "layers")
    kv_head_indices = parse_index_spec(config.kv_heads, num_kv_heads, "kv_heads")
    q_per_kv = max(1, num_heads // num_kv_heads)
    input_device = pick_input_device(model)

    k_samples: list[torch.Tensor] = []
    v_samples: list[torch.Tensor] = []
    q_samples: list[torch.Tensor] = []
    y_samples: list[torch.Tensor] = []
    meta: list[SampleMeta] = []

    for case_name in config.cases:
        case = make_case(tokenizer, case_name, config.prompt_tokens, config.page_tokens)
        query_ids = fixed_query_ids(tokenizer, case.query, config.max_query_tokens)
        input_ids = torch.tensor([case.context_ids + query_ids], dtype=torch.long, device=input_device)
        context_len = len(case.context_ids)
        seq_len = int(input_ids.shape[1])
        if context_len % config.block_size != 0:
            usable = (context_len // config.block_size) * config.block_size
            if usable <= 0:
                raise ValueError("context length is shorter than one block")
            context_len = usable

        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_device).unsqueeze(0)
        position_embeddings = core.rotary_emb(hidden_states[0].to(input_device), position_ids)

        for layer_idx in layer_indices:
            q_all, k_all, v_all = compute_layer_qkv(
                model,
                hidden_states[layer_idx],
                layer_idx,
                position_embeddings,
                num_heads,
                num_kv_heads,
                head_dim,
            )
            for kv_head in kv_head_indices:
                q_start = kv_head * q_per_kv
                q_end = min(q_start + q_per_kv, q_all.shape[0])
                q_group = q_all[q_start:q_end, context_len:seq_len, :].reshape(-1, head_dim)
                if q_group.numel() == 0:
                    continue
                k_context = k_all[kv_head, :context_len, :].contiguous()
                v_context = v_all[kv_head, :context_len, :].contiguous()
                y_full = attention(q_group.unsqueeze(0), k_context.unsqueeze(0), v_context.unsqueeze(0))[0]
                k_samples.append(k_context)
                v_samples.append(v_context)
                q_samples.append(q_group)
                y_samples.append(y_full)
                meta.append(SampleMeta(case=case.name, layer=layer_idx, kv_head=kv_head))
            del q_all, k_all, v_all
        del outputs, hidden_states, input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not k_samples:
        raise RuntimeError("No trace samples were collected.")
    return TraceDataset(
        k=torch.stack(k_samples, dim=0),
        v=torch.stack(v_samples, dim=0),
        q=torch.stack(q_samples, dim=0),
        y=torch.stack(y_samples, dim=0),
        meta=tuple(meta),
    )


def block_scores(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    token_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    return block_scores_from_token_scores(token_scores, block_size)


def block_targets(q: torch.Tensor, k: torch.Tensor, block_size: int) -> torch.Tensor:
    return block_scores(q, k, block_size).argmax(dim=-1)


def rare_reconstruction_loss(
    k_recon: torch.Tensor,
    v_recon: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_fraction: float,
) -> torch.Tensor:
    _, seq_len, _ = k.shape
    count = max(1, min(seq_len, int(math.ceil(seq_len * token_fraction))))
    token_energy = k.float().norm(dim=-1) + v.float().norm(dim=-1)
    idx = token_energy.topk(count, dim=-1).indices
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, k.shape[-1])
    return (k_recon - k).pow(2).gather(1, gather_idx).mean() + (v_recon - v).pow(2).gather(1, gather_idx).mean()


def topk_score_distill_loss(student: torch.Tensor, teacher: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    topk = max(1, min(topk, teacher.shape[-1]))
    idx = teacher.topk(topk, dim=-1).indices
    teacher_top = teacher.gather(-1, idx) / max(temperature, 1e-6)
    student_top = student.gather(-1, idx) / max(temperature, 1e-6)
    teacher_top = teacher_top - teacher_top.mean(dim=-1, keepdim=True)
    student_top = student_top - student_top.mean(dim=-1, keepdim=True)
    return F.mse_loss(student_top, teacher_top.detach())


def split_indices(total: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = max(1, min(total - 1, int(round(total * train_fraction)))) if total > 1 else 1
    return indices[:train_count], indices[train_count:] or indices[:train_count]


def select_dataset(dataset: TraceDataset, indices: list[int]) -> TraceDataset:
    idx = torch.tensor(indices, dtype=torch.long)
    return TraceDataset(
        k=dataset.k.index_select(0, idx),
        v=dataset.v.index_select(0, idx),
        q=dataset.q.index_select(0, idx),
        y=dataset.y.index_select(0, idx),
        meta=tuple(dataset.meta[i] for i in indices),
    )


def train_models(train: TraceDataset, config: Config, device: torch.device) -> tuple[SeqAutoencoder, LatentSearcher, TrainStats]:
    model = SeqAutoencoder(train.k.shape[-1], config.block_size, config.latent_dim).to(device)
    searcher = LatentSearcher(train.k.shape[-1], config.latent_dim).to(device)
    params = list(model.parameters()) + list(searcher.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=1e-4)
    samples = train.k.shape[0]
    final_total = final_recon = final_block = final_topk = final_rare = 0.0
    start_wall = time.perf_counter()
    for _ in range(config.ae_epochs):
        order = torch.randperm(samples)
        for start in range(0, samples, config.batch_size):
            idx = order[start : start + config.batch_size]
            k = train.k.index_select(0, idx).to(device)
            v = train.v.index_select(0, idx).to(device)
            q = train.q.index_select(0, idx).to(device)
            z, k_recon, v_recon = model(k, v)
            recon_loss = F.mse_loss(k_recon, k) + F.mse_loss(v_recon, v)
            loss = recon_loss
            block_loss = torch.zeros((), device=device)
            topk_loss = torch.zeros((), device=device)
            rare_loss = torch.zeros((), device=device)
            if config.rare_recon_weight > 0:
                rare_loss = rare_reconstruction_loss(k_recon, v_recon, k, v, config.rare_token_fraction)
                loss = loss + config.rare_recon_weight * rare_loss
            scores = searcher(q, z)
            if config.block_search_weight > 0:
                target = block_targets(q, k, config.block_size)
                block_loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
                loss = loss + config.block_search_weight * block_loss
            if config.topk_score_weight > 0:
                teacher = block_scores(q, k, config.block_size)
                topk_loss = topk_score_distill_loss(scores, teacher, config.score_topk, config.score_temperature)
                loss = loss + config.topk_score_weight * topk_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_total = float(loss.detach().cpu())
            final_recon = float(recon_loss.detach().cpu())
            final_block = float(block_loss.detach().cpu())
            final_topk = float(topk_loss.detach().cpu())
            final_rare = float(rare_loss.detach().cpu())

    model.eval()
    for _ in range(config.search_epochs):
        order = torch.randperm(samples)
        for start in range(0, samples, config.batch_size):
            idx = order[start : start + config.batch_size]
            k = train.k.index_select(0, idx).to(device)
            v = train.v.index_select(0, idx).to(device)
            q = train.q.index_select(0, idx).to(device)
            with torch.no_grad():
                z = model.encode(k, v)
            target = block_targets(q, k, config.block_size)
            scores = searcher(q, z)
            search_loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            search_loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - start_wall
    return model, searcher, TrainStats(
        total_loss=final_total,
        recon_loss=final_recon,
        block_loss=final_block,
        topk_loss=final_topk,
        rare_loss=final_rare,
        search_loss=float(search_loss.detach().cpu()) if samples else 0.0,
        train_seconds=train_seconds,
    )


@torch.no_grad()
def eval_dataset(
    split: str,
    group: str,
    dataset: TraceDataset,
    model: SeqAutoencoder,
    searcher: LatentSearcher,
    config: Config,
    device: torch.device,
) -> EvalRow:
    k = dataset.k.to(device)
    v = dataset.v.to(device)
    q = dataset.q.to(device)
    y = dataset.y.to(device)
    z, k_recon, v_recon = model(k, v)
    y_recon = attention(q, k_recon, v_recon)
    full_token_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    recon_token_scores = torch.matmul(q, k_recon.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    full_block_scores = block_scores_from_token_scores(full_token_scores, config.block_size)
    recon_block_scores = block_scores_from_token_scores(recon_token_scores, config.block_size)
    mean_k = mean_block_k(k, config.block_size)
    mean_block_scores = torch.matmul(q, mean_k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    latent_block_scores = searcher(q, z)
    attn_cos = float(F.cosine_similarity(y_recon.reshape(-1, k.shape[-1]), y.reshape(-1, k.shape[-1]), dim=-1).mean().cpu())
    blocks = k.shape[1] // config.block_size
    return EvalRow(
        split=split,
        group=group,
        samples=int(k.shape[0]),
        seq_len=int(k.shape[1]),
        block_size=config.block_size,
        blocks=int(blocks),
        dim=int(k.shape[-1]),
        latent_dim=config.latent_dim,
        latent_storage_ratio_vs_kv=(blocks * config.latent_dim) / float(k.shape[1] * 2 * k.shape[-1]),
        recon_k_relative_mse=relative_mse(k_recon, k),
        recon_v_relative_mse=relative_mse(v_recon, v),
        recon_attention_relative_mse=relative_mse(y_recon, y),
        recon_attention_cosine=attn_cos,
        recon_token_top1_recall=topk_recall(full_token_scores, recon_token_scores, 1),
        recon_token_top5_recall=topk_recall(full_token_scores, recon_token_scores, 5),
        mean_block_top1_recall=block_recall(full_block_scores, mean_block_scores, 1),
        mean_block_top3_recall=block_recall(full_block_scores, mean_block_scores, 3),
        recon_block_top1_recall=block_recall(full_block_scores, recon_block_scores, 1),
        recon_block_top3_recall=block_recall(full_block_scores, recon_block_scores, 3),
        latent_block_top1_recall=block_recall(full_block_scores, latent_block_scores, 1),
        latent_block_top3_recall=block_recall(full_block_scores, latent_block_scores, 3),
    )


def grouped_eval_rows(
    split: str,
    dataset: TraceDataset,
    model: SeqAutoencoder,
    searcher: LatentSearcher,
    config: Config,
    device: torch.device,
) -> list[EvalRow]:
    rows = [eval_dataset(split, "overall", dataset, model, searcher, config, device)]
    by_case: dict[str, list[int]] = {}
    for idx, item in enumerate(dataset.meta):
        by_case.setdefault(f"case:{item.case}", []).append(idx)
    for group, indices in sorted(by_case.items()):
        if indices:
            rows.append(eval_dataset(split, group, select_dataset(dataset, indices), model, searcher, config, device))
    return rows


def main() -> None:
    config = parse_args()
    if config.prompt_tokens % config.block_size != 0:
        raise ValueError("--prompt_tokens must be divisible by --block_size")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(config.dtype)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "attn_implementation": config.attn_implementation,
        "local_files_only": config.local_files_only,
    }
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype
    if config.device_map and config.device_map.strip().lower() not in {"none", "null", "empty"}:
        load_kwargs["device_map"] = config.device_map
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    model.eval()

    start_wall = time.perf_counter()
    dataset = collect_trace_dataset(model, tokenizer, config)
    train_indices, test_indices = split_indices(dataset.k.shape[0], config.train_fraction, config.seed)
    train = select_dataset(dataset, train_indices)
    test = select_dataset(dataset, test_indices)
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, searcher, train_stats = train_models(train, config, train_device)
    rows = grouped_eval_rows("train", train, ae, searcher, config, train_device)
    rows.extend(grouped_eval_rows("test", test, ae, searcher, config, train_device))
    wall_seconds = time.perf_counter() - start_wall

    write_csv(output_dir / "real_qwen_seq_ae_search.csv", [asdict(row) for row in rows])
    payload = {
        "config": asdict(config),
        "trace_samples": int(dataset.k.shape[0]),
        "train_samples": int(train.k.shape[0]),
        "test_samples": int(test.k.shape[0]),
        "train_stats": asdict(train_stats),
        "wall_seconds": wall_seconds,
        "rows": [asdict(row) for row in rows],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("split,group,samples,storage_ratio,k_mse,v_mse,attn_mse,mean_b1,recon_b1,latent_b1")
    for row in rows:
        print(
            f"{row.split},{row.group},{row.samples},{row.latent_storage_ratio_vs_kv:.4f},"
            f"{row.recon_k_relative_mse:.6f},{row.recon_v_relative_mse:.6f},"
            f"{row.recon_attention_relative_mse:.6f},{row.mean_block_top1_recall:.4f},"
            f"{row.recon_block_top1_recall:.4f},{row.latent_block_top1_recall:.4f}"
        )


if __name__ == "__main__":
    main()
