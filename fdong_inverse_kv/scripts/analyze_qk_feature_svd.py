#!/usr/bin/env python3
"""Offline QK feature-source analysis for Qwen-style attention.

This script answers one concrete question:

For strict-history top attention keys of each layer/head/query, are the high QK
scores explained by hidden similarity, K-space similarity, or a small number of
full-WQ/WK singular feature pairs after slicing U_Q/U_K by head output coords?
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
except Exception as exc:  # pragma: no cover - only used on incompatible installs.
    apply_rotary_pos_emb = None
    repeat_kv = None
    _QWEN3_IMPORT_ERROR = exc
else:
    _QWEN3_IMPORT_ERROR = None


TEXT_KEYS = ("text", "content", "document", "raw_content")


class MeanStats:
    def __init__(self) -> None:
        self.sum: Dict[str, float] = defaultdict(float)
        self.count: Dict[str, int] = defaultdict(int)

    def add(self, name: str, value: float) -> None:
        if math.isfinite(value):
            self.sum[name] += float(value)
            self.count[name] += 1

    def mean(self, name: str) -> Optional[float]:
        if self.count[name] == 0:
            return None
        return self.sum[name] / self.count[name]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {name: self.mean(name) for name in sorted(self.sum)}


def str_to_bool(value: str) -> bool:
    normalized = str(value).lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze top-attention QK feature sources with SVD.")
    parser.add_argument("--model_dir", type=str, default="../../../Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default="../../../dclm/global-shard_01_of_10")
    parser.add_argument(
        "--input_source",
        choices=["synthetic_long_qa", "dclm", "text_file"],
        default="synthetic_long_qa",
    )
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="../outputs/qk_feature_svd")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--svd_device", choices=["cpu", "model"], default="cpu")
    parser.add_argument("--svd_method", choices=["exact", "randomized"], default="exact")
    parser.add_argument("--svd_oversample", type=int, default=32)
    parser.add_argument("--svd_power_iters", type=int, default=2)
    parser.add_argument("--token_start", type=int, default=5000)
    parser.add_argument("--num_query_tokens", type=int, default=100)
    parser.add_argument("--extra_tokens", type=int, default=8)
    parser.add_argument("--top_ratio", type=float, default=0.02)
    parser.add_argument("--layers", type=str, default="all", help="'all' or comma-separated layer ids.")
    parser.add_argument("--heads", type=str, default="all", help="'all' or comma-separated query-head ids.")
    parser.add_argument("--rope_ablation", type=str_to_bool, default=True)
    parser.add_argument(
        "--full_k_direction_layers",
        type=str,
        default="",
        help="Comma-separated layers for exact full-rank K direction attribution; empty disables it.",
    )
    parser.add_argument(
        "--full_k_keys_per_query",
        type=int,
        default=8,
        help="Top and random keys per query used by full-rank K direction attribution.",
    )
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--add_eos_between_docs", type=str_to_bool, default=True)
    parser.add_argument("--random_negatives_per_query", type=int, default=8)
    parser.add_argument("--distance_negatives_per_query", type=int, default=8)
    parser.add_argument(
        "--svd_keys_per_query",
        type=int,
        default=4,
        help="Number of top-attention key tokens per query used for SVD attribution. <=0 means all top-ratio keys.",
    )
    parser.add_argument(
        "--tail_svd_keys_per_query",
        type=int,
        default=4,
        help=(
            "Number of tail key tokens per query used for SVD attribution. "
            "<=0 means match the number of selected top keys."
        ),
    )
    parser.add_argument(
        "--tail_sample_mode",
        choices=["random_tail", "low_score"],
        default="low_score",
        help="How to sample non-top keys for SVD contribution comparison.",
    )
    parser.add_argument("--band_mode", choices=["equal_energy", "fixed"], default="equal_energy")
    parser.add_argument("--num_energy_bands", type=int, default=8)
    parser.add_argument(
        "--svd_rank_limit",
        type=int,
        default=256,
        help="Use only the first N singular directions for pair contribution attribution. <=0 means full rank.",
    )
    parser.add_argument(
        "--fixed_energy_edges",
        type=str,
        default="0,0.01,0.05,0.1,0.2,0.4,0.7,1.0",
        help="Cumulative singular-value energy edges for --band_mode fixed.",
    )
    parser.add_argument("--max_examples", type=int, default=256)
    parser.add_argument("--example_top_pairs", type=int, default=16)
    parser.add_argument("--example_rank_limit", type=int, default=256)
    parser.add_argument("--save_input_tokens", type=str_to_bool, default=True)
    parser.add_argument(
        "--center_keys",
        type=str_to_bool,
        default=True,
        help=(
            "Measure causal-mean-centered hidden/K cosine, verify the exact "
            "post-RoPE key-centering invariance, and recompute linear-K spectral attribution."
        ),
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def iter_input_files(data_dir: str, max_files: int) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(data_dir):
        for name in names:
            if name.endswith((".txt", ".jsonl")):
                files.append(os.path.join(root, name))
    files.sort()
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .txt or .jsonl files found under {data_dir}")
    return files


def extract_text(line: str) -> str:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return line
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in TEXT_KEYS:
            value = record.get(key)
            if isinstance(value, str):
                return value
    return ""


def collect_token_stream(
    data_dir: str,
    tokenizer,
    required_tokens: int,
    *,
    max_files: int,
    add_eos_between_docs: bool,
) -> Tuple[List[int], Dict[str, object]]:
    token_ids: List[int] = []
    docs = 0
    files_used = 0
    eos = tokenizer.eos_token_id
    for path in iter_input_files(data_dir, max_files=max_files):
        files_used += 1
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                text = extract_text(raw)
                if not text:
                    continue
                ids = tokenizer(text, add_special_tokens=False).input_ids
                if not ids:
                    continue
                token_ids.extend(ids)
                if add_eos_between_docs and eos is not None:
                    token_ids.append(int(eos))
                docs += 1
                if len(token_ids) >= required_tokens:
                    return token_ids[:required_tokens], {
                        "docs_used": docs,
                        "files_used": files_used,
                        "required_tokens": required_tokens,
                    }
    raise RuntimeError(f"Only collected {len(token_ids)} tokens; required {required_tokens}.")


def synthetic_fact_bank() -> List[Tuple[str, str, str]]:
    return [
        ("RAVEN-17", "the botanist Mira stored the blue compass inside the brass tea tin", "blue compass"),
        ("ORCHID-42", "the winter archive password is the phrase silver rain", "silver rain"),
        ("HARBOR-09", "the missing treaty was copied onto a green ceramic tile", "green ceramic tile"),
        ("LANTERN-31", "Professor Ilya changed the delivery city from Bergen to Valencia", "Valencia"),
        ("FALCON-58", "the safe opens only after the second bell and the word ember", "ember"),
        ("CIRRUS-76", "the medical sample marked C7 must be kept below minus eighteen degrees", "minus eighteen degrees"),
        ("ONYX-24", "the old radio frequency is 143.7 kilohertz", "143.7 kilohertz"),
        ("MAPLE-63", "the witness used the alias Nora Vale in the hotel ledger", "Nora Vale"),
        ("DELTA-88", "the expedition buried the spare battery under the west stair", "west stair"),
        ("IVORY-12", "the correct invoice total is 7314 credits", "7314 credits"),
    ]


def distractor_paragraph(index: int) -> str:
    colors = ["red", "violet", "amber", "white", "black", "green", "blue", "silver"]
    places = ["north dock", "library annex", "market road", "east tower", "glass station", "river gate"]
    objects = ["ledger", "sample box", "folded map", "metal key", "weather note", "ticket stub"]
    color = colors[index % len(colors)]
    place = places[(index * 3) % len(places)]
    obj = objects[(index * 5) % len(objects)]
    return (
        f"Distractor record {index:04d}. The {color} {obj} was moved near the {place}. "
        f"This paragraph is intentionally similar to other records, but it is not one of the target facts. "
        f"It mentions dates, names, containers, and locations so that lexical overlap alone is unreliable. "
        f"The local note number is {10000 + index}, and the clerk initials are {chr(65 + index % 26)}{chr(65 + (index * 7) % 26)}.\n"
    )


def build_synthetic_long_qa_text(tokenizer, required_tokens: int, token_start: int, seed: int) -> Tuple[str, Dict[str, object]]:
    """Create a long prompt whose late question depends on early buried facts."""
    rng = random.Random(seed)
    facts = synthetic_fact_bank()
    rng.shuffle(facts)
    lines = [
        "You are reading a long investigation notebook. Several target facts appear early. "
        "Much later, a question asks about those facts. Keep exact identifiers, objects, names, and numbers.\n",
        "BEGIN BURIED FACTS\n",
    ]
    for code, statement, answer in facts:
        lines.append(f"Target fact {code}: {statement}. The short answer for {code} is {answer}.\n")
    lines.append("END BURIED FACTS\n\n")

    distractor_idx = 0
    while len(tokenizer("".join(lines), add_special_tokens=False).input_ids) < max(0, token_start - 192):
        lines.append(distractor_paragraph(distractor_idx))
        distractor_idx += 1

    selected = facts[:5]
    lines.append("\nFINAL LONG QUESTION BLOCK\n")
    lines.append(
        "Use only the buried target facts above. Ignore the distractor records unless they repeat the exact target identifier. "
        "For each requested identifier, recover the short answer and then explain which early record supports it.\n"
    )
    for code, _, _ in selected:
        lines.append(f"Question item for {code}: what is the exact short answer stored in the buried fact?\n")
    lines.append(
        "Now produce the answers in order, with identifiers preserved. The first requested identifier is "
        f"{selected[0][0]}, then {selected[1][0]}, then {selected[2][0]}, then {selected[3][0]}, then {selected[4][0]}.\n"
    )

    while len(tokenizer("".join(lines), add_special_tokens=False).input_ids) < required_tokens:
        lines.append(
            "Reasoning reminder: the answer must come from the early buried facts, not from nearby distractors. "
            f"Repeat target order: {', '.join(code for code, _, _ in selected)}.\n"
        )

    text = "".join(lines)
    meta = {
        "source": "synthetic_long_qa",
        "facts": [{"code": code, "statement": statement, "answer": answer} for code, statement, answer in facts],
        "queried_codes": [code for code, _, _ in selected],
        "distractor_paragraphs": distractor_idx,
    }
    return text, meta


def collect_analysis_tokens(args, tokenizer, required_tokens: int) -> Tuple[List[int], Dict[str, object], Optional[str]]:
    if args.input_source == "dclm":
        tokens, meta = collect_token_stream(
            args.data_dir,
            tokenizer,
            required_tokens,
            max_files=args.max_files,
            add_eos_between_docs=args.add_eos_between_docs,
        )
        return tokens, meta, None

    if args.input_source == "text_file":
        if not args.prompt_file:
            raise ValueError("--prompt_file is required when --input_source text_file")
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(token_ids) < required_tokens:
            raise RuntimeError(f"Prompt file has {len(token_ids)} tokens; required {required_tokens}.")
        return token_ids[:required_tokens], {"source": "text_file", "prompt_file": args.prompt_file}, text

    text, meta = build_synthetic_long_qa_text(tokenizer, required_tokens, args.token_start, args.seed)
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    return token_ids[:required_tokens], meta, text


def parse_indices(spec: str, total: int) -> List[int]:
    if spec == "all":
        return list(range(total))
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0 or value >= total:
            raise ValueError(f"Index {value} out of range [0, {total})")
        values.append(value)
    return sorted(set(values))


def parse_optional_indices(spec: str, total: int) -> List[int]:
    if not spec.strip():
        return []
    return parse_indices(spec, total)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def inverse_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return values * cos - rotate_half(values) * sin


def distance_bucket(distance: int) -> str:
    if distance <= 16:
        return "1_16"
    if distance <= 64:
        return "17_64"
    if distance <= 256:
        return "65_256"
    if distance <= 1024:
        return "257_1024"
    return "gt_1024"


def load_optional_checkpoint(model, path: str) -> Dict[str, object]:
    if not path:
        return {"loaded": False}
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    return {
        "loaded": True,
        "path": path,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_examples": list(missing)[:20],
        "unexpected_key_examples": list(unexpected)[:20],
    }


def get_qwen_core(model):
    return getattr(model, "model", model)


def attention_shape(attn, config) -> Tuple[int, int, int, int]:
    num_heads = int(getattr(attn, "num_heads", getattr(config, "num_attention_heads")))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", getattr(config, "num_key_value_heads")))
    head_dim = int(getattr(attn, "head_dim", getattr(config, "head_dim", config.hidden_size // num_heads)))
    groups = int(getattr(attn, "num_key_value_groups", num_heads // num_kv_heads))
    return num_heads, num_kv_heads, head_dim, groups


def compute_hidden_states(model, input_ids: torch.Tensor) -> Sequence[torch.Tensor]:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
    return outputs.hidden_states


def qwen_rotary(core, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    if not hasattr(core, "rotary_emb"):
        raise AttributeError("Model core has no rotary_emb; this script expects Qwen-style modules.")
    return core.rotary_emb(hidden_states, position_ids)


def project_qk_for_layer(layer, core, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    if apply_rotary_pos_emb is None or repeat_kv is None:
        raise ImportError(f"Could not import Qwen3 rotary helpers: {_QWEN3_IMPORT_ERROR}")
    attn = layer.self_attn
    num_heads, num_kv_heads, head_dim, groups = attention_shape(attn, core.config)
    normed = layer.input_layernorm(hidden_states)
    batch, seq_len, _ = normed.shape

    q_raw = attn.q_proj(normed).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k_raw = attn.k_proj(normed).view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q = attn.q_norm(q_raw)
    k = attn.k_norm(k_raw)
    cos, sin = qwen_rotary(core, normed, position_ids)
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    k_no_rope = repeat_kv(k, groups)
    k_rope = repeat_kv(k_rope, groups)
    return (
        normed.detach(),
        q.detach(),
        k_no_rope.detach(),
        q_rope.detach(),
        k_rope.detach(),
        k_raw.detach(),
        cos.detach(),
        sin.detach(),
    )


def energy_band_edges(singular_values: torch.Tensor, mode: str, num_bands: int, fixed_edges: str) -> List[Tuple[int, int, str]]:
    energy = singular_values.float().square()
    total = energy.sum().item()
    rank = int(singular_values.numel())
    if rank == 0 or total <= 0:
        return [(0, rank, "all")]
    cumulative = torch.cumsum(energy, dim=0) / total
    if mode == "fixed":
        edges = [float(item) for item in fixed_edges.split(",") if item.strip()]
        if edges[0] != 0.0:
            edges.insert(0, 0.0)
        if edges[-1] != 1.0:
            edges.append(1.0)
    else:
        edges = [idx / num_bands for idx in range(num_bands + 1)]

    result: List[Tuple[int, int, str]] = []
    prev = 0
    cumulative_list = cumulative.tolist()
    for left, right in zip(edges[:-1], edges[1:]):
        if prev >= rank:
            break
        start = prev
        end = bisect.bisect_left(cumulative_list, right) + 1
        end = max(start + 1, min(end, rank))
        label = f"{left:.4g}-{right:.4g}"
        result.append((start, end, label))
        prev = end
    if result:
        last_start, _, last_label = result[-1]
        result[-1] = (last_start, rank, last_label)
    return result


def head_mass(U: torch.Tensor, singular_values: torch.Tensor, head_dim: int, index: int) -> Dict[str, float]:
    start = index * head_dim
    end = start + head_dim
    local_mass = U[start:end, :].float().square().sum(dim=0).clamp_min(0.0)
    weights = singular_values.float().square()
    weighted = float((local_mass * weights).sum().item() / weights.sum().clamp_min(1e-12).item())
    unweighted = float(local_mass.mean().item())
    top10 = int(max(1, math.ceil(0.1 * local_mass.numel())))
    top_mass = float(local_mass[:top10].mean().item())
    return {
        "weighted_head_mass": weighted,
        "unweighted_head_mass": unweighted,
        "top10pct_unweighted_head_mass": top_mass,
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=0, eps=1e-8).item())


def causal_prefix_mean(values: torch.Tensor) -> torch.Tensor:
    """Return row i = mean(values[:i]); row 0 is zero."""
    result = torch.zeros_like(values, dtype=torch.float32)
    if values.shape[0] <= 1:
        return result
    prefix_sum = torch.cumsum(values.float(), dim=0)
    counts = torch.arange(1, values.shape[0], dtype=torch.float32, device=values.device)
    result[1:] = prefix_sum[:-1] / counts.unsqueeze(-1)
    return result


def causal_mean_energy_fractions(
    values: torch.Tensor,
    causal_means: torch.Tensor,
) -> torch.Tensor:
    """Fraction of historical mean-vector energy at every causal position."""
    result = torch.zeros(values.shape[0], dtype=torch.float32, device=values.device)
    if values.shape[0] <= 1:
        return result
    per_token_energy = values.float().square().sum(dim=-1)
    prefix_energy = torch.cumsum(per_token_energy, dim=0)
    counts = torch.arange(1, values.shape[0], dtype=torch.float32, device=values.device)
    historical_mean_energy = prefix_energy[:-1] / counts
    mean_vector_energy = causal_means[1:].float().square().sum(dim=-1)
    result[1:] = mean_vector_energy / historical_mean_energy.clamp_min(1e-12)
    return result


def randomized_truncated_svd(
    matrix: torch.Tensor,
    rank: int,
    oversample: int,
    power_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized truncated SVD with large products on model device.

    Apple MPS currently lacks native SVD and QR. Only QR of the skinny sketch
    and SVD of the compressed matrix run on CPU; large matrix products remain
    on MPS. Returned tensors are on CPU, matching the downstream attribution.
    """
    if rank <= 0:
        raise ValueError("Randomized SVD requires --svd_rank_limit > 0.")
    target_rank = min(rank, matrix.shape[0], matrix.shape[1])
    sketch_rank = min(target_rank + oversample, matrix.shape[0], matrix.shape[1])
    omega = torch.randn(
        matrix.shape[1],
        sketch_rank,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    y = matrix @ omega
    for _ in range(power_iters):
        q, _ = torch.linalg.qr(y.cpu(), mode="reduced")
        q = q.to(matrix.device)
        y = matrix @ (matrix.transpose(0, 1) @ q)
    q, _ = torch.linalg.qr(y.cpu(), mode="reduced")
    q = q.to(matrix.device)

    compressed = (q.transpose(0, 1) @ matrix).cpu()
    u_small, singular_values, vh = torch.linalg.svd(compressed, full_matrices=False)
    u = (q @ u_small.to(matrix.device)).cpu()
    return (
        u[:, :target_rank],
        singular_values[:target_rank],
        vh[:target_rank],
    )


def compute_svd(
    matrix: torch.Tensor,
    method: str,
    rank_limit: int,
    oversample: int,
    power_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if method == "randomized":
        return randomized_truncated_svd(
            matrix,
            rank_limit,
            oversample,
            power_iters,
        )
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u.cpu(), singular_values.cpu(), vh.cpu()


def choose_random_negatives(
    history_len: int,
    positives: torch.Tensor,
    count: int,
    rng: random.Random,
) -> List[int]:
    if count <= 0 or history_len <= 0:
        return []
    positive_set = set(int(x) for x in positives.tolist())
    candidates = [idx for idx in range(history_len) if idx not in positive_set]
    if not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    return rng.sample(candidates, count)


def choose_distance_negatives(
    history_len: int,
    positive_indices: Sequence[int],
    positives: torch.Tensor,
    count: int,
) -> List[int]:
    if count <= 0 or history_len <= 0:
        return []
    positive_set = set(int(x) for x in positives.tolist())
    available = [idx for idx in range(history_len) if idx not in positive_set]
    if not available:
        return []
    selected: List[int] = []
    for pos in positive_indices[:count]:
        target_distance = history_len - int(pos)
        best = min(available, key=lambda idx: abs((history_len - idx) - target_distance))
        selected.append(best)
        available.remove(best)
        if not available:
            break
    return selected


def choose_tail_for_svd(
    scores: torch.Tensor,
    positives: torch.Tensor,
    count: int,
    mode: str,
    rng: random.Random,
) -> List[int]:
    if count <= 0 or scores.numel() == 0:
        return []
    positive_set = set(int(x) for x in positives.tolist())
    candidates = [idx for idx in range(scores.numel()) if idx not in positive_set]
    if not candidates:
        return []
    if mode == "random_tail":
        if len(candidates) <= count:
            return candidates
        return rng.sample(candidates, count)
    ranked = sorted(candidates, key=lambda idx: float(scores[idx].item()))
    return ranked[: min(count, len(ranked))]


def band_contribution_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    bridge: torch.Tensor,
    q_bands: Sequence[Tuple[int, int, str]],
    k_bands: Sequence[Tuple[int, int, str]],
) -> torch.Tensor:
    matrix = torch.zeros(len(q_bands), len(k_bands), dtype=torch.float64)
    for qi, (qs, qe, _) in enumerate(q_bands):
        av = a[qs:qe].double()
        if av.numel() == 0:
            continue
        for ki, (ks, ke, _) in enumerate(k_bands):
            bv = b[ks:ke].double()
            if bv.numel() == 0:
                continue
            block = bridge[qs:qe, ks:ke].double()
            matrix[qi, ki] += (av @ block @ bv).item()
    return matrix


def aggregate_band_contribution_matrix(
    q_scaled_all: torch.Tensor,
    k_scaled_all: torch.Tensor,
    bridge: torch.Tensor,
    q_bands: Sequence[Tuple[int, int, str]],
    k_bands: Sequence[Tuple[int, int, str]],
    pairs: Sequence[Tuple[int, int]],
    k_causal_means: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    signed = torch.zeros(len(q_bands), len(k_bands), dtype=torch.float64)
    absolute = torch.zeros_like(signed)
    if not pairs:
        return signed, absolute
    query_indices = torch.tensor([item[0] for item in pairs], dtype=torch.long)
    key_indices = torch.tensor([item[1] for item in pairs], dtype=torch.long)
    for qi, (qs, qe, _) in enumerate(q_bands):
        av = q_scaled_all[query_indices, qs:qe].double()
        if av.numel() == 0:
            continue
        for ki, (ks, ke, _) in enumerate(k_bands):
            bv = k_scaled_all[key_indices, ks:ke].double()
            if k_causal_means is not None:
                bv = bv - k_causal_means[query_indices, ks:ke].double()
            if bv.numel() == 0:
                continue
            block = bridge[qs:qe, ks:ke].double()
            values = (av @ block * bv).sum(dim=1)
            signed[qi, ki] = values.sum()
            absolute[qi, ki] = values.abs().sum()
    return signed, absolute


def top_feature_pairs(
    a: torch.Tensor,
    b: torch.Tensor,
    bridge: torch.Tensor,
    limit_rank: int,
    top_k: int,
) -> List[Dict[str, float]]:
    rq = min(limit_rank, a.numel(), bridge.shape[0])
    rk = min(limit_rank, b.numel(), bridge.shape[1])
    if rq <= 0 or rk <= 0 or top_k <= 0:
        return []
    contrib = (a[:rq, None].float() * bridge[:rq, :rk].float()) * b[None, :rk].float()
    flat = contrib.abs().flatten()
    k = min(top_k, flat.numel())
    values, indices = torch.topk(flat, k=k)
    result = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        r = flat_index // rk
        s = flat_index % rk
        result.append({
            "q_feature": int(r),
            "k_feature": int(s),
            "contribution": float(contrib[r, s].item()),
            "abs_contribution": float(value),
        })
    return result


def normalize_matrix(matrix: torch.Tensor, mode: str) -> List[List[float]]:
    if mode == "abs":
        values = matrix.abs()
    else:
        values = matrix
    denom = values.abs().sum().clamp_min(1e-12)
    return (values / denom).tolist()


def full_k_direction_attribution(
    top_pairs: Sequence[Tuple[int, int]],
    random_pairs: Sequence[Tuple[int, int]],
    q_rope_head: torch.Tensor,
    k_rope_head: torch.Tensor,
    k_raw_head: torch.Tensor,
    z_all: torch.Tensor,
    u_k_head: torch.Tensor,
    singular_values: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_eps: float,
    head_dim: int,
) -> Dict[str, object]:
    def contributions(pairs: Sequence[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not pairs:
            empty = torch.zeros(0, singular_values.numel())
            return empty, torch.zeros(0)
        query_indices = torch.tensor([pair[0] for pair in pairs], dtype=torch.long, device=q_rope_head.device)
        key_indices = torch.tensor([pair[1] for pair in pairs], dtype=torch.long, device=q_rope_head.device)
        q_values = q_rope_head[query_indices].float()
        q_before_key_rope = inverse_rope(
            q_values,
            rope_cos[key_indices].float(),
            rope_sin[key_indices].float(),
        )
        raw_keys = k_raw_head[key_indices].float()
        rms_scale = torch.rsqrt(raw_keys.square().mean(dim=-1) + k_norm_eps)
        sensitivity = (
            q_before_key_rope
            * k_norm_weight.float().unsqueeze(0)
            * rms_scale.unsqueeze(-1)
        ) @ u_k_head.float()
        per_direction = z_all[key_indices].float() * sensitivity / math.sqrt(head_dim)
        actual_scores = (
            q_values * k_rope_head[key_indices].float()
        ).sum(dim=-1) / math.sqrt(head_dim)
        return per_direction.cpu(), actual_scores.cpu()

    top_contrib, top_actual = contributions(top_pairs)
    random_contrib, random_actual = contributions(random_pairs)
    top_mean = top_contrib.mean(dim=0)
    random_mean = random_contrib.mean(dim=0)
    top_abs_mean = top_contrib.abs().mean(dim=0)
    random_abs_mean = random_contrib.abs().mean(dim=0)
    contrast = top_mean - random_mean
    abs_contrast = contrast.abs()
    order = torch.argsort(abs_contrast, descending=True)
    cumulative = torch.cumsum(abs_contrast[order], dim=0) / abs_contrast.sum().clamp_min(1e-12)

    def directions_for_fraction(fraction: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(fraction)).item() + 1)

    top_reconstructed = top_contrib.sum(dim=-1)
    random_reconstructed = random_contrib.sum(dim=-1)
    singular_cpu = singular_values.float().cpu()
    if singular_cpu.numel() > 1 and abs_contrast.std().item() > 0:
        sigma_contrast_corr = float(torch.corrcoef(torch.stack((singular_cpu, abs_contrast)))[0, 1].item())
    else:
        sigma_contrast_corr = 0.0

    return {
        "definition": "per-K-direction contribution to actual q_norm/k_norm/RoPE score; top-minus-random cancels causal common offsets",
        "top_pair_count": len(top_pairs),
        "random_pair_count": len(random_pairs),
        "singular_values": singular_cpu.tolist(),
        "top_mean_signed_contribution": top_mean.tolist(),
        "random_mean_signed_contribution": random_mean.tolist(),
        "top_minus_random_signed_contribution": contrast.tolist(),
        "top_mean_abs_contribution": top_abs_mean.tolist(),
        "random_mean_abs_contribution": random_abs_mean.tolist(),
        "directions_ranked_by_abs_contrast": order.tolist(),
        "directions_for_50pct_abs_contrast": directions_for_fraction(0.5),
        "directions_for_90pct_abs_contrast": directions_for_fraction(0.9),
        "singular_value_abs_contrast_correlation": sigma_contrast_corr,
        "top_reconstruction_mae": float((top_reconstructed - top_actual).abs().mean().item()),
        "random_reconstruction_mae": float((random_reconstructed - random_actual).abs().mean().item()),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    required_tokens = args.token_start + args.num_query_tokens + args.extra_tokens
    tokens, token_meta, source_text = collect_analysis_tokens(args, tokenizer, required_tokens)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)

    print(f"[load] model={args.model_dir} device={device} dtype={dtype} tokens={input_ids.shape[1]}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    checkpoint_info = load_optional_checkpoint(model, args.checkpoint_path)
    model.eval()
    core = get_qwen_core(model)
    layers = parse_indices(args.layers, len(core.layers))
    full_k_direction_layers = parse_optional_indices(args.full_k_direction_layers, len(core.layers))
    num_heads, _, _, _ = attention_shape(core.layers[0].self_attn, core.config)
    heads = parse_indices(args.heads, num_heads)

    print("[forward] collecting hidden states", flush=True)
    hidden_states = compute_hidden_states(model, input_ids)
    print(f"[forward] hidden_states={len(hidden_states)}", flush=True)

    examples_path = output_dir / "qk_feature_svd_examples.jsonl"
    examples_written = 0
    examples_handle = examples_path.open("w", encoding="utf-8")
    summaries = []

    metadata = {
        "model_dir": args.model_dir,
        "checkpoint": checkpoint_info,
        "data_dir": args.data_dir,
        "input_source": args.input_source,
        "prompt_file": args.prompt_file,
        "token_start": args.token_start,
        "num_query_tokens": args.num_query_tokens,
        "top_ratio": args.top_ratio,
        "layers": layers,
        "heads": heads,
        "rope_ablation": args.rope_ablation,
        "full_k_direction_layers": full_k_direction_layers,
        "full_k_keys_per_query": args.full_k_keys_per_query,
        "device": str(device),
        "dtype": str(dtype),
        "svd_device": args.svd_device,
        "svd_method": args.svd_method,
        "svd_oversample": args.svd_oversample,
        "svd_power_iters": args.svd_power_iters,
        "token_meta": token_meta,
        "created_at": time.time(),
        "formula": (
            "c_h_ijrs=(x_i V_Q)_r sigma_Q_r (U_Q_h^T U_K_h)_{r,s} "
            "sigma_K_s (x_j V_K)_s"
        ),
        "qk_selection_space": "actual q_norm/k_norm + RoPE attention score",
        "svd_attribution_space": "linear full WQ/WK SVD before q_norm/k_norm and RoPE",
        "center_keys": args.center_keys,
        "centering_contract": {
            "causal_mean": "mean of positions [0, query_pos)",
            "exact_k_space": "post q_norm and RoPE; subtract the same causal mean from every historical key",
            "linear_svd_space": "subtract causal mean from (x V_K) Sigma_K before spectral attribution",
        },
    }

    if args.save_input_tokens:
        target_positions = list(range(args.token_start, args.token_start + args.num_query_tokens))
        token_dump = {
            "input_ids": tokens,
            "target_positions": target_positions,
            "target_token_ids": [tokens[pos] for pos in target_positions],
            "target_token_text": [tokenizer.decode([tokens[pos]]) for pos in target_positions],
            "context_tail_text": tokenizer.decode(tokens[max(0, args.token_start - 256):args.token_start]),
        }
        (output_dir / "input_tokens.json").write_text(
            json.dumps(token_dump, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if source_text is not None:
            (output_dir / "source_prompt.txt").write_text(source_text, encoding="utf-8")

    for layer_idx in layers:
        layer = core.layers[layer_idx]
        attn = layer.self_attn
        _, num_kv_heads, head_dim, groups = attention_shape(attn, core.config)
        layer_hidden = hidden_states[layer_idx].to(device)
        print(f"[layer {layer_idx}] projecting Q/K", flush=True)
        (
            normed,
            q_no_rope,
            k_no_rope,
            q_states,
            k_states,
            k_raw,
            rope_cos,
            rope_sin,
        ) = project_qk_for_layer(layer, core, layer_hidden, position_ids)
        normed_cpu = normed[0].detach().float().cpu()
        hidden_causal_means = causal_prefix_mean(normed_cpu) if args.center_keys else None
        hidden_common_mean_energy = (
            causal_mean_energy_fractions(normed_cpu, hidden_causal_means)
            if args.center_keys
            else None
        )
        q_states = q_states[0]  # [H, T, D]
        k_states = k_states[0]  # [H, T, D] repeated to query heads
        q_no_rope = q_no_rope[0]
        k_no_rope = k_no_rope[0]
        k_raw = k_raw[0]  # [KV_H, T, D]
        rope_cos = rope_cos[0]
        rope_sin = rope_sin[0]

        print(f"[layer {layer_idx}] SVD WQ/WK", flush=True)
        svd_device = device if args.svd_device == "model" else torch.device("cpu")
        if args.svd_method == "randomized" and args.svd_device != "model":
            raise ValueError("Randomized SVD requires --svd_device model to accelerate large matrix products.")
        Wq = attn.q_proj.weight.detach().float().to(svd_device)
        Wk = attn.k_proj.weight.detach().float().to(svd_device)
        Uq, Sq, Vhq = compute_svd(
            Wq,
            args.svd_method,
            args.svd_rank_limit,
            args.svd_oversample,
            args.svd_power_iters,
        )
        if layer_idx in full_k_direction_layers:
            print(f"[layer {layer_idx}] exact full-rank K SVD for direction attribution", flush=True)
            Uk, Sk, Vhk = compute_svd(
                Wk.cpu(),
                "exact",
                0,
                0,
                0,
            )
        else:
            Uk, Sk, Vhk = compute_svd(
                Wk,
                args.svd_method,
                args.svd_rank_limit,
                args.svd_oversample,
                args.svd_power_iters,
            )
        Vq = Vhq.T.cpu()
        Vk = Vhk.T.cpu()

        q_rank = Sq.numel() if args.svd_rank_limit <= 0 else min(args.svd_rank_limit, Sq.numel())
        k_rank = Sk.numel() if args.svd_rank_limit <= 0 else min(args.svd_rank_limit, Sk.numel())
        q_bands = energy_band_edges(Sq[:q_rank], args.band_mode, args.num_energy_bands, args.fixed_energy_edges)
        k_bands = energy_band_edges(Sk[:k_rank], args.band_mode, args.num_energy_bands, args.fixed_energy_edges)
        q_features_all = normed_cpu @ Vq[:, :q_rank]
        k_features_all = normed_cpu @ Vk[:, :k_rank]
        Sq_used = Sq[:q_rank]
        Sk_used = Sk[:k_rank]
        q_scaled_all = q_features_all * Sq_used
        k_scaled_all = k_features_all * Sk_used
        linear_k_causal_means = causal_prefix_mean(k_scaled_all) if args.center_keys else None
        linear_k_common_mean_energy = (
            causal_mean_energy_fractions(k_scaled_all, linear_k_causal_means)
            if args.center_keys
            else None
        )
        if layer_idx in full_k_direction_layers:
            full_k_u = Uk.to(device)
            full_k_s = Sk.to(device)
            full_k_v = Vk.to(device)
            full_k_z_all = (normed[0].float() @ full_k_v.float()) * full_k_s.float()
        else:
            full_k_u = full_k_s = full_k_z_all = None

        for head_idx in heads:
            kv_head = head_idx // groups
            if kv_head >= num_kv_heads:
                kv_head = num_kv_heads - 1
            q_start = head_idx * head_dim
            q_end = q_start + head_dim
            k_start = kv_head * head_dim
            k_end = k_start + head_dim
            bridge = Uq[q_start:q_end, :q_rank].T @ Uk[k_start:k_end, :k_rank]

            stats = MeanStats()
            top_svd_pairs: List[Tuple[int, int]] = []
            tail_svd_pairs: List[Tuple[int, int]] = []
            full_k_top_pairs: List[Tuple[int, int]] = []
            full_k_random_pairs: List[Tuple[int, int]] = []
            q_mass = head_mass(Uq, Sq, head_dim, head_idx)
            k_mass = head_mass(Uk, Sk, head_dim, kv_head)
            k_head_cpu = k_states[head_idx].detach().float().cpu()
            k_causal_means = causal_prefix_mean(k_head_cpu) if args.center_keys else None
            k_common_mean_energy = (
                causal_mean_energy_fractions(k_head_cpu, k_causal_means)
                if args.center_keys
                else None
            )
            k_causal_means_device = (
                k_causal_means.to(device=k_states.device) if args.center_keys else None
            )
            if args.rope_ablation:
                k_no_rope_head_cpu = k_no_rope[head_idx].detach().float().cpu()
                k_no_rope_causal_means = causal_prefix_mean(k_no_rope_head_cpu)
                k_no_rope_causal_means_device = k_no_rope_causal_means.to(k_no_rope.device)
            else:
                k_no_rope_head_cpu = None
                k_no_rope_causal_means = None
                k_no_rope_causal_means_device = None

            for query_pos in range(args.token_start, args.token_start + args.num_query_tokens):
                if query_pos <= 0 or query_pos >= input_ids.shape[1]:
                    continue
                q_vec = q_states[head_idx, query_pos].detach()
                key_mat = k_states[head_idx, :query_pos].detach()
                scores = torch.matmul(key_mat.float(), q_vec.float()) / math.sqrt(head_dim)
                if args.rope_ablation:
                    q_no_rope_vec = q_no_rope[head_idx, query_pos].detach()
                    k_no_rope_mat = k_no_rope[head_idx, :query_pos].detach()
                    no_rope_scores = (
                        torch.matmul(k_no_rope_mat.float(), q_no_rope_vec.float())
                        / math.sqrt(head_dim)
                    )
                    no_rope_mean = k_no_rope_causal_means_device[query_pos]
                    centered_no_rope_scores = (
                        torch.matmul(
                            k_no_rope_mat.float() - no_rope_mean.unsqueeze(0),
                            q_no_rope_vec.float(),
                        )
                        / math.sqrt(head_dim)
                    )
                else:
                    no_rope_scores = centered_no_rope_scores = None
                if args.center_keys:
                    k_mean = k_causal_means_device[query_pos]
                    centered_scores = torch.matmul(
                        key_mat.float() - k_mean.unsqueeze(0),
                        q_vec.float(),
                    ) / math.sqrt(head_dim)
                    score_shift = scores - centered_scores
                    centered_softmax = torch.softmax(centered_scores, dim=0)
                    original_softmax = torch.softmax(scores, dim=0)
                    stats.add(
                        "key_centered_logit_shift_residual_max",
                        float((score_shift - score_shift.mean()).abs().max().item()),
                    )
                    stats.add(
                        "key_centered_softmax_max_abs_diff",
                        float((original_softmax - centered_softmax).abs().max().item()),
                    )
                    stats.add(
                        "key_common_mean_energy_fraction",
                        float(k_common_mean_energy[query_pos].item()),
                    )
                    stats.add(
                        "hidden_common_mean_energy_fraction",
                        float(hidden_common_mean_energy[query_pos].item()),
                    )
                    stats.add(
                        "linear_k_common_mean_energy_fraction",
                        float(linear_k_common_mean_energy[query_pos].item()),
                    )
                else:
                    centered_scores = None
                top_k = max(1, int(math.ceil(query_pos * args.top_ratio)))
                top_k = min(top_k, scores.numel())
                top_scores, top_indices = torch.topk(scores, k=top_k, largest=True)
                if args.rope_ablation:
                    _, no_rope_top_indices = torch.topk(no_rope_scores, k=top_k, largest=True)
                    overlap = len(set(top_indices.tolist()) & set(no_rope_top_indices.tolist())) / top_k
                    stats.add("rope_no_rope_top_set_overlap", overlap)
                svd_top_count = top_k if args.svd_keys_per_query <= 0 else min(args.svd_keys_per_query, top_k)
                tail_svd_count = svd_top_count if args.tail_svd_keys_per_query <= 0 else args.tail_svd_keys_per_query
                tail_indices_for_svd = choose_tail_for_svd(
                    scores,
                    top_indices,
                    tail_svd_count,
                    args.tail_sample_mode,
                    rng,
                )
                stats.add("top_count", float(top_k))
                stats.add("top_score_mean", float(top_scores.mean().item()))
                stats.add("top_score_cutoff", float(top_scores[-1].item()))
                if tail_indices_for_svd:
                    tail_scores = scores[torch.tensor(tail_indices_for_svd, device=scores.device)]
                    stats.add("tail_svd_score_mean", float(tail_scores.mean().item()))
                    stats.add("tail_svd_score_max", float(tail_scores.max().item()))

                pos_metric_count = max(
                    args.random_negatives_per_query,
                    args.distance_negatives_per_query,
                    svd_top_count,
                )
                pos_for_metrics = top_indices[:pos_metric_count]
                random_negs = choose_random_negatives(query_pos, top_indices, args.random_negatives_per_query, rng)
                dist_negs = choose_distance_negatives(query_pos, pos_for_metrics.tolist(), top_indices, args.distance_negatives_per_query)

                def add_pair_metrics(prefix: str, key_idx: int) -> None:
                    hq = normed_cpu[query_pos]
                    hk = normed_cpu[key_idx]
                    stats.add(f"{prefix}_hidden_cos", cosine(hq, hk))
                    stats.add(f"{prefix}_k_cos", cosine(k_head_cpu[query_pos], k_head_cpu[key_idx]))
                    stats.add(f"{prefix}_qk_score", float(scores[key_idx].item()))
                    stats.add(f"{prefix}_distance", float(query_pos - key_idx))
                    if args.rope_ablation:
                        rope_delta = float((scores[key_idx] - no_rope_scores[key_idx]).item())
                        stats.add(f"{prefix}_no_rope_qk_score", float(no_rope_scores[key_idx].item()))
                        stats.add(f"{prefix}_centered_no_rope_qk_score", float(centered_no_rope_scores[key_idx].item()))
                        stats.add(f"{prefix}_rope_delta_score", rope_delta)
                        stats.add(
                            f"{prefix}_rope_delta_distance_{distance_bucket(query_pos - key_idx)}",
                            rope_delta,
                        )
                    if args.center_keys:
                        hidden_mean = hidden_causal_means[query_pos]
                        stats.add(
                            f"{prefix}_centered_hidden_cos",
                            cosine(hq - hidden_mean, hk - hidden_mean),
                        )
                        key_mean = k_causal_means[query_pos]
                        stats.add(
                            f"{prefix}_centered_k_cos",
                            cosine(
                                k_head_cpu[query_pos] - key_mean,
                                k_head_cpu[key_idx] - key_mean,
                            ),
                        )
                        if args.rope_ablation:
                            no_rope_mean_cpu = k_no_rope_causal_means[query_pos]
                            stats.add(
                                f"{prefix}_centered_no_rope_k_cos",
                                cosine(
                                    k_no_rope_head_cpu[query_pos] - no_rope_mean_cpu,
                                    k_no_rope_head_cpu[key_idx] - no_rope_mean_cpu,
                                ),
                            )
                        stats.add(
                            f"{prefix}_centered_qk_score",
                            float(centered_scores[key_idx].item()),
                        )

                for key_idx in pos_for_metrics.tolist():
                    add_pair_metrics("pos", int(key_idx))
                for key_idx in random_negs:
                    add_pair_metrics("random_neg", int(key_idx))
                for key_idx in dist_negs:
                    add_pair_metrics("distance_neg", int(key_idx))

                svd_keys = top_indices[:svd_top_count].tolist()
                for key_idx in svd_keys:
                    top_svd_pairs.append((query_pos, int(key_idx)))
                    if examples_written < args.max_examples:
                        a = q_scaled_all[query_pos]
                        b = k_scaled_all[int(key_idx)]
                        example = {
                            "layer": layer_idx,
                            "head": head_idx,
                            "kv_head": kv_head,
                            "query_pos": query_pos,
                            "key_pos": int(key_idx),
                            "distance": query_pos - int(key_idx),
                            "qk_score": float(scores[int(key_idx)].item()),
                            "query_token_id": int(tokens[query_pos]),
                            "key_token_id": int(tokens[int(key_idx)]),
                            "query_token_text": tokenizer.decode([tokens[query_pos]]),
                            "key_token_text": tokenizer.decode([tokens[int(key_idx)]]),
                            "top_feature_pairs": top_feature_pairs(
                                a,
                                b,
                                bridge,
                                args.example_rank_limit,
                                args.example_top_pairs,
                            ),
                        }
                        examples_handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                        examples_written += 1

                for key_idx in tail_indices_for_svd:
                    tail_svd_pairs.append((query_pos, int(key_idx)))

                if layer_idx in full_k_direction_layers:
                    direction_count = min(args.full_k_keys_per_query, top_k)
                    direction_top = top_indices[:direction_count].tolist()
                    direction_random = choose_random_negatives(
                        query_pos,
                        top_indices,
                        direction_count,
                        rng,
                    )
                    full_k_top_pairs.extend((query_pos, int(key_idx)) for key_idx in direction_top)
                    full_k_random_pairs.extend((query_pos, int(key_idx)) for key_idx in direction_random)

            top_band_signed, top_band_abs = aggregate_band_contribution_matrix(
                q_scaled_all, k_scaled_all, bridge, q_bands, k_bands, top_svd_pairs
            )
            tail_band_signed, tail_band_abs = aggregate_band_contribution_matrix(
                q_scaled_all, k_scaled_all, bridge, q_bands, k_bands, tail_svd_pairs
            )
            if args.center_keys:
                centered_top_band_signed, centered_top_band_abs = aggregate_band_contribution_matrix(
                    q_scaled_all,
                    k_scaled_all,
                    bridge,
                    q_bands,
                    k_bands,
                    top_svd_pairs,
                    k_causal_means=linear_k_causal_means,
                )
                centered_tail_band_signed, centered_tail_band_abs = aggregate_band_contribution_matrix(
                    q_scaled_all,
                    k_scaled_all,
                    bridge,
                    q_bands,
                    k_bands,
                    tail_svd_pairs,
                    k_causal_means=linear_k_causal_means,
                )
            else:
                centered_top_band_signed = centered_top_band_abs = None
                centered_tail_band_signed = centered_tail_band_abs = None
            top_band_pair_count = len(top_svd_pairs)
            tail_band_pair_count = len(tail_svd_pairs)
            if layer_idx in full_k_direction_layers:
                k_norm_eps = float(
                    getattr(
                        attn.k_norm,
                        "variance_epsilon",
                        getattr(attn.k_norm, "eps", 1e-6),
                    )
                )
                full_k_attribution = full_k_direction_attribution(
                    full_k_top_pairs,
                    full_k_random_pairs,
                    q_states[head_idx],
                    k_states[head_idx],
                    k_raw[kv_head],
                    full_k_z_all,
                    full_k_u[k_start:k_end],
                    full_k_s,
                    rope_cos,
                    rope_sin,
                    attn.k_norm.weight.to(device),
                    k_norm_eps,
                    head_dim,
                )
            else:
                full_k_attribution = None

            summary = {
                "layer": layer_idx,
                "head": head_idx,
                "kv_head": kv_head,
                "head_dim": head_dim,
                "stats": stats.to_dict(),
                "q_svd_head_mass": q_mass,
                "k_svd_head_mass": k_mass,
                "q_band_labels": [label for _, _, label in q_bands],
                "k_band_labels": [label for _, _, label in k_bands],
                "svd_rank_limit": args.svd_rank_limit,
                "q_rank_used": q_rank,
                "k_rank_used": k_rank,
                "full_k_direction_attribution": full_k_attribution,
                "top_svd_band_pair_count": top_band_pair_count,
                "tail_svd_band_pair_count": tail_band_pair_count,
                "top_svd_band_signed_fraction": normalize_matrix(top_band_signed, mode="signed") if top_band_pair_count else [],
                "top_svd_band_abs_fraction": normalize_matrix(top_band_abs, mode="abs") if top_band_pair_count else [],
                "tail_svd_band_signed_fraction": normalize_matrix(tail_band_signed, mode="signed") if tail_band_pair_count else [],
                "tail_svd_band_abs_fraction": normalize_matrix(tail_band_abs, mode="abs") if tail_band_pair_count else [],
                "top_minus_tail_svd_band_abs_fraction": (
                    (
                        torch.tensor(normalize_matrix(top_band_abs, mode="abs"))
                        - torch.tensor(normalize_matrix(tail_band_abs, mode="abs"))
                    ).tolist()
                    if top_band_pair_count and tail_band_pair_count
                    else []
                ),
                "centered_linear_k_top_svd_band_signed_fraction": (
                    normalize_matrix(centered_top_band_signed, mode="signed")
                    if args.center_keys and top_band_pair_count
                    else []
                ),
                "centered_linear_k_top_svd_band_abs_fraction": (
                    normalize_matrix(centered_top_band_abs, mode="abs")
                    if args.center_keys and top_band_pair_count
                    else []
                ),
                "centered_linear_k_tail_svd_band_signed_fraction": (
                    normalize_matrix(centered_tail_band_signed, mode="signed")
                    if args.center_keys and tail_band_pair_count
                    else []
                ),
                "centered_linear_k_tail_svd_band_abs_fraction": (
                    normalize_matrix(centered_tail_band_abs, mode="abs")
                    if args.center_keys and tail_band_pair_count
                    else []
                ),
                "centered_linear_k_top_minus_tail_svd_band_abs_fraction": (
                    (
                        torch.tensor(normalize_matrix(centered_top_band_abs, mode="abs"))
                        - torch.tensor(normalize_matrix(centered_tail_band_abs, mode="abs"))
                    ).tolist()
                    if args.center_keys and top_band_pair_count and tail_band_pair_count
                    else []
                ),
            }
            summaries.append(summary)

        del (
            q_states,
            k_states,
            q_no_rope,
            k_no_rope,
            k_raw,
            rope_cos,
            rope_sin,
            normed,
            normed_cpu,
            Uq,
            Sq,
            Vq,
            Uk,
            Sk,
            Vk,
        )
        if layer_idx in full_k_direction_layers:
            del full_k_u, full_k_s, full_k_z_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

    examples_handle.close()
    result = {
        "metadata": metadata,
        "summaries": summaries,
    }
    summary_path = output_dir / "qk_feature_svd_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    compact_path = output_dir / "qk_feature_svd_summary.jsonl"
    with compact_path.open("w", encoding="utf-8") as handle:
        for item in summaries:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[done] summary={summary_path}", flush=True)
    print(f"[done] examples={examples_path} examples_written={examples_written}", flush=True)


if __name__ == "__main__":
    main()
