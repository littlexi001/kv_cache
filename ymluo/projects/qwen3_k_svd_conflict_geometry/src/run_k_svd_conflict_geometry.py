from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_RULE_SRC = SCRIPT_DIR.parents[1] / "qwen3_local_rule_failure_boundary" / "src"
if str(LOCAL_RULE_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_RULE_SRC))

import run_local_rule_failure_boundary as base  # noqa: E402
from run_four_condition_answer_eval_20260717 import build_variant  # noqa: E402


PAIR_CONDITIONS = {
    "short": ("gold_only", "gold_plus_conflict"),
    "filler_8k": ("filler_plus_gold", "filler_plus_gold_plus_conflict"),
}


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for start in range(last):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def code_positions(
    tokenizer: Any,
    prompt_ids: list[int],
    event: base.RuleEvent,
    code: str,
) -> list[int]:
    """Find one code occurrence inside an event, preserving its contextual tokenization."""

    event_ids = prompt_ids[event.start_token : event.end_token]
    candidates: list[list[int]] = []
    for text in (" " + code, code):
        ids = base.token_ids(tokenizer, text)
        if ids and ids not in candidates:
            candidates.append(ids)
    # Prefer the longer contextual encoding.  It normally includes the same leading
    # whitespace piece in both VERIFIED and DECOY rule lines.
    candidates.sort(key=len, reverse=True)
    for ids in candidates:
        offset = find_subsequence(event_ids, ids)
        if offset is not None:
            positions = list(range(event.start_token + offset, event.start_token + offset + len(ids)))
            # A standalone whitespace token is not part of the numbered identifier.
            while positions and not tokenizer.decode([prompt_ids[positions[0]]]).strip():
                positions.pop(0)
            return positions
    raise ValueError(f"Could not locate code {code!r} in event {event.label}: {event.text!r}")


def build_position_roles(tokenizer: Any, variant: dict[str, Any]) -> dict[str, Any]:
    prompt_ids = variant["prompt_ids"][0].tolist()
    events: list[base.RuleEvent] = variant["events"]
    gold_events = sorted((event for event in events if event.kind == "relevant"), key=lambda e: e.step)
    conflict_events = sorted((event for event in events if event.kind == "conflict"), key=lambda e: e.step)

    roles: dict[str, list[int]] = {
        "gold_rule": [
            pos for event in gold_events for pos in range(event.start_token, event.end_token)
        ],
        "conflict_rule": [
            pos for event in conflict_events for pos in range(event.start_token, event.end_token)
        ],
        "gold_code": [],
        "conflict_code": [],
    }
    for event in gold_events:
        roles["gold_code"].extend(code_positions(tokenizer, prompt_ids, event, event.antecedent))
        roles["gold_code"].extend(code_positions(tokenizer, prompt_ids, event, event.consequent))
    for event in conflict_events:
        roles["conflict_code"].extend(code_positions(tokenizer, prompt_ids, event, event.antecedent))
        roles["conflict_code"].extend(code_positions(tokenizer, prompt_ids, event, event.consequent))
    for name in roles:
        roles[name] = sorted(set(roles[name]))

    shared_pairs: list[dict[str, Any]] = []
    if conflict_events:
        shared_code = gold_events[0].antecedent
        gold_positions = code_positions(tokenizer, prompt_ids, gold_events[0], shared_code)
        conflict_positions = code_positions(tokenizer, prompt_ids, conflict_events[0], shared_code)
        gold_ids = [prompt_ids[pos] for pos in gold_positions]
        conflict_ids = [prompt_ids[pos] for pos in conflict_positions]
        if gold_ids != conflict_ids:
            raise ValueError(
                f"Contextual tokenization differs for shared code {shared_code}: "
                f"{gold_ids} != {conflict_ids}"
            )
        for index, (gold_pos, conflict_pos, token_id) in enumerate(
            zip(gold_positions, conflict_positions, gold_ids)
        ):
            decoded = tokenizer.decode([token_id])
            shared_pairs.append(
                {
                    "code": shared_code,
                    "subtoken_index": index,
                    "token_id": token_id,
                    "decoded": decoded,
                    "is_numeric": int(any(char.isdigit() for char in decoded)),
                    "gold_position": gold_pos,
                    "conflict_position": conflict_pos,
                }
            )

    gold_start = min((event.start_token for event in gold_events), default=-1)
    conflict_start = min((event.start_token for event in conflict_events), default=-1)
    return {
        "roles": roles,
        "shared_pairs": shared_pairs,
        "gold_start": gold_start,
        "conflict_start": conflict_start,
        "conflict_order": (
            "none" if conflict_start < 0 else "before_gold" if conflict_start < gold_start else "after_gold"
        ),
    }


@torch.inference_mode()
def collect_features(
    model: Any,
    tokenizer: Any,
    variant: dict[str, Any],
    prefill_chunk: int,
) -> dict[str, Any]:
    prompt_ids: torch.Tensor = variant["prompt_ids"]
    positions = build_position_roles(tokenizer, variant)
    device = base.input_device(model)
    layers = list(model.model.layers)
    selected_positions = sorted(
        {
            pos
            for values in positions["roles"].values()
            for pos in values
        }
        | {
            item[key]
            for item in positions["shared_pairs"]
            for key in ("gold_position", "conflict_position")
        }
    )
    captured_pre_rope_k: dict[int, dict[int, torch.Tensor]] = {
        layer_idx: {} for layer_idx in range(len(layers))
    }
    key_offsets = [0 for _ in layers]
    captured_q: dict[int, torch.Tensor] = {}
    handles = []

    def make_pre_rope_k_hook(layer_idx: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            if hidden_states is None:
                return
            projected = module.k_proj(hidden_states)
            batch, q_len, _ = projected.shape
            head_dim = int(module.head_dim)
            kv_heads = int(projected.shape[-1] // head_dim)
            key = projected.view(batch, q_len, kv_heads, head_dim)
            if getattr(module, "k_norm", None) is not None:
                key = module.k_norm(key)
            start = key_offsets[layer_idx]
            end = start + q_len
            for position in selected_positions:
                if start <= position < end:
                    captured_pre_rope_k[layer_idx][position] = (
                        key[0, position - start].detach().float().cpu()
                    )
            key_offsets[layer_idx] = end

        return hook

    def make_hook(layer_idx: int):
        def hook(module: Any, hook_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and hook_args:
                hidden_states = hook_args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(hook_args) >= 2:
                position_embeddings = hook_args[1]
            if hidden_states is None:
                return
            projected = module.q_proj(hidden_states)
            batch, q_len, _ = projected.shape
            head_dim = int(module.head_dim)
            num_heads = int(projected.shape[-1] // head_dim)
            q = projected.view(batch, q_len, num_heads, head_dim)
            if getattr(module, "q_norm", None) is not None:
                q = module.q_norm(q)
            q = q.transpose(1, 2)
            q = base.apply_rope_to_q(q, position_embeddings)
            captured_q[layer_idx] = q[:, :, -1, :].detach()

        return hook

    prefix = prompt_ids[:, :-1]
    pre_k_handles = [
        layer.self_attn.register_forward_pre_hook(make_pre_rope_k_hook(layer_idx), with_kwargs=True)
        for layer_idx, layer in enumerate(layers)
    ]
    try:
        prefix_cache, prefill_seconds = base.prefill_sequence(model, prefix, prefill_chunk)
    finally:
        for handle in pre_k_handles:
            handle.remove()
    for layer_idx in range(len(layers)):
        missing = set(selected_positions) - set(captured_pre_rope_k[layer_idx])
        if missing:
            raise RuntimeError(
                f"Layer {layer_idx} did not capture {len(missing)} selected pre-RoPE K positions"
            )
    for layer_idx, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
    started = time.perf_counter()
    try:
        out = base.forward_with_cache(
            model,
            prompt_ids[:, -1:].to(device),
            base.cache_from_legacy(prefix_cache),
            int(prefix.shape[1]),
        )
    finally:
        for handle in handles:
            handle.remove()
    base.synchronize()
    final_seconds = time.perf_counter() - started
    cache = base.legacy_cache(out.past_key_values)
    if len(captured_q) != len(layers):
        raise RuntimeError(f"Captured {len(captured_q)} query layers, expected {len(layers)}")

    role_names = ("gold_rule", "conflict_rule", "gold_code", "conflict_code")
    covariances: list[torch.Tensor] = []
    role_means: dict[str, list[torch.Tensor]] = {name: [] for name in role_names}
    pre_rope_role_means: dict[str, list[torch.Tensor]] = {name: [] for name in role_names}
    shared_layers: list[torch.Tensor] = []
    pre_rope_shared_layers: list[torch.Tensor] = []
    query_layers: list[torch.Tensor] = []
    history_tokens = int(prompt_ids.shape[1]) - 1

    for layer_idx in range(len(layers)):
        key = cache[layer_idx][0][0, :, :history_tokens, :]
        key_float = key.float()
        covariance = torch.matmul(key_float.transpose(-1, -2), key_float) / max(history_tokens, 1)
        covariances.append(covariance.cpu())
        for name in role_names:
            selected = positions["roles"][name]
            if selected:
                index = torch.tensor(selected, dtype=torch.long, device=key.device)
                role_means[name].append(key_float.index_select(1, index).mean(dim=1).cpu())
            else:
                role_means[name].append(
                    torch.full((key.shape[0], key.shape[-1]), float("nan"), dtype=torch.float32)
                )
            if selected:
                pre_rope_role_means[name].append(
                    torch.stack([captured_pre_rope_k[layer_idx][pos] for pos in selected]).mean(dim=0)
                )
            else:
                pre_rope_role_means[name].append(
                    torch.full((key.shape[0], key.shape[-1]), float("nan"), dtype=torch.float32)
                )
        pair_positions = [item["gold_position"] for item in positions["shared_pairs"]] + [
            item["conflict_position"] for item in positions["shared_pairs"]
        ]
        if pair_positions:
            index = torch.tensor(pair_positions, dtype=torch.long, device=key.device)
            shared_layers.append(key_float.index_select(1, index).transpose(0, 1).cpu())
            pre_rope_shared_layers.append(
                torch.stack([captured_pre_rope_k[layer_idx][pos] for pos in pair_positions])
            )
        else:
            shared_layers.append(torch.empty((0, key.shape[0], key.shape[-1]), dtype=torch.float32))
            pre_rope_shared_layers.append(
                torch.empty((0, key.shape[0], key.shape[-1]), dtype=torch.float32)
            )
        query_layers.append(captured_q[layer_idx][0].float().cpu())
        del key_float, covariance

    return {
        "condition": variant["condition"],
        "prompt_tokens": int(prompt_ids.shape[1]),
        "history_tokens": history_tokens,
        "covariance": torch.stack(covariances),
        "q": torch.stack(query_layers),
        "roles": {name: torch.stack(values) for name, values in role_means.items()},
        "pre_rope_roles": {
            name: torch.stack(values) for name, values in pre_rope_role_means.items()
        },
        "shared": torch.stack(shared_layers),
        "pre_rope_shared": torch.stack(pre_rope_shared_layers),
        "shared_pairs": positions["shared_pairs"],
        "gold_start": positions["gold_start"],
        "conflict_start": positions["conflict_start"],
        "conflict_order": positions["conflict_order"],
        "prefill_seconds": prefill_seconds,
        "final_seconds": final_seconds,
    }


def safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if not math.isfinite(denom) or denom <= 1e-20:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def energy_fraction(coords: torch.Tensor, rank: int) -> float:
    total = float(torch.dot(coords, coords))
    if not math.isfinite(total) or total <= 1e-20:
        return float("nan")
    return float(torch.dot(coords[:rank], coords[:rank]) / total)


def relative_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = 0.5 * float(torch.linalg.vector_norm(a) + torch.linalg.vector_norm(b))
    return float(torch.linalg.vector_norm(a - b)) / max(denom, 1e-20)


def projected_metrics(prefix: str, a: torch.Tensor, b: torch.Tensor, rank: int) -> dict[str, float]:
    delta = a - b
    return {
        f"{prefix}_cos_raw": safe_cosine(a, b),
        f"{prefix}_cos_top": safe_cosine(a[:rank], b[:rank]),
        f"{prefix}_cos_tail": safe_cosine(a[rank:], b[rank:]),
        f"{prefix}_a_top_energy_fraction": energy_fraction(a, rank),
        f"{prefix}_b_top_energy_fraction": energy_fraction(b, rank),
        f"{prefix}_delta_top_energy_fraction": energy_fraction(delta, rank),
        f"{prefix}_relative_delta": relative_delta(a, b),
    }


def qk_metrics(
    prefix: str,
    q: torch.Tensor,
    gold: torch.Tensor,
    conflict: torch.Tensor,
    rank: int,
    scale: float,
) -> dict[str, float]:
    delta = gold - conflict
    return {
        f"{prefix}_gold_dot_top": float(torch.dot(q[:rank], gold[:rank])) * scale,
        f"{prefix}_gold_dot_tail": float(torch.dot(q[rank:], gold[rank:])) * scale,
        f"{prefix}_conflict_dot_top": float(torch.dot(q[:rank], conflict[:rank])) * scale,
        f"{prefix}_conflict_dot_tail": float(torch.dot(q[rank:], conflict[rank:])) * scale,
        f"{prefix}_gold_minus_conflict_dot_top": float(torch.dot(q[:rank], delta[:rank])) * scale,
        f"{prefix}_gold_minus_conflict_dot_tail": float(torch.dot(q[rank:], delta[rank:])) * scale,
        f"{prefix}_delta_q_cos_top": safe_cosine(delta[:rank], q[:rank]),
        f"{prefix}_delta_q_cos_tail": safe_cosine(delta[rank:], q[rank:]),
    }


def analyze_pair(
    seed: int,
    pair_context: str,
    clean: dict[str, Any],
    conflict: dict[str, Any],
    ranks: list[int],
    eig_device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    length_matched = int(clean["prompt_tokens"] == conflict["prompt_tokens"])
    shared_count = len(conflict["shared_pairs"])
    numeric_indices = [
        index for index, item in enumerate(conflict["shared_pairs"]) if item["is_numeric"]
    ]
    all_indices = list(range(shared_count))
    num_layers, kv_heads, head_dim, _ = conflict["covariance"].shape
    query_heads = int(conflict["q"].shape[1])
    groups = max(1, query_heads // kv_heads)
    scale = head_dim ** -0.5

    for layer in range(num_layers):
        weighted_covariance = (
            clean["covariance"][layer] * clean["history_tokens"]
            + conflict["covariance"][layer] * conflict["history_tokens"]
        ) / (clean["history_tokens"] + conflict["history_tokens"])
        eigenvalues, eigenvectors = torch.linalg.eigh(weighted_covariance.to(eig_device))
        eigenvalues = eigenvalues.flip(-1).clamp_min(0).cpu()
        eigenvectors = eigenvectors.flip(-1).cpu()
        for kv_head in range(kv_heads):
            basis = eigenvectors[kv_head]
            spectrum = eigenvalues[kv_head]
            spectrum_total = max(float(spectrum.sum()), 1e-20)

            def coord(vector: torch.Tensor) -> torch.Tensor:
                return torch.matmul(vector, basis)

            gold_rule = coord(conflict["roles"]["gold_rule"][layer, kv_head])
            conflict_rule = coord(conflict["roles"]["conflict_rule"][layer, kv_head])
            gold_code = coord(conflict["roles"]["gold_code"][layer, kv_head])
            conflict_code = coord(conflict["roles"]["conflict_code"][layer, kv_head])
            clean_gold_rule = coord(clean["roles"]["gold_rule"][layer, kv_head])
            clean_gold_code = coord(clean["roles"]["gold_code"][layer, kv_head])
            gold_rule_pre = conflict["pre_rope_roles"]["gold_rule"][layer, kv_head]
            conflict_rule_pre = conflict["pre_rope_roles"]["conflict_rule"][layer, kv_head]
            gold_code_pre = conflict["pre_rope_roles"]["gold_code"][layer, kv_head]
            conflict_code_pre = conflict["pre_rope_roles"]["conflict_code"][layer, kv_head]
            clean_gold_rule_pre = clean["pre_rope_roles"]["gold_rule"][layer, kv_head]
            clean_gold_code_pre = clean["pre_rope_roles"]["gold_code"][layer, kv_head]

            shared_gold_vectors = conflict["shared"][layer, :shared_count, kv_head]
            shared_conf_vectors = conflict["shared"][layer, shared_count:, kv_head]
            shared_gold_pre_vectors = conflict["pre_rope_shared"][layer, :shared_count, kv_head]
            shared_conf_pre_vectors = conflict["pre_rope_shared"][layer, shared_count:, kv_head]
            shared_role_coords: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            shared_role_pre: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for shared_role, indices in (("shared_code", all_indices), ("shared_numeric", numeric_indices)):
                if indices:
                    shared_role_coords[shared_role] = (
                        coord(shared_gold_vectors[indices].mean(dim=0)),
                        coord(shared_conf_vectors[indices].mean(dim=0)),
                    )
                    shared_role_pre[shared_role] = (
                        shared_gold_pre_vectors[indices].mean(dim=0),
                        shared_conf_pre_vectors[indices].mean(dim=0),
                    )

            for q_head in range(kv_head * groups, min((kv_head + 1) * groups, query_heads)):
                q_conflict = coord(conflict["q"][layer, q_head])
                q_clean = coord(clean["q"][layer, q_head])
                for rank in ranks:
                    row: dict[str, Any] = {
                        "seed": seed,
                        "pair_context": pair_context,
                        "length_matched": length_matched,
                        "conflict_order": conflict["conflict_order"],
                        "rank": rank,
                        "layer": layer,
                        "kv_head": kv_head,
                        "q_head": q_head,
                        "prompt_tokens_clean": clean["prompt_tokens"],
                        "prompt_tokens_conflict": conflict["prompt_tokens"],
                        "spectral_top_energy_fraction": float(spectrum[:rank].sum()) / spectrum_total,
                        "q_conflict_top_energy_fraction": energy_fraction(q_conflict, rank),
                        "q_clean_top_energy_fraction": energy_fraction(q_clean, rank),
                        "gold_vs_conflict_rule_pre_rope_cos_raw": safe_cosine(
                            gold_rule_pre, conflict_rule_pre
                        ),
                        "gold_vs_conflict_code_pre_rope_cos_raw": safe_cosine(
                            gold_code_pre, conflict_code_pre
                        ),
                        "gold_rule_clean_vs_conflict_pre_rope_cos_raw": safe_cosine(
                            clean_gold_rule_pre, gold_rule_pre
                        ),
                        "gold_code_clean_vs_conflict_pre_rope_cos_raw": safe_cosine(
                            clean_gold_code_pre, gold_code_pre
                        ),
                    }
                    row.update(projected_metrics("gold_vs_conflict_rule", gold_rule, conflict_rule, rank))
                    row.update(projected_metrics("gold_vs_conflict_code", gold_code, conflict_code, rank))
                    row.update(projected_metrics("query_clean_vs_conflict", q_clean, q_conflict, rank))
                    row.update(projected_metrics("gold_rule_clean_vs_conflict", clean_gold_rule, gold_rule, rank))
                    row.update(projected_metrics("gold_code_clean_vs_conflict", clean_gold_code, gold_code, rank))
                    row.update(qk_metrics("rule_qk", q_conflict, gold_rule, conflict_rule, rank, scale))
                    row.update(qk_metrics("code_qk", q_conflict, gold_code, conflict_code, rank, scale))
                    q_delta = q_conflict - q_clean
                    rule_delta = gold_rule - conflict_rule
                    row.update(
                        {
                            "query_delta_top_energy_fraction": energy_fraction(q_delta, rank),
                            "query_delta_relative_norm": relative_delta(q_clean, q_conflict),
                            "query_delta_vs_rule_delta_cos_top": safe_cosine(
                                q_delta[:rank], rule_delta[:rank]
                            ),
                            "query_delta_vs_rule_delta_cos_tail": safe_cosine(
                                q_delta[rank:], rule_delta[rank:]
                            ),
                        }
                    )
                    for shared_role, (shared_gold, shared_conf) in shared_role_coords.items():
                        row[f"{shared_role}_gold_vs_conflict_pre_rope_cos_raw"] = safe_cosine(
                            shared_role_pre[shared_role][0], shared_role_pre[shared_role][1]
                        )
                        row.update(
                            projected_metrics(
                                f"{shared_role}_gold_vs_conflict", shared_gold, shared_conf, rank
                            )
                        )
                        row.update(
                            qk_metrics(
                                f"{shared_role}_qk",
                                q_conflict,
                                shared_gold,
                                shared_conf,
                                rank,
                                scale,
                            )
                        )
                    rows.append(row)

                detail_rank = 16
                if detail_rank in ranks:
                    for index, pair in enumerate(conflict["shared_pairs"]):
                        shared_gold = coord(shared_gold_vectors[index])
                        shared_conf = coord(shared_conf_vectors[index])
                        token_row: dict[str, Any] = {
                            "seed": seed,
                            "pair_context": pair_context,
                            "length_matched": length_matched,
                            "conflict_order": conflict["conflict_order"],
                            "rank": detail_rank,
                            "layer": layer,
                            "kv_head": kv_head,
                            "q_head": q_head,
                            **pair,
                            "token_gold_vs_conflict_pre_rope_cos_raw": safe_cosine(
                                shared_gold_pre_vectors[index], shared_conf_pre_vectors[index]
                            ),
                        }
                        token_row.update(
                            projected_metrics(
                                "token_gold_vs_conflict", shared_gold, shared_conf, detail_rank
                            )
                        )
                        token_row.update(
                            qk_metrics(
                                "token_qk",
                                q_conflict,
                                shared_gold,
                                shared_conf,
                                detail_rank,
                                scale,
                            )
                        )
                        token_rows.append(token_row)
        del weighted_covariance, eigenvalues, eigenvectors
    return rows, token_rows


def make_model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_name_or_path=args.model,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        original_max_position_embeddings=args.original_max_position_embeddings,
        rope_factor=args.rope_factor,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="K-SVD geometry of gold and conflicting rule chains")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--filler-tokens", type=int, default=8192)
    parser.add_argument("--chain-length", type=int, default=2)
    parser.add_argument("--ranks", default="4,8,16,32,64")
    parser.add_argument("--pair-contexts", default="short,filler_8k")
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--original-max-position-embeddings", type=int, default=32768)
    parser.add_argument("--rope-factor", type=float, default=1.0)
    args = parser.parse_args()

    ranks = parse_ints(args.ranks)
    pair_contexts = [item.strip() for item in args.pair_contexts.split(",") if item.strip()]
    if any(rank <= 0 or rank >= 128 for rank in ranks):
        raise ValueError("Ranks must lie between 1 and 127 for a non-empty tail subspace")
    if any(context not in PAIR_CONDITIONS for context in pair_contexts):
        raise ValueError(f"Unknown pair context in {pair_contexts}")

    max_position = max(args.filler_tokens + 256, 512)
    model, tokenizer = base.load_model_and_tokenizer(
        make_model_args(args), max_case_position=max_position, max_factor=args.rope_factor
    )
    eig_device = base.input_device(model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_token_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for pair_context in pair_contexts:
            clean_condition, conflict_condition = PAIR_CONDITIONS[pair_context]
            pair_started = time.perf_counter()
            clean_variant = build_variant(
                tokenizer,
                seed=seed,
                condition=clean_condition,
                filler_tokens=args.filler_tokens,
                chain_length=args.chain_length,
            )
            conflict_variant = build_variant(
                tokenizer,
                seed=seed,
                condition=conflict_condition,
                filler_tokens=args.filler_tokens,
                chain_length=args.chain_length,
            )
            clean = collect_features(model, tokenizer, clean_variant, args.prefill_chunk)
            conflict = collect_features(model, tokenizer, conflict_variant, args.prefill_chunk)
            rows, token_rows = analyze_pair(
                seed, pair_context, clean, conflict, ranks, eig_device
            )
            all_rows.extend(rows)
            all_token_rows.extend(token_rows)
            run_rows.append(
                {
                    "seed": seed,
                    "pair_context": pair_context,
                    "clean_condition": clean_condition,
                    "conflict_condition": conflict_condition,
                    "prompt_tokens_clean": clean["prompt_tokens"],
                    "prompt_tokens_conflict": conflict["prompt_tokens"],
                    "history_tokens_clean": clean["history_tokens"],
                    "history_tokens_conflict": conflict["history_tokens"],
                    "gold_start": conflict["gold_start"],
                    "conflict_start": conflict["conflict_start"],
                    "conflict_order": conflict["conflict_order"],
                    "shared_code_subtokens": len(conflict["shared_pairs"]),
                    "shared_numeric_subtokens": sum(
                        item["is_numeric"] for item in conflict["shared_pairs"]
                    ),
                    "prefill_seconds_clean": clean["prefill_seconds"],
                    "prefill_seconds_conflict": conflict["prefill_seconds"],
                    "pair_seconds": time.perf_counter() - pair_started,
                }
            )
            del clean, conflict
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "pair_context": pair_context,
                        "rows": len(rows),
                        "token_rows": len(token_rows),
                        "seconds": run_rows[-1]["pair_seconds"],
                    }
                ),
                flush=True,
            )

    write_csv(output_dir / "geometry_rows.csv", all_rows)
    write_csv(output_dir / "shared_code_token_rows_r16.csv", all_token_rows)
    write_csv(output_dir / "run_manifest.csv", run_rows)
    metadata = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "model": args.model,
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "filler_tokens": args.filler_tokens,
        "chain_length": args.chain_length,
        "ranks": ranks,
        "pair_contexts": pair_contexts,
        "svd_definition": "shared uncentered covariance of post-RoPE historical K from paired prompts",
        "query_definition": "post-q_norm and post-RoPE query at final prompt token",
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
