from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from run_head_function_stability import (
    CATEGORIES,
    ControlledSample,
    build_controlled_samples,
    build_token_probes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end next-token NLL intervention for the top heads of every "
            "controlled functional category."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--rankings_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--top_heads_per_category", type=int, default=3)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--min_history", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recent_window", type=int, default=16)
    parser.add_argument("--manual_query_tail", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_top_heads(path: Path, top_k: int) -> dict[str, list[tuple[int, int]]]:
    selected: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            category = row["category"]
            if category not in CATEGORIES or len(selected[category]) >= top_k:
                continue
            selected[category].append((int(row["layer"]), int(row["head"])))
    missing = [category for category in CATEGORIES if len(selected[category]) != top_k]
    if missing:
        raise ValueError(f"rankings did not supply {top_k} heads for: {missing}")
    return dict(selected)


def manual_answer(sample: ControlledSample) -> str | None:
    for probe in sample.probes:
        if probe.category == "semantic_evidence":
            return sample.text[probe.key_span[0] : probe.key_span[1]]
    return None


def evaluation_text(sample: ControlledSample) -> str:
    """Give terminal manual queries one meaningful next-token target."""
    answer = manual_answer(sample)
    if answer is not None:
        return sample.text + " " + answer
    if sample.probes and sample.probes[0].category == "structural_anchor":
        return sample.text + "\n"
    return sample.text


def merge_probes(
    probes: Sequence[tuple[int, tuple[int, ...]]], token_count: int
) -> dict[int, tuple[int, ...]]:
    merged: dict[int, set[int]] = defaultdict(set)
    for query, keys in probes:
        if query + 1 >= token_count:
            continue
        merged[int(query)].update(int(key) for key in keys if 0 <= key <= query)
    return {query: tuple(sorted(keys)) for query, keys in sorted(merged.items()) if keys}


def stable_seed(*items: object) -> int:
    digest = hashlib.sha256("|".join(str(item) for item in items).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def matched_random_keys(
    links: dict[int, tuple[int, ...]], *, seed: int
) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}
    for query, target in links.items():
        candidates = [index for index in range(query + 1) if index not in target]
        if not candidates:
            continue
        rng = random.Random(stable_seed(seed, query, target))
        count = min(len(target), len(candidates))
        result[query] = tuple(sorted(rng.sample(candidates, count)))
    return result


def mean_next_token_nll(
    logits: torch.Tensor, input_ids: torch.Tensor, positions: Sequence[int]
) -> float:
    if not positions:
        raise ValueError("NLL evaluation requires at least one query position")
    pos = torch.tensor(list(positions), device=logits.device, dtype=torch.long)
    selected_logits = logits[0].index_select(0, pos).float()
    labels = input_ids[0].index_select(0, pos + 1)
    return float(F.cross_entropy(selected_logits, labels, reduction="mean").item())


class SingleHeadLinkAblation:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        *,
        layer: int,
        head: int,
        links: dict[int, tuple[int, ...]],
    ) -> None:
        self.layer = layer
        self.head = head
        self.links = links
        self.module = model.model.layers[layer].self_attn
        self.original = self.module.forward
        self.module.forward = self._wrapped  # type: ignore[method-assign]

    def _wrapped(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Any = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if past_key_value is not None:
            raise ValueError("link ablation expects a cache-free full forward")
        original_result = self.original(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        module = self.module
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, *position_embeddings
        )
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = repeat_kv(key_states, repeat_groups)
        value_states = repeat_kv(value_states, repeat_groups)

        output_list = list(original_result)
        attention_output = output_list[0].clone()
        num_heads = int(query_states.shape[1])
        for query_position, removed_keys in self.links.items():
            q = query_states[0, self.head, query_position].float()
            keys = key_states[0, self.head, : query_position + 1].float()
            values = value_states[0, self.head, : query_position + 1].float()
            logits = (keys @ q) * float(module.scaling)
            full_attention = torch.softmax(logits, dim=-1)
            keep = torch.ones(query_position + 1, dtype=torch.bool, device=logits.device)
            keep[list(removed_keys)] = False
            if not bool(keep.any()):
                continue
            ablated_attention = torch.softmax(logits.masked_fill(~keep, -torch.inf), dim=-1)
            delta_head = (ablated_attention - full_attention) @ values
            delta_heads = torch.zeros(
                (num_heads, int(module.head_dim)), device=hidden_states.device, dtype=torch.float32
            )
            delta_heads[self.head] = delta_head
            projected_delta = F.linear(
                delta_heads.reshape(1, -1).to(hidden_states.dtype),
                module.o_proj.weight,
                bias=None,
            )[0]
            attention_output[0, query_position] += projected_delta
        output_list[0] = attention_output
        return tuple(output_list)

    def close(self) -> None:
        self.module.forward = self.original  # type: ignore[method-assign]


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["category"]), int(row["layer"]), int(row["head"]))].append(row)
    result: list[dict[str, Any]] = []
    for (category, layer, head), group in sorted(groups.items()):
        target = [float(row["target_delta_nll"]) for row in group]
        random_delta = [float(row["random_delta_nll"]) for row in group]
        excess = [left - right for left, right in zip(target, random_delta)]
        result.append(
            {
                "category": category,
                "layer": layer,
                "head": head,
                "sample_count": len(group),
                "query_count": sum(int(row["query_count"]) for row in group),
                "mean_baseline_nll": statistics.fmean(float(row["baseline_nll"]) for row in group),
                "mean_target_delta_nll": statistics.fmean(target),
                "median_target_delta_nll": statistics.median(target),
                "target_positive_fraction": statistics.fmean(value > 0 for value in target),
                "mean_random_delta_nll": statistics.fmean(random_delta),
                "mean_target_minus_random_delta_nll": statistics.fmean(excess),
                "target_gt_random_fraction": statistics.fmean(value > 0 for value in excess),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    selected = read_top_heads(Path(args.rankings_csv), args.top_heads_per_category)
    samples = build_controlled_samples()
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    rows: list[dict[str, Any]] = []
    started = time.time()
    for sample_index, sample in enumerate(samples, start=1):
        text = evaluation_text(sample)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
        offsets_tensor = encoded.pop("offset_mapping")[0]
        offsets = [(int(item[0]), int(item[1])) for item in offsets_tensor.tolist()]
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        probes = build_token_probes(
            sample,
            offsets,
            min_history=args.min_history,
            sink_tokens=args.sink_tokens,
            recent_window=args.recent_window,
            manual_query_tail=args.manual_query_tail,
            exclude_local_from_typed=True,
        )
        token_count = int(model_inputs["input_ids"].shape[1])
        with torch.inference_mode():
            baseline_logits = model(**model_inputs, use_cache=False, return_dict=True).logits
        applicable: list[str] = []
        for category in CATEGORIES:
            links = merge_probes(probes.get(category, ()), token_count)
            if not links:
                continue
            applicable.append(category)
            positions = list(links)
            baseline_nll = mean_next_token_nll(
                baseline_logits, model_inputs["input_ids"], positions
            )
            random_links = matched_random_keys(
                links, seed=stable_seed(args.seed, sample.sample_id, category)
            )
            for rank, (layer, head) in enumerate(selected[category], start=1):
                intervention = SingleHeadLinkAblation(
                    model, layer=layer, head=head, links=links
                )
                try:
                    with torch.inference_mode():
                        target_logits = model(
                            **model_inputs, use_cache=False, return_dict=True
                        ).logits
                finally:
                    intervention.close()
                random_intervention = SingleHeadLinkAblation(
                    model, layer=layer, head=head, links=random_links
                )
                try:
                    with torch.inference_mode():
                        random_logits = model(
                            **model_inputs, use_cache=False, return_dict=True
                        ).logits
                finally:
                    random_intervention.close()
                target_nll = mean_next_token_nll(
                    target_logits, model_inputs["input_ids"], positions
                )
                random_nll = mean_next_token_nll(
                    random_logits, model_inputs["input_ids"], positions
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "domain": sample.domain,
                        "category": category,
                        "head_rank": rank,
                        "layer": layer,
                        "head": head,
                        "token_count": token_count,
                        "query_count": len(positions),
                        "removed_link_count": sum(len(value) for value in links.values()),
                        "baseline_nll": baseline_nll,
                        "target_ablation_nll": target_nll,
                        "target_delta_nll": target_nll - baseline_nll,
                        "random_ablation_nll": random_nll,
                        "random_delta_nll": random_nll - baseline_nll,
                        "target_minus_random_delta_nll": target_nll - random_nll,
                    }
                )
        print(
            f"[sample {sample_index:02d}/{len(samples):02d}] {sample.sample_id} "
            f"tokens={token_count} categories={','.join(applicable)}",
            flush=True,
        )
        del baseline_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = aggregate(rows)
    write_csv(output_dir / "per_sample_nll_ablation.csv", rows)
    write_csv(output_dir / "head_category_nll_ablation.csv", summary_rows)
    by_category = {}
    for category in CATEGORIES:
        group = [row for row in summary_rows if row["category"] == category]
        by_category[category] = {
            "tested_heads": len(group),
            "mean_target_delta_nll": (
                statistics.fmean(float(row["mean_target_delta_nll"]) for row in group)
                if group
                else None
            ),
            "mean_target_minus_random_delta_nll": (
                statistics.fmean(
                    float(row["mean_target_minus_random_delta_nll"]) for row in group
                )
                if group
                else None
            ),
        }
    payload = {
        "model_name_or_path": args.model_name_or_path,
        "sample_count": len(samples),
        "top_heads_per_category": args.top_heads_per_category,
        "row_count": len(rows),
        "summary_row_count": len(summary_rows),
        "runtime_seconds": time.time() - started,
        "by_category": by_category,
        "interpretation": (
            "Positive delta NLL means removing the annotated links from one head "
            "hurt next-token prediction. The matched random deletion controls for "
            "the number of removed links. Only the top-ranked heads are tested."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
