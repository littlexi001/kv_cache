from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SRC = PROJECT_ROOT / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

import run_age_distractor_prerope_ablation_8b as q_ablation  # noqa: E402
import run_incremental_nine_newline_boundary_8b as incremental  # noqa: E402
import run_incremental_twohop_first_token_8b as first_token  # noqa: E402
import run_local_rule_failure_boundary as base  # noqa: E402
import run_onehop_layerwise_amplification_patch_8b as old_patch  # noqa: E402
import run_overnight_onehop_q_rope_probe_8b as qprobe  # noqa: E402
import run_twohop_age_distractor_failure_boundary_8b as twohop  # noqa: E402


TARGET_TOTAL = 140 * 1024 + 128
DEFAULT_OFFSETS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def csv_ints(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def csv_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test single- and multi-anchor historical representation fusion."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--critical-heads-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-offsets",
        default=",".join(str(value) for value in DEFAULT_OFFSETS),
    )
    parser.add_argument("--layers", default="20,22,23")
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    parser.add_argument(
        "--modes",
        default="residual_linear,q_pre_current_phase,q_native_phase",
    )
    parser.add_argument(
        "--strategies",
        default="offset64,diverse1,diverse2,diverse4",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--fixed-rope-factor", type=float, default=4.0)
    parser.add_argument("--fixed-max-position-embeddings", type=int, default=144 * 1024)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.candidate_offsets = csv_ints(args.candidate_offsets)
    args.layers = csv_ints(args.layers)
    args.alphas = csv_floats(args.alphas)
    args.modes = csv_strings(args.modes)
    args.strategies = csv_strings(args.strategies)
    if 64 not in args.candidate_offsets and "offset64" in args.strategies:
        raise ValueError("offset64 strategy requires candidate offset 64")
    if min(args.candidate_offsets) <= 0:
        raise ValueError("candidate offsets must be positive")
    return args


def load_model_and_tokenizer(
    args: argparse.Namespace,
    max_case_position: int,
    max_factor: float,
) -> tuple[Any, Any]:
    if args.device_map != "long_context_2gpu":
        return base.load_model_and_tokenizer(args, max_case_position, max_factor)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "long_context_2gpu requires exactly two visible CUDA devices"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    config.max_position_embeddings = max_case_position
    config.rope_scaling = {
        "type": "yarn",
        "factor": float(max_factor),
        "original_max_position_embeddings": int(
            args.original_max_position_embeddings
        ),
    }
    layer_count = int(config.num_hidden_layers)
    split = layer_count // 2
    device_map: dict[str, int] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 1,
        "lm_head": 1,
    }
    for layer_index in range(layer_count):
        device_map[f"model.layers.{layer_index}"] = (
            0 if layer_index < split else 1
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=base.resolve_dtype(args.dtype),
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    print(
        json.dumps(
            {
                "stage": "device_map",
                "visible_devices": 2,
                "layer_split": [split, layer_count - split],
                "embedding_device": 0,
                "lm_head_device": 1,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return model, tokenizer


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    old_patch.write_csv(path, rows)


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def clear_allocator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HistoryCollector:
    def __init__(self, model: Any, layers: Sequence[int]) -> None:
        self.enabled = False
        self.layers = set(layers)
        self.residual: dict[int, torch.Tensor] = {}
        self.q_pre: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        for layer_index in sorted(self.layers):
            layer = model.model.layers[layer_index]
            self.handles.append(
                layer.register_forward_pre_hook(self._residual_hook(layer_index))
            )
            self.handles.append(
                layer.self_attn.q_norm.register_forward_hook(self._q_hook(layer_index))
            )

    def _residual_hook(self, layer_index: int) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...]) -> None:
            if self.enabled:
                self.residual[layer_index] = (
                    args[0][0, -1].detach().to(dtype=torch.bfloat16, device="cpu")
                )

        return hook

    def _q_hook(self, layer_index: int) -> Callable[..., None]:
        def hook(module: Any, args: tuple[Any, ...], output: torch.Tensor) -> None:
            if self.enabled:
                self.q_pre[layer_index] = output[0, -1].detach().float().cpu()

        return hook

    def start(self) -> None:
        self.residual = {}
        self.q_pre = {}
        self.enabled = True

    def stop(self) -> dict[int, dict[str, torch.Tensor]]:
        self.enabled = False
        missing_residual = self.layers - set(self.residual)
        missing_q = self.layers - set(self.q_pre)
        if missing_residual or missing_q:
            raise RuntimeError(
                f"history capture missing residual={missing_residual}, q={missing_q}"
            )
        return {
            layer: {
                "residual": self.residual[layer],
                "q_pre": self.q_pre[layer],
            }
            for layer in sorted(self.layers)
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def norm_match(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    scale = reference.float().norm(dim=-1, keepdim=True) / candidate.float().norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    return candidate.float() * scale


def critical_attention_metrics(
    model: Any,
    cache_with_query: Any,
    query_pre: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    query_position: int,
    gold_position: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    legacy = base.legacy_cache(cache_with_query)
    num_heads = int(model.config.num_attention_heads)
    masses: list[float] = []
    rows: list[dict[str, Any]] = []
    for index, (layer_index, head) in enumerate(pairs):
        key = legacy[layer_index][0][0]
        kv_heads = int(key.shape[0])
        groups = max(1, num_heads // kv_heads)
        kv_head = min(kv_heads - 1, head // groups)
        pre = query_pre[index].to(key.device).view(1, -1)
        post = incremental.apply_rope_at_position(model, pre, query_position)[0]
        scale = float(
            getattr(
                model.model.layers[layer_index].self_attn,
                "scaling",
                post.shape[-1] ** -0.5,
            )
        )
        logits = torch.mv(key[kv_head].float(), post.float()) * scale
        probabilities = torch.softmax(logits, dim=0)
        mass = float(probabilities[gold_position].item())
        qk = float(logits[gold_position].item())
        masses.append(mass)
        rows.append(
            {
                "layer": layer_index,
                "head": head,
                "weight": rounded(weights[index]),
                "gold_qk": rounded(qk),
                "gold_attention": rounded(mass),
                "gold_rank": int((logits > logits[gold_position]).sum().item()) + 1,
            }
        )
        del logits, probabilities
    weighted = sum(float(weight) * mass for weight, mass in zip(weights, masses))
    return float(weighted), float(sum(masses) / len(masses)), rows


def baseline_query(
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
    collector: HistoryCollector,
    answer_variants: dict[str, list[int]],
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    gold_keys: torch.Tensor,
    gold_position: int,
    total_tokens: int,
) -> tuple[Any, dict[str, Any], torch.Tensor, dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    clear_allocator()
    q_collector = old_patch.QCollector(model, pairs)
    collector.start()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = incremental.forward_with_shared_mask(
                model,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                with_logits=True,
            )
        elapsed = time.perf_counter() - started
        state = collector.stop()
        logits = output.logits[0, -1].detach().float().cpu()
        score = first_token.score_first_token(tokenizer, logits, answer_variants)
        critical_q = q_collector.tensor(len(pairs))
        weighted_qk, _ = old_patch.score_qk(
            model,
            critical_q,
            gold_keys,
            pairs,
            weights,
            total_tokens - 1,
        )
        weighted_mass, mean_mass, head_rows = critical_attention_metrics(
            model,
            output.past_key_values,
            critical_q,
            pairs,
            weights,
            total_tokens - 1,
            gold_position,
        )
        result = {
            "total_tokens": total_tokens,
            "query_position": total_tokens - 1,
            "gold_probability": score["gold_exact_probability"],
            "gold_first_token_ppl": rounded(
                1.0 / max(float(score["gold_exact_probability"]), 1e-30)
            ),
            "gold_vs_strongest_margin": score["gold_exact_vs_competitor_margin"],
            "top_token_id": score["top_token_id"],
            "top_token_label": score["top_token_label"],
            "strongest_competitor_token_id": score["strongest_competitor_token_id"],
            "strongest_competitor_token_label": score[
                "strongest_competitor_token_label"
            ],
            "critical_qk": rounded(weighted_qk),
            "critical_evidence_attention_weighted": rounded(weighted_mass),
            "critical_evidence_attention_mean": rounded(mean_mass),
            "elapsed_seconds": rounded(elapsed),
        }
        cache = incremental.crop_cache(output.past_key_values, body_length)
        del output, critical_q
        return cache, result, logits, state, head_rows
    finally:
        q_collector.close()
        collector.enabled = False
        clear_allocator()


def native_post(
    model: Any,
    q_pre: torch.Tensor,
    position: int,
) -> torch.Tensor:
    device = model.model.layers[0].self_attn.q_proj.weight.device
    return q_ablation.apply_rope_at_position(model, q_pre.to(device), position).float().cpu()


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return 1.0 - float(
        F.cosine_similarity(left.float().reshape(-1), right.float().reshape(-1), dim=0)
    )


def select_diverse_offsets(
    model: Any,
    bank: dict[int, dict[int, dict[str, torch.Tensor]]],
    target_state: dict[int, dict[str, torch.Tensor]],
    target_position: int,
    layers: Sequence[int],
    max_count: int = 4,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    selected_by_layer: dict[int, list[int]] = {}
    rows: list[dict[str, Any]] = []
    for layer in layers:
        current = native_post(model, target_state[layer]["q_pre"], target_position)
        representations = {
            offset: native_post(
                model,
                state[layer]["q_pre"],
                target_position - offset,
            )
            for offset, state in bank.items()
        }
        selected: list[int] = []
        while len(selected) < min(max_count, len(representations)):
            best_offset = None
            best_score = -math.inf
            for offset, representation in representations.items():
                if offset in selected:
                    continue
                references = [current] + [representations[value] for value in selected]
                score = min(cosine_distance(representation, ref) for ref in references)
                if score > best_score:
                    best_score = score
                    best_offset = offset
            if best_offset is None:
                break
            selected.append(best_offset)
            rows.append(
                {
                    "layer": layer,
                    "selection_rank": len(selected),
                    "offset": best_offset,
                    "min_cosine_distance_at_selection": rounded(best_score),
                    "distance_from_current": rounded(
                        cosine_distance(representations[best_offset], current)
                    ),
                }
            )
        selected_by_layer[layer] = selected
    return selected_by_layer, rows


def strategy_offsets(
    strategy: str,
    layer: int,
    selected_by_layer: dict[int, list[int]],
) -> list[int]:
    if strategy == "offset64":
        return [64]
    if strategy.startswith("diverse"):
        count = int(strategy.removeprefix("diverse"))
        values = selected_by_layer[layer][:count]
        if len(values) != count:
            raise RuntimeError(f"strategy {strategy} has only {len(values)} anchors")
        return values
    raise ValueError(f"unknown strategy: {strategy}")


def residual_hook(
    model: Any,
    layer_index: int,
    old_vectors: Sequence[torch.Tensor],
    alpha: float,
) -> Any:
    old_mean_cpu = torch.stack([value.float() for value in old_vectors]).mean(dim=0)

    def hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        hidden = args[0]
        current = hidden[0, -1].float()
        old_mean = old_mean_cpu.to(current.device)
        mixed = (1.0 - alpha) * current + alpha * old_mean
        replacement = hidden.clone()
        replacement[0, -1] = mixed.to(hidden.dtype)
        return (replacement, *args[1:])

    return model.model.layers[layer_index].register_forward_pre_hook(hook)


def q_hook(
    model: Any,
    layer_index: int,
    old_queries: Sequence[torch.Tensor],
    old_positions: Sequence[int],
    target_position: int,
    alpha: float,
    native_phase: bool,
) -> Any:
    def hook(module: Any, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        current = output[0, -1].float()
        if native_phase:
            current_post = q_ablation.apply_rope_at_position(
                model, current, target_position
            ).float()
            old_posts = [
                q_ablation.apply_rope_at_position(
                    model,
                    query.to(current.device),
                    position,
                ).float()
                for query, position in zip(old_queries, old_positions)
            ]
            old_mean = torch.stack(old_posts).mean(dim=0)
            mixed_post = norm_match(
                (1.0 - alpha) * current_post + alpha * old_mean,
                current_post,
            )
            reference = mixed_post.view(1, mixed_post.shape[0], 1, mixed_post.shape[1])
            cos, sin = q_ablation.rotary_components(model, reference, target_position)
            mixed_pre = q_ablation.inverse_rope(mixed_post, cos, sin)
        else:
            old_mean = torch.stack(
                [query.to(current.device).float() for query in old_queries]
            ).mean(dim=0)
            mixed_pre = norm_match(
                (1.0 - alpha) * current + alpha * old_mean,
                current,
            )
        replacement = output.clone()
        replacement[0, -1] = mixed_pre.to(output.dtype)
        return replacement

    return model.model.layers[layer_index].self_attn.q_norm.register_forward_hook(hook)


def intervention_query(
    model: Any,
    tokenizer: Any,
    query_ids: torch.Tensor,
    cache: Any,
    body_length: int,
    attention_mask_buffer: torch.Tensor,
    answer_variants: dict[str, list[int]],
    pairs: Sequence[tuple[int, int]],
    weights: torch.Tensor,
    gold_keys: torch.Tensor,
    gold_position: int,
    gold_token_id: int,
    fixed_competitor_id: int,
    target_position: int,
    mode: str,
    layer_index: int,
    alpha: float,
    offsets: Sequence[int],
    bank: dict[int, dict[int, dict[str, torch.Tensor]]],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    if mode == "residual_linear":
        handle = residual_hook(
            model,
            layer_index,
            [bank[offset][layer_index]["residual"] for offset in offsets],
            alpha,
        )
    elif mode in {"q_pre_current_phase", "q_native_phase"}:
        handle = q_hook(
            model,
            layer_index,
            [bank[offset][layer_index]["q_pre"] for offset in offsets],
            [target_position - offset for offset in offsets],
            target_position,
            alpha,
            native_phase=mode == "q_native_phase",
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    q_collector = old_patch.QCollector(model, pairs)
    clear_allocator()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = incremental.forward_with_shared_mask(
                model,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                with_logits=True,
            )
        elapsed = time.perf_counter() - started
        logits = output.logits[0, -1].detach().float().cpu()
        score = first_token.score_first_token(tokenizer, logits, answer_variants)
        critical_q = q_collector.tensor(len(pairs))
        weighted_qk, _ = old_patch.score_qk(
            model,
            critical_q,
            gold_keys,
            pairs,
            weights,
            target_position,
        )
        weighted_mass, mean_mass, head_rows = critical_attention_metrics(
            model,
            output.past_key_values,
            critical_q,
            pairs,
            weights,
            target_position,
            gold_position,
        )
        row = {
            "mode": mode,
            "layer": layer_index,
            "alpha": alpha,
            "anchor_count": len(offsets),
            "anchor_offsets": "+".join(str(value) for value in offsets),
            "gold_probability": score["gold_exact_probability"],
            "gold_first_token_ppl": rounded(
                1.0 / max(float(score["gold_exact_probability"]), 1e-30)
            ),
            "gold_vs_fixed_competitor_margin": rounded(
                old_patch.fixed_margin(logits, gold_token_id, fixed_competitor_id)
            ),
            "gold_vs_strongest_margin": score["gold_exact_vs_competitor_margin"],
            "top_token_id": score["top_token_id"],
            "top_token_label": score["top_token_label"],
            "strongest_competitor_token_id": score[
                "strongest_competitor_token_id"
            ],
            "strongest_competitor_token_label": score[
                "strongest_competitor_token_label"
            ],
            "critical_qk": rounded(weighted_qk),
            "critical_evidence_attention_weighted": rounded(weighted_mass),
            "critical_evidence_attention_mean": rounded(mean_mass),
            "elapsed_seconds": rounded(elapsed),
        }
        for head_row in head_rows:
            head_row.update(
                {
                    "mode": mode,
                    "layer_intervention": layer_index,
                    "alpha": alpha,
                    "anchor_offsets": row["anchor_offsets"],
                }
            )
        cache = incremental.crop_cache(output.past_key_values, body_length)
        del output, logits, critical_q
        return cache, row, head_rows
    finally:
        handle.remove()
        q_collector.close()
        clear_allocator()


def summarize(
    rows: Sequence[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    best_by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({str(row["mode"]) for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        best_by_mode[mode] = max(
            subset,
            key=lambda row: float(row["gold_vs_fixed_competitor_margin"]),
        )
    pre_lookup = {
        (row["layer"], row["alpha"], row["anchor_offsets"]): row
        for row in rows
        if row["mode"] == "q_pre_current_phase"
    }
    phase_pairs = []
    for row in rows:
        if row["mode"] != "q_native_phase":
            continue
        key = (row["layer"], row["alpha"], row["anchor_offsets"])
        if key not in pre_lookup:
            continue
        control = pre_lookup[key]
        phase_pairs.append(
            {
                "layer": row["layer"],
                "alpha": row["alpha"],
                "anchor_offsets": row["anchor_offsets"],
                "native_minus_current_phase_margin": rounded(
                    float(row["gold_vs_fixed_competitor_margin"])
                    - float(control["gold_vs_fixed_competitor_margin"])
                ),
                "native_minus_current_phase_qk": rounded(
                    float(row["critical_qk"]) - float(control["critical_qk"])
                ),
                "native_minus_current_phase_attention": rounded(
                    float(row["critical_evidence_attention_weighted"])
                    - float(control["critical_evidence_attention_weighted"])
                ),
            }
        )
    jointly_better = [
        row
        for row in phase_pairs
        if row["native_minus_current_phase_margin"] > 0
        and row["native_minus_current_phase_qk"] > 0
        and row["native_minus_current_phase_attention"] > 0
    ]
    return {
        "target_baseline": baseline,
        "best_by_mode": best_by_mode,
        "native_vs_current_phase_pair_count": len(phase_pairs),
        "native_jointly_better_count": len(jointly_better),
        "native_jointly_better_fraction": (
            len(jointly_better) / len(phase_pairs) if phase_pairs else None
        ),
        "best_native_phase_advantage": (
            max(
                phase_pairs,
                key=lambda row: float(row["native_minus_current_phase_margin"]),
            )
            if phase_pairs
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = {
        "target_total": TARGET_TOTAL,
        "candidate_offsets": args.candidate_offsets,
        "layers": args.layers,
        "alphas": args.alphas,
        "modes": args.modes,
        "strategies": args.strategies,
        "planned_intervention_forwards": (
            len(args.layers) * len(args.alphas) * len(args.modes) * len(args.strategies)
        ),
    }
    if args.dry_run:
        write_json(output_dir / "dry_run.json", planned)
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    answer_variants = twohop.validate_answer_variants(tokenizer)
    case, query_ids_list, base_body, continuation, _ = qprobe.prepare_case(tokenizer)
    _, weight_map, _ = incremental.load_critical_heads(args.critical_heads_csv)
    pairs = sorted(weight_map)
    weights = torch.tensor([weight_map[pair] for pair in pairs], dtype=torch.float64)
    earliest_total = TARGET_TOTAL - max(args.candidate_offsets)
    if earliest_total < qprobe.START_TOTAL:
        raise ValueError("candidate history precedes available base prompt")

    model, model_tokenizer = load_model_and_tokenizer(
        args,
        args.fixed_max_position_embeddings,
        args.fixed_rope_factor,
    )
    if model_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("tokenizer changed while loading model")
    tokenizer = model_tokenizer
    query_ids = torch.tensor(query_ids_list, dtype=torch.long).view(1, -1)
    attention_mask_buffer = torch.ones(
        (1, args.fixed_max_position_embeddings),
        dtype=torch.long,
        device=base.input_device(model),
    )
    collector = HistoryCollector(model, args.layers)
    started = time.perf_counter()

    try:
        cache, body_length = old_patch.forward_body_tokens(
            model,
            base_body,
            None,
            0,
            attention_mask_buffer,
            args.prefill_chunk_size,
        )
        gold_key_map = incremental.extract_gold_keys(
            cache,
            model,
            pairs,
            case["gold_age_span"][0],
        )
        gold_keys = torch.stack([gold_key_map[pair] for pair in pairs], dim=0)

        bank: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
        baseline_rows: list[dict[str, Any]] = []
        baseline_logits: dict[int, torch.Tensor] = {}
        baseline_head_rows: list[dict[str, Any]] = []
        totals = sorted(
            [TARGET_TOTAL - offset for offset in args.candidate_offsets]
            + [TARGET_TOTAL]
        )
        added_so_far = 0
        target_state: dict[int, dict[str, torch.Tensor]] | None = None
        target_result: dict[str, Any] | None = None
        for total in totals:
            desired_added = total - qprobe.START_TOTAL
            cache, body_length = old_patch.forward_body_tokens(
                model,
                continuation[added_so_far:desired_added],
                cache,
                body_length,
                attention_mask_buffer,
                1,
            )
            added_so_far = desired_added
            cache, result, logits, state, head_rows = baseline_query(
                model,
                tokenizer,
                query_ids,
                cache,
                body_length,
                attention_mask_buffer,
                collector,
                answer_variants,
                pairs,
                weights,
                gold_keys,
                case["gold_age_span"][0],
                total,
            )
            offset = TARGET_TOTAL - total
            result["offset_from_target"] = offset
            baseline_rows.append(result)
            baseline_logits[offset] = logits
            for head_row in head_rows:
                head_row.update({"condition": "baseline", "offset_from_target": offset})
            baseline_head_rows.extend(head_rows)
            if offset == 0:
                target_state = state
                target_result = result
            else:
                bank[offset] = state
            print(
                json.dumps(
                    {
                        "stage": "history_scan",
                        "offset": offset,
                        "top": result["top_token_label"],
                        "p_gold": result["gold_probability"],
                        "qk": result["critical_qk"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )

        if target_state is None or target_result is None:
            raise RuntimeError("target baseline was not captured")
        target_logits = baseline_logits[0]
        gold_token_id = int(answer_variants[twohop.GOLD_ANSWER][0])
        fixed_competitor_id = int(target_result["strongest_competitor_token_id"])
        for row in baseline_rows:
            logits = baseline_logits[int(row["offset_from_target"])]
            row["gold_vs_fixed_competitor_margin"] = rounded(
                old_patch.fixed_margin(logits, gold_token_id, fixed_competitor_id)
            )
        target_result = next(
            row for row in baseline_rows if int(row["offset_from_target"]) == 0
        )

        selected_by_layer, selection_rows = select_diverse_offsets(
            model,
            bank,
            target_state,
            TARGET_TOTAL - 1,
            args.layers,
        )
        write_csv(output_dir / "anchor_selection.csv", selection_rows)
        write_csv(output_dir / "history_baselines.csv", baseline_rows)
        write_csv(output_dir / "baseline_head_metrics.csv", baseline_head_rows)

        intervention_rows: list[dict[str, Any]] = []
        intervention_head_rows: list[dict[str, Any]] = []
        for layer_index in args.layers:
            for strategy in args.strategies:
                offsets = strategy_offsets(strategy, layer_index, selected_by_layer)
                for mode in args.modes:
                    for alpha in args.alphas:
                        cache, row, head_rows = intervention_query(
                            model,
                            tokenizer,
                            query_ids,
                            cache,
                            body_length,
                            attention_mask_buffer,
                            answer_variants,
                            pairs,
                            weights,
                            gold_keys,
                            case["gold_age_span"][0],
                            gold_token_id,
                            fixed_competitor_id,
                            TARGET_TOTAL - 1,
                            mode,
                            layer_index,
                            alpha,
                            offsets,
                            bank,
                        )
                        row["strategy"] = strategy
                        row["margin_recovery"] = rounded(
                            float(row["gold_vs_fixed_competitor_margin"])
                            - float(target_result["gold_vs_fixed_competitor_margin"])
                        )
                        row["qk_recovery"] = rounded(
                            float(row["critical_qk"])
                            - float(target_result["critical_qk"])
                        )
                        row["attention_recovery"] = rounded(
                            float(row["critical_evidence_attention_weighted"])
                            - float(
                                target_result["critical_evidence_attention_weighted"]
                            )
                        )
                        intervention_rows.append(row)
                        for head_row in head_rows:
                            head_row["strategy"] = strategy
                        intervention_head_rows.extend(head_rows)
                        print(
                            json.dumps(
                                {
                                    "stage": "intervention",
                                    "mode": mode,
                                    "layer": layer_index,
                                    "strategy": strategy,
                                    "alpha": alpha,
                                    "offsets": offsets,
                                    "top": row["top_token_label"],
                                    "margin": row[
                                        "gold_vs_fixed_competitor_margin"
                                    ],
                                    "qk": row["critical_qk"],
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            flush=True,
                        )
                        write_csv(output_dir / "interventions.partial.csv", intervention_rows)

        write_csv(output_dir / "interventions.csv", intervention_rows)
        write_csv(output_dir / "intervention_head_metrics.csv", intervention_head_rows)
        summary = summarize(intervention_rows, target_result)
        write_json(output_dir / "summary.json", summary)
        torch.save(
            {
                "target_total": TARGET_TOTAL,
                "candidate_offsets": args.candidate_offsets,
                "layers": args.layers,
                "selected_by_layer": selected_by_layer,
                "history_bank": bank,
                "target_state": target_state,
            },
            output_dir / "history_bank.pt",
        )
        write_json(
            output_dir / "manifest.json",
            {
                "schema_version": 1,
                "experiment": "temporal_representation_fusion",
                **planned,
                "critical_head_count": len(pairs),
                "query_tokens": int(query_ids.shape[1]),
                "fixed_competitor_token_id": fixed_competitor_id,
                "fixed_competitor_token_label": target_result[
                    "strongest_competitor_token_label"
                ],
                "selected_by_layer": selected_by_layer,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": args.fixed_rope_factor,
                    "original_max_position_embeddings": (
                        args.original_max_position_embeddings
                    ),
                    "max_position_embeddings": args.fixed_max_position_embeddings,
                },
                "elapsed_seconds": time.perf_counter() - started,
                "complete": True,
            },
        )
    finally:
        collector.close()
        clear_allocator()


if __name__ == "__main__":
    main()
