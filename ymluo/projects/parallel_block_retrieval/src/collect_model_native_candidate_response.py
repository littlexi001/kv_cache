from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from profile_real_qk import resolve_dtype


DEPTHS = (3, 8, 16, 32)
TRANSITIONS = tuple(zip(DEPTHS[:-1], DEPTHS[1:]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect candidate-conditioned QK and Value response sketches without "
            "running the reader on an expanded workset."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--candidate_rows", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", default="3,7,11,15,19,23,27")
    parser.add_argument("--query_vector_tokens", type=int, default=8)
    parser.add_argument("--block_batch_size", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_queries", type=int, default=0)
    return parser.parse_args()


def parse_ints(spec: str) -> list[int]:
    values = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not values or min(values) < 0:
        raise ValueError("integer list must contain non-negative values")
    return values


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_context(block_ids: Sequence[int], base_blocks: np.ndarray) -> np.ndarray:
    if not block_ids:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(
        [np.asarray(base_blocks[int(item)], dtype=np.int64) for item in block_ids]
    )


class QKVStateCapture:
    def __init__(self, model: Any, layers: Sequence[int]) -> None:
        self.q: dict[int, torch.Tensor] = {}
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.hidden: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.head_dim = int(model.config.head_dim)
        for layer in layers:
            attention = model.model.layers[layer].self_attn
            self.handles.append(
                attention.q_norm.register_forward_hook(self._output_hook(self.q, layer))
            )
            self.handles.append(
                attention.k_norm.register_forward_hook(self._output_hook(self.k, layer))
            )
            self.handles.append(
                attention.v_proj.register_forward_hook(self._value_hook(layer))
            )
            self.handles.append(
                attention.q_proj.register_forward_pre_hook(self._input_hook(layer))
            )

    @staticmethod
    def _output_hook(target: dict[int, torch.Tensor], layer: int):
        def capture(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            target[layer] = output.detach()

        return capture

    def _value_hook(self, layer: int):
        def capture(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            self.v[layer] = output.detach().view(
                output.shape[0], output.shape[1], self.num_kv_heads, self.head_dim
            )

        return capture

    def _input_hook(self, layer: int):
        def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            self.hidden[layer] = inputs[0].detach()

        return capture

    def clear(self) -> None:
        self.q.clear()
        self.k.clear()
        self.v.clear()
        self.hidden.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def run_model(model: Any, capture: QKVStateCapture, input_ids: torch.Tensor) -> None:
    capture.clear()
    model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.long),
        use_cache=False,
        return_dict=True,
    )


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    left, right = values.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_single_rope(
    values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return values * cos[:, None, :] + rotate_half(values) * sin[:, None, :]


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=-1, eps=1.0e-8)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def sequence_summary(values: Sequence[float], name: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    x = np.arange(len(array), dtype=np.float64)
    slope = float(np.polyfit(x, array, 1)[0]) if len(array) > 1 else 0.0
    late = array[len(array) // 2 :]
    return {
        f"{name}_layer_mean": float(array.mean()),
        f"{name}_layer_std": float(array.std()),
        f"{name}_layer_min": float(array.min()),
        f"{name}_layer_max": float(array.max()),
        f"{name}_late_mean": float(late.mean()),
        f"{name}_layer_slope": slope,
    }


def head_summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    array = values.float().reshape(-1)
    return {
        f"{prefix}_mean": float(array.mean().item()),
        f"{prefix}_max": float(array.max().item()),
        f"{prefix}_p90": float(torch.quantile(array, 0.9).item()),
        f"{prefix}_std": float(array.std(unbiased=False).item()),
    }


def profile_candidate_sidecar(
    *,
    model: Any,
    capture: QKVStateCapture,
    base_blocks: np.ndarray,
    block_ids: list[int],
    layers: list[int],
    batch_size: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    block_tokens = int(base_blocks.shape[1])
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.head_dim)
    k_path = output_dir / "candidate_k_fp16.npy"
    v_path = output_dir / "candidate_v_fp16.npy"
    expected_shape = (len(block_ids), block_tokens, len(layers), kv_heads, head_dim)
    manifest_path = output_dir / "candidate_sidecar_manifest.json"
    ids_path = output_dir / "candidate_block_ids.npy"
    if k_path.exists() and v_path.exists() and manifest_path.exists() and ids_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_ids = np.load(ids_path)
        if (
            tuple(manifest["shape"]) == expected_shape
            and manifest["layers"] == layers
            and np.array_equal(cached_ids, np.asarray(block_ids, dtype=np.int32))
        ):
            return (
                np.load(k_path, mmap_mode="r"),
                np.load(v_path, mmap_mode="r"),
                0.0,
            )

    keys = np.lib.format.open_memmap(k_path, mode="w+", dtype=np.float16, shape=expected_shape)
    values = np.lib.format.open_memmap(v_path, mode="w+", dtype=np.float16, shape=expected_shape)
    started = time.perf_counter()
    for start in range(0, len(block_ids), batch_size):
        batch_ids = block_ids[start : start + batch_size]
        input_ids = torch.from_numpy(
            np.asarray(base_blocks[np.asarray(batch_ids, dtype=np.int64)], dtype=np.int64)
        ).to(device)
        run_model(model, capture, input_ids)
        batch_keys = torch.stack([capture.k[layer] for layer in layers], dim=2)
        batch_values = torch.stack([capture.v[layer] for layer in layers], dim=2)
        stop = start + len(batch_ids)
        keys[start:stop] = batch_keys.to(torch.float16).cpu().numpy()
        values[start:stop] = batch_values.to(torch.float16).cpu().numpy()
    keys.flush()
    values.flush()
    elapsed = time.perf_counter() - started
    np.save(ids_path, np.asarray(block_ids, dtype=np.int32))
    manifest_path.write_text(
        json.dumps(
            {
                "source": "sparse candidate block-local pre-RoPE Qwen K/V sidecar",
                "shape": expected_shape,
                "layers": layers,
                "dtype": "float16",
                "blocks": len(block_ids),
                "tokens": len(block_ids) * block_tokens,
                "elapsed_seconds": elapsed,
                "selection_uses_target": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return np.load(k_path, mmap_mode="r"), np.load(v_path, mmap_mode="r"), elapsed


@torch.inference_mode()
def layer_response_metrics(
    *,
    model: Any,
    layer: int,
    q_pre: torch.Tensor,
    hidden: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    expanded_k: torch.Tensor,
    expanded_v: torch.Tensor,
    expanded_ids: list[int],
    added_ids: set[int],
    state_suffix_tokens: int,
) -> dict[str, float]:
    query_tokens, query_heads, head_dim = q_pre.shape
    kv_heads = expanded_k.shape[1]
    repeats = query_heads // kv_heads
    current_context_tokens = current_k.shape[0]
    expanded_context_tokens = expanded_k.shape[0]

    def response(keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        context_tokens = keys.shape[0]
        position_ids = torch.arange(
            context_tokens + state_suffix_tokens,
            device=q_pre.device,
            dtype=torch.long,
        )[None, :]
        sample = q_pre[None, :, :, :]
        cos, sin = model.model.rotary_emb(sample, position_ids)
        cos = cos[0]
        sin = sin[0]
        query_positions = torch.arange(
            context_tokens + state_suffix_tokens - query_tokens,
            context_tokens + state_suffix_tokens,
            device=q_pre.device,
        )
        q = apply_single_rope(
            q_pre.float(),
            cos.index_select(0, query_positions),
            sin.index_select(0, query_positions),
        )
        if context_tokens == 0:
            empty_scores = torch.empty(
                query_tokens, query_heads, 0, device=q_pre.device
            )
            empty_heads = torch.zeros(
                query_tokens, query_heads, head_dim, device=q_pre.device
            )
            projected = torch.zeros_like(hidden, dtype=torch.float32)
            return empty_scores, empty_scores, empty_heads, projected
        rotated = apply_single_rope(keys.float(), cos[:context_tokens], sin[:context_tokens])
        repeated_k = rotated.repeat_interleave(repeats, dim=1)
        repeated_v = values.float().repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,thd->qht", q, repeated_k) / math.sqrt(head_dim)
        weights = torch.softmax(scores, dim=-1)
        response_heads = torch.einsum("qht,thd->qhd", weights, repeated_v)
        attention = model.model.layers[layer].self_attn
        projected = F.linear(
            response_heads.reshape(query_tokens, -1).to(attention.o_proj.weight.dtype),
            attention.o_proj.weight,
            attention.o_proj.bias,
        ).float()
        return scores, weights, response_heads, projected

    current_scores, current_weights, current_heads, current_response = response(
        current_k, current_v
    )
    expanded_scores, expanded_weights, expanded_heads, expanded_response = response(
        expanded_k, expanded_v
    )
    current_entropy = (
        -(
            current_weights * torch.log(current_weights.clamp_min(1.0e-30))
        ).sum(dim=-1)
        / math.log(current_context_tokens)
        if current_context_tokens > 1
        else torch.zeros(query_tokens, query_heads, device=q_pre.device)
    )
    expanded_entropy = -(
        expanded_weights * torch.log(expanded_weights.clamp_min(1.0e-30))
    ).sum(dim=-1) / math.log(expanded_context_tokens)
    block_tokens = expanded_context_tokens // len(expanded_ids)
    block_mass = expanded_weights.view(
        query_tokens, query_heads, len(expanded_ids), block_tokens
    ).sum(dim=-1)
    added_mask = torch.tensor(
        [block_id in added_ids for block_id in expanded_ids],
        device=q_pre.device,
        dtype=torch.bool,
    )
    added_mass = (
        block_mass.index_select(
            2, torch.nonzero(added_mask, as_tuple=False).flatten()
        ).sum(dim=-1)
        if bool(added_mask.any())
        else torch.zeros(query_tokens, query_heads, device=q_pre.device)
    )
    expected_added = float(added_mask.float().mean().item())
    active_fraction = float((added_mass > 2.0 * expected_added).float().mean().item())
    block_hhi = torch.square(block_mass).sum(dim=-1)

    response_delta = expanded_response - current_response
    response_delta_norm = response_delta.norm(dim=-1)
    current_norm = current_response.norm(dim=-1).clamp_min(1.0e-8)
    expanded_norm = expanded_response.norm(dim=-1).clamp_min(1.0e-8)
    reference_norm = torch.maximum(current_norm, expanded_norm)
    head_delta_norm = (expanded_heads - current_heads).norm(dim=-1)
    metrics = {
        "model_qk_token_max_current": float(
            current_scores.max(dim=-1).values.mean().item()
        )
        if current_context_tokens
        else 0.0,
        "model_qk_token_max_expanded": float(expanded_scores.max(dim=-1).values.mean().item()),
        "model_qk_token_max_gain": float(
            (
                expanded_scores.max(dim=-1).values
                - (
                    current_scores.max(dim=-1).values
                    if current_context_tokens
                    else torch.zeros(
                        query_tokens, query_heads, device=q_pre.device
                    )
                )
            )
            .mean()
            .item()
        ),
        "model_qk_logsumexp_gain": float(
            (
                torch.logsumexp(expanded_scores, dim=-1)
                - (
                    torch.logsumexp(current_scores, dim=-1)
                    if current_context_tokens
                    else torch.zeros(
                        query_tokens, query_heads, device=q_pre.device
                    )
                )
            )
            .mean()
            .item()
        ),
        "model_qk_entropy_current": float(current_entropy.mean().item()),
        "model_qk_entropy_expanded": float(expanded_entropy.mean().item()),
        "model_qk_entropy_gain": float((expanded_entropy - current_entropy).mean().item()),
        "model_qk_block_hhi": float(block_hhi.mean().item()),
        "model_qk_added_mass_mean": float(added_mass.mean().item()),
        "model_qk_added_mass_max": float(added_mass.max().item()),
        "model_qk_added_mass_p90": float(torch.quantile(added_mass, 0.9).item()),
        "model_qk_added_mass_excess": float(added_mass.mean().item() - expected_added),
        "model_qk_added_specialist_fraction": active_fraction,
        "model_value_response_cosine": float(
            cosine_rows(current_response, expanded_response).mean().item()
        ),
        "model_value_delta_norm_ratio": float(
            (response_delta_norm / reference_norm).mean().item()
        ),
        "model_value_delta_norm_ratio_max": float(
            (response_delta_norm / reference_norm).max().item()
        ),
        "model_value_delta_hidden_alignment": float(
            cosine_rows(response_delta, hidden).mean().item()
        ),
        "model_value_expanded_hidden_alignment_gain": float(
            (
                cosine_rows(expanded_response, hidden)
                - cosine_rows(current_response, hidden)
            )
            .mean()
            .item()
        ),
        "model_value_head_delta_mean": float(head_delta_norm.mean().item()),
        "model_value_head_delta_p90": float(torch.quantile(head_delta_norm, 0.9).item()),
        "model_value_head_delta_max": float(head_delta_norm.max().item()),
    }
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    candidate_rows = read_jsonl(args.candidate_rows)
    query_ids = sorted({int(row["query_id"]) for row in candidate_rows})
    if args.max_queries > 0:
        query_ids = query_ids[: args.max_queries]
    candidate_lookup = {
        (
            int(row["query_id"]),
            int(row["state_suffix_tokens"]),
            int(row["previous_depth"]),
            int(row["expanded_depth"]),
            int(row.get("candidate_id", 0)),
        ): row
        for row in candidate_rows
        if int(row["query_id"]) in query_ids
    }
    expected = len(
        {
            (
                int(row["query_id"]),
                int(row["state_suffix_tokens"]),
                int(row["previous_depth"]),
                int(row["expanded_depth"]),
                int(row.get("candidate_id", 0)),
            )
            for row in candidate_rows
            if int(row["query_id"]) in query_ids
        }
    )
    if len(candidate_lookup) != expected:
        raise RuntimeError(f"expected {expected} candidate rows, found {len(candidate_lookup)}")
    if any(
        bool(row["reader_forward_used"])
        or bool(row["future_target_used"])
        or bool(row["selection_uses_target"])
        for row in candidate_lookup.values()
    ):
        raise RuntimeError("input candidate protocol contains reader or future leakage")

    data_dir = Path(args.data_dir)
    data_summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    base_blocks = np.load(data_dir / "base_blocks.npy", mmap_mode="r")
    queries = np.load(data_dir / "queries.npy", mmap_mode="r")
    block_ids = sorted(
        {
            int(block_id)
            for row in candidate_lookup.values()
            for field in ("previous_block_ids", "expanded_block_ids")
            for block_id in row[field]
        }
    )
    block_to_row = {block_id: row for row, block_id in enumerate(block_ids)}

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    model.config.use_cache = False
    if max(layers) >= int(model.config.num_hidden_layers):
        raise ValueError("requested layer exceeds model depth")
    capture = QKVStateCapture(model, layers)
    candidate_k, candidate_v, sidecar_seconds = profile_candidate_sidecar(
        model=model,
        capture=capture,
        base_blocks=base_blocks,
        block_ids=block_ids,
        layers=layers,
        batch_size=args.block_batch_size,
        output_dir=output_dir,
        device=device,
    )

    rows = []
    forward_seconds = []
    feature_seconds = []
    state_cache: dict[
        tuple[int, int, tuple[int, ...]],
        tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]],
    ] = {}
    state_cache_hits = 0
    block_tokens = int(base_blocks.shape[1])
    for query_id, suffix, previous_depth, expanded_depth, candidate_id in sorted(
        candidate_lookup
    ):
                state = np.asarray(queries[query_id, -suffix:], dtype=np.int64)
                candidate = candidate_lookup[
                    (query_id, suffix, previous_depth, expanded_depth, candidate_id)
                ]
                current_ids = [int(item) for item in candidate["previous_block_ids"]]
                expanded_ids = [int(item) for item in candidate["expanded_block_ids"]]
                added_ids = set(expanded_ids) - set(current_ids)
                state_key = (query_id, suffix, tuple(current_ids))
                if state_key not in state_cache:
                    model_input = np.concatenate(
                        [selected_context(current_ids, base_blocks), state]
                    )
                    torch.cuda.synchronize(device) if device.type == "cuda" else None
                    started = time.perf_counter()
                    run_model(
                        model,
                        capture,
                        torch.from_numpy(model_input[None, :]).to(device),
                    )
                    torch.cuda.synchronize(device) if device.type == "cuda" else None
                    forward_seconds.append(time.perf_counter() - started)
                    state_cache[state_key] = (
                        {
                            layer: capture.q[layer][
                                0, -args.query_vector_tokens :
                            ].to(torch.float16).cpu()
                            for layer in layers
                        },
                        {
                            layer: capture.hidden[layer][
                                0, -args.query_vector_tokens :
                            ].to(torch.float16).cpu()
                            for layer in layers
                        },
                    )
                else:
                    state_cache_hits += 1
                state_q, state_hidden = state_cache[state_key]

                current_rows = np.asarray(
                    [block_to_row[item] for item in current_ids], dtype=np.int64
                )
                expanded_rows = np.asarray(
                    [block_to_row[item] for item in expanded_ids], dtype=np.int64
                )
                current_k = torch.from_numpy(
                    np.asarray(candidate_k[current_rows], dtype=np.float32)
                ).to(device)
                current_v = torch.from_numpy(
                    np.asarray(candidate_v[current_rows], dtype=np.float32)
                ).to(device)
                expanded_k = torch.from_numpy(
                    np.asarray(candidate_k[expanded_rows], dtype=np.float32)
                ).to(device)
                expanded_v = torch.from_numpy(
                    np.asarray(candidate_v[expanded_rows], dtype=np.float32)
                ).to(device)
                started = time.perf_counter()
                layer_metrics = []
                for layer_index, layer in enumerate(layers):
                    layer_metrics.append(
                        layer_response_metrics(
                            model=model,
                            layer=layer,
                            q_pre=state_q[layer].to(device).float(),
                            hidden=state_hidden[layer].to(device).float(),
                            current_k=current_k[:, :, layer_index].reshape(
                                len(current_ids) * block_tokens,
                                int(model.config.num_key_value_heads),
                                int(model.config.head_dim),
                            ),
                            current_v=current_v[:, :, layer_index].reshape(
                                len(current_ids) * block_tokens,
                                int(model.config.num_key_value_heads),
                                int(model.config.head_dim),
                            ),
                            expanded_k=expanded_k[:, :, layer_index].reshape(
                                len(expanded_ids) * block_tokens,
                                int(model.config.num_key_value_heads),
                                int(model.config.head_dim),
                            ),
                            expanded_v=expanded_v[:, :, layer_index].reshape(
                                len(expanded_ids) * block_tokens,
                                int(model.config.num_key_value_heads),
                                int(model.config.head_dim),
                            ),
                            expanded_ids=expanded_ids,
                            added_ids=added_ids,
                            state_suffix_tokens=suffix,
                        )
                    )
                compact: dict[str, float] = {}
                for name in sorted(layer_metrics[0]):
                    compact.update(
                        sequence_summary([item[name] for item in layer_metrics], name)
                    )
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                feature_seconds.append(time.perf_counter() - started)
                if not all(math.isfinite(value) for value in compact.values()):
                    raise ValueError("non-finite model-native feature")
                rows.append(
                    {
                        "query_id": query_id,
                        "state_suffix_tokens": suffix,
                        "previous_depth": previous_depth,
                        "expanded_depth": expanded_depth,
                        "candidate_id": candidate_id,
                        "features": compact,
                        "layer_metrics": {
                            str(layer): metrics
                            for layer, metrics in zip(layers, layer_metrics)
                        },
                        "query_vector_tokens": args.query_vector_tokens,
                        "current_workset_reader_forward_used": True,
                        "expanded_workset_reader_forward_used": False,
                        "candidate_sidecar_is_block_local": True,
                        "future_target_used": False,
                        "selection_uses_target": False,
                    }
                )
    capture.close()
    states = sorted({int(row["state_suffix_tokens"]) for row in rows})
    transitions = sorted(
        {
            (int(row["previous_depth"]), int(row["expanded_depth"]))
            for row in rows
        }
    )
    summary = {
        "source": "candidate-conditioned QK/Value response sidecar experiment",
        "protocol": {
            "queries": len(query_ids),
            "states": states,
            "transitions": [f"{left}->{right}" for left, right in transitions],
            "rows": len(rows),
            "candidate_blocks": len(block_ids),
            "candidate_tokens": len(block_ids) * block_tokens,
            "layers": layers,
            "query_vector_tokens": args.query_vector_tokens,
            "current_workset_reader_forward_used": True,
            "expanded_workset_reader_forward_used": False,
            "candidate_sidecar_is_block_local": True,
            "future_target_used": False,
            "selection_uses_target": False,
        },
        "feature_count": len(rows[0]["features"]),
        "sidecar_profile_seconds": sidecar_seconds,
        "sidecar_bytes": int(candidate_k.nbytes + candidate_v.nbytes),
        "mean_current_workset_forward_seconds": mean(forward_seconds),
        "current_workset_forward_calls": len(forward_seconds),
        "current_workset_state_cache_hits": state_cache_hits,
        "mean_response_feature_seconds": mean(feature_seconds),
    }
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
