from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    model_forward,
    pick_input_device,
    resolve_dtype,
)
from run_qabs_downstream_task_suite import BUILDERS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare K-cache SVD right singular subspaces estimated from first x samples "
            "against first y samples."
        )
    )
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", default="compact_kv,json_kv,needle_sentence,topic_table")
    parser.add_argument("--tasks_per_variant", type=int, default=4)
    parser.add_argument("--records_per_task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026063003)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="0,4,8,13,20,27")
    parser.add_argument("--kv_heads", default="0,2,4,6")
    parser.add_argument("--sample_sizes", default="64,128,256,512,768")
    parser.add_argument("--ranks", default="4,8,16,32,64,128")
    parser.add_argument("--svd_device", default="cuda")
    parser.add_argument("--svd_dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--center_k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


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
    invalid = sorted(index for index in selected if index < 0 or index >= max_count)
    if invalid:
        raise ValueError(f"{name} out of range 0..{max_count - 1}: {invalid}")
    if not selected:
        raise ValueError(f"No {name} selected from spec {spec!r}")
    return sorted(selected)


def parse_positive_ints(value: str, name: str, max_value: int | None = None) -> list[int]:
    values = sorted({int(part) for part in value.split(",") if part.strip()})
    values = [item for item in values if item > 0 and (max_value is None or item <= max_value)]
    if not values:
        raise ValueError(f"No positive {name} parsed from {value!r}")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_layer_cache_tensors(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        return [(layer_cache[0], layer_cache[1]) for layer_cache in legacy_cache]
    if isinstance(past_key_values, (list, tuple)):
        if past_key_values and isinstance(past_key_values[0], (list, tuple)):
            return [(layer_cache[0], layer_cache[1]) for layer_cache in past_key_values]
    if hasattr(past_key_values, "layers"):
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_cache in past_key_values.layers:
            key_tensor = None
            value_tensor = None
            for attr_name in ("keys", "key_cache", "key_states"):
                if hasattr(layer_cache, attr_name):
                    key_tensor = getattr(layer_cache, attr_name)
                    break
            for attr_name in ("values", "value_cache", "value_states"):
                if hasattr(layer_cache, attr_name):
                    value_tensor = getattr(layer_cache, attr_name)
                    break
            if key_tensor is None or value_tensor is None:
                raise TypeError(f"Unsupported cache layer type: {type(layer_cache)!r}")
            pairs.append((key_tensor, value_tensor))
        if pairs:
            return pairs
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def cache_tensor_to_head_token_dim(tensor: torch.Tensor, expected_heads: int | None) -> torch.Tensor:
    cache = tensor.detach()
    if cache.ndim == 4:
        batch, dim1, dim2, head_dim = cache.shape
        if expected_heads is not None and dim1 == expected_heads:
            return cache.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        if expected_heads is not None and dim2 == expected_heads:
            return cache.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
        if dim1 <= dim2:
            return cache.permute(1, 0, 2, 3).reshape(dim1, batch * dim2, head_dim)
        return cache.permute(2, 0, 1, 3).reshape(dim2, batch * dim1, head_dim)
    if cache.ndim == 3:
        dim1, dim2, head_dim = cache.shape
        if expected_heads is not None and dim1 == expected_heads:
            return cache.reshape(dim1, dim2, head_dim)
        if expected_heads is not None and dim2 == expected_heads:
            return cache.permute(1, 0, 2).reshape(dim2, dim1, head_dim)
    raise ValueError(f"Unsupported cache tensor shape: {tuple(cache.shape)}")


@torch.inference_mode()
def build_kv_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    chunk_size: int,
    input_device: torch.device,
) -> Any:
    past_key_values = None
    total_tokens = int(input_ids.shape[-1])
    total_chunks = math.ceil(total_tokens / chunk_size)
    for chunk_idx, start in enumerate(range(0, total_tokens, chunk_size), start=1):
        end = min(start + chunk_size, total_tokens)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, start:end].to(input_device),
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "cache_position": torch.arange(start, end, device=input_device),
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model_forward(model, kwargs)
        past_key_values = outputs.past_key_values
        del outputs
        if input_device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"cache chunk {chunk_idx}/{total_chunks}: tokens {start}-{end - 1}", flush=True)
    return past_key_values


def svd_basis(matrix: torch.Tensor, rank: int, center: bool) -> tuple[torch.Tensor, torch.Tensor, float]:
    working = matrix.float()
    if center:
        working = working - working.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(working, full_matrices=False)
    basis = vh[:rank].contiguous()
    energy = singular_values.square()
    total = float(energy.sum().item())
    own_energy = float(energy[:rank].sum().item()) / total if total > 0.0 else 0.0
    return basis, singular_values, own_energy


def energy_capture(matrix: torch.Tensor, basis: torch.Tensor, center: bool) -> float:
    working = matrix.float()
    if center:
        working = working - working.mean(dim=0, keepdim=True)
    total = float(working.square().sum().item())
    if total <= 0.0:
        return 0.0
    projected = working @ basis.transpose(0, 1)
    return float(projected.square().sum().item()) / total


def compare_bases(vx: torch.Tensor, vy: torch.Tensor) -> dict[str, float]:
    gram = vx @ vy.transpose(0, 1)
    vector_cos = torch.diagonal(gram).abs()
    principal = torch.linalg.svdvals(gram).clamp(0.0, 1.0)
    return {
        "diag_abs_cos_mean": float(vector_cos.mean().item()),
        "diag_abs_cos_min": float(vector_cos.min().item()),
        "principal_cos_mean": float(principal.mean().item()),
        "principal_cos_min": float(principal.min().item()),
        "subspace_overlap": float(principal.square().mean().item()),
    }


@dataclass
class MeanAccumulator:
    cases: int = 0
    sums: dict[str, float] | None = None

    def add(self, values: dict[str, float]) -> None:
        if self.sums is None:
            self.sums = defaultdict(float)
        self.cases += 1
        for key, value in values.items():
            if math.isfinite(value):
                self.sums[key] += float(value)

    def row(self, extra: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        sums = self.sums or {}
        return {
            **extra,
            "cases": self.cases,
            **{field: (sums.get(field, 0.0) / self.cases if self.cases else 0.0) for field in fields},
        }


def summarize(rows: list[dict[str, Any]], group_fields: list[str], metric_fields: list[str]) -> list[dict[str, Any]]:
    accs: dict[tuple[Any, ...], MeanAccumulator] = defaultdict(MeanAccumulator)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        accs[key].add({field: float(row[field]) for field in metric_fields})
    out = []
    for key, acc in sorted(accs.items(), key=lambda item: str(item[0])):
        out.append(acc.row(dict(zip(group_fields, key)), metric_fields))
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; available={sorted(BUILDERS)}")

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype, requested_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)
    layer_count = int(model.config.num_hidden_layers)
    kv_head_count = int(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads))
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
    selected_layers = parse_index_spec(args.layers, layer_count, "layers")
    selected_heads = parse_index_spec(args.kv_heads, kv_head_count, "kv_heads")
    sample_sizes = parse_positive_ints(args.sample_sizes, "sample_sizes")
    ranks = parse_positive_ints(args.ranks, "ranks", max_value=head_dim)
    svd_device = torch.device(args.svd_device if torch.cuda.is_available() else "cpu")

    task_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for variant_index, variant in enumerate(variants):
        rng = random.Random(args.seed + 1009 * variant_index)
        tasks = [BUILDERS[variant](rng, idx, args.records_per_task) for idx in range(args.tasks_per_variant)]
        for task_number, task in enumerate(tasks, start=1):
            if task_number == 1 or task_number == len(tasks) or task_number % args.log_every == 0:
                print(f"{variant} task {task_number}/{len(tasks)}", flush=True)
            context_ids = tokenizer(task["context"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            token_count = int(context_ids.shape[-1])
            usable_sizes = [size for size in sample_sizes if size <= token_count]
            if len(usable_sizes) < 2:
                continue
            task_rows.append(
                {
                    "variant": variant,
                    "task_id": task["task_id"],
                    "token_count": token_count,
                    "usable_sample_sizes": " ".join(str(size) for size in usable_sizes),
                    "target_key": task["target_key"],
                    "target_label": task["target_label"],
                }
            )
            past = build_kv_cache(model, context_ids, args.chunk_size, input_device)
            layer_cache_pairs = extract_layer_cache_tensors(past)

            for layer in selected_layers:
                key_tensor = layer_cache_pairs[layer][0]
                by_head = cache_tensor_to_head_token_dim(key_tensor, expected_heads=kv_head_count)
                for head in selected_heads:
                    key_matrix = by_head[head, :token_count, :].detach().to(device=svd_device)
                    basis_cache: dict[tuple[int, int], tuple[torch.Tensor, float]] = {}
                    for size in usable_sizes:
                        prefix = key_matrix[:size]
                        max_rank = min(max(ranks), prefix.shape[0], prefix.shape[1])
                        basis, _, _ = svd_basis(prefix, max_rank, args.center_k)
                        for rank in ranks:
                            if rank <= max_rank:
                                own_energy = energy_capture(prefix, basis[:rank], args.center_k)
                                basis_cache[(size, rank)] = (basis[:rank], own_energy)
                    for x in usable_sizes:
                        for y in usable_sizes:
                            if x >= y:
                                continue
                            y_matrix = key_matrix[:y]
                            for rank in ranks:
                                if (x, rank) not in basis_cache or (y, rank) not in basis_cache:
                                    continue
                                vx, own_x = basis_cache[(x, rank)]
                                vy, own_y = basis_cache[(y, rank)]
                                metrics = compare_bases(vx, vy)
                                x_to_y = energy_capture(y_matrix, vx, args.center_k)
                                metrics.update(
                                    {
                                        "variant": variant,
                                        "task_id": task["task_id"],
                                        "layer": layer,
                                        "kv_head": head,
                                        "x_samples": x,
                                        "y_samples": y,
                                        "rank": rank,
                                        "own_energy_x": own_x,
                                        "own_energy_y": own_y,
                                        "energy_y_captured_by_x_basis": x_to_y,
                                        "energy_capture_ratio": x_to_y / own_y if own_y > 0.0 else 0.0,
                                    }
                                )
                                pair_rows.append(metrics)
            del past, layer_cache_pairs
            if input_device.type == "cuda":
                torch.cuda.empty_cache()

    metric_fields = [
        "diag_abs_cos_mean",
        "diag_abs_cos_min",
        "principal_cos_mean",
        "principal_cos_min",
        "subspace_overlap",
        "own_energy_x",
        "own_energy_y",
        "energy_y_captured_by_x_basis",
        "energy_capture_ratio",
    ]
    pair_fields = ["variant", "task_id", "layer", "kv_head", "x_samples", "y_samples", "rank"] + metric_fields
    write_csv(output_dir / "pair_stability.csv", pair_rows, pair_fields)
    summary_rows = summarize(pair_rows, ["x_samples", "y_samples", "rank"], metric_fields)
    write_csv(output_dir / "summary_by_pair_rank.csv", summary_rows, ["x_samples", "y_samples", "rank", "cases"] + metric_fields)
    layer_rows = summarize(pair_rows, ["layer", "x_samples", "y_samples", "rank"], metric_fields)
    write_csv(
        output_dir / "summary_by_layer_pair_rank.csv",
        layer_rows,
        ["layer", "x_samples", "y_samples", "rank", "cases"] + metric_fields,
    )
    write_csv(
        output_dir / "tasks.csv",
        task_rows,
        ["variant", "task_id", "token_count", "usable_sample_sizes", "target_key", "target_label"],
    )

    seconds = time.perf_counter() - started
    summary = {
        "args": vars(args),
        "resolved": {
            "layer_count": layer_count,
            "kv_head_count": kv_head_count,
            "head_dim": head_dim,
            "selected_layers": selected_layers,
            "selected_kv_heads": selected_heads,
            "sample_sizes": sample_sizes,
            "ranks": ranks,
            "tasks": len(task_rows),
            "pair_rows": len(pair_rows),
            "seconds": seconds,
            "metric_notes": {
                "subspace_overlap": "mean squared principal cosine, trace(Px Py) / rank; 1 means identical subspaces.",
                "principal_cos_mean": "mean principal-angle cosine between V_r(x) and V_r(y).",
                "diag_abs_cos_mean": "mean abs cosine of same-index singular vectors; sign-invariant but rotation-sensitive.",
                "energy_capture_ratio": "energy of first-y K captured by V_r(x), divided by energy captured by V_r(y).",
            },
        },
        "paths": {
            "pair_stability": str(output_dir / "pair_stability.csv"),
            "summary_by_pair_rank": str(output_dir / "summary_by_pair_rank.csv"),
            "summary_by_layer_pair_rank": str(output_dir / "summary_by_layer_pair_rank.csv"),
            "tasks": str(output_dir / "tasks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "pair_rows": len(pair_rows), "seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
