from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_qwen3_top2_head_limit3_ppl import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    compute_eval_loss,
    install_qwen3_attention_patch,
    pick_input_device,
    prefill_cache,
    read_text_prefix,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual Influence Cache (CIC) prototype: measure the PPL impact of compressing each "
            "layer, then test influence-ranked layer budget combinations."
        )
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=2048)
    parser.add_argument("--eval_tokens", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=2_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--landmark_recent", type=int, default=4096)
    parser.add_argument("--landmark_stride", type=int, default=64)
    parser.add_argument(
        "--layers",
        default="",
        help="Optional comma-separated layer ids for a quick subset. Empty means all layers.",
    )
    parser.add_argument(
        "--candidate_counts",
        default="1,2,3,4",
        help="Comma-separated numbers of influence-safe layers to compress in combined tests.",
    )
    parser.add_argument("--reuse_prefill_cache", type=str2bool, default=True)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    result: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def write_full_layer_map(path: Path, layer_count: int, compressed_layers: list[int], metadata: dict[str, Any]) -> None:
    compressed_set = set(compressed_layers)
    full_layers = [layer for layer in range(layer_count) if layer not in compressed_set]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "top_layers": full_layers + compressed_layers,
                "compressed_layers": compressed_layers,
                "layer_count": layer_count,
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_mode(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    eval_chunk_size: int,
    input_device: torch.device,
    mode: str,
    full_layer_map_path: str,
    initial_past_key_values: Any | None,
    initial_prev_logits: torch.Tensor | None,
    clone_initial_cache: bool,
    log_every: int,
) -> tuple[float, float, int, float]:
    return compute_eval_loss(
        model=model,
        input_ids=input_ids,
        prefill_tokens=prefill_tokens,
        eval_tokens=eval_tokens,
        prefill_chunk_size=chunk_size,
        eval_chunk_size=eval_chunk_size,
        input_device=input_device,
        mode=mode,
        top_fraction=0.02,
        max_heads_per_token=3,
        always_keep_self=True,
        protect_sink_tokens=0,
        protect_recent_tokens=0,
        load_stats=None,
        qabs_fast_path=False,
        qabs_cuda_final_kernel=False,
        qabs_cuda_candidate_kernel=False,
        qabs_cuda_reuse_select_kernel=False,
        qabs_candidate_selection="topk",
        qabs_threshold_sample_size=256,
        full_layer_map_path=full_layer_map_path,
        initial_past_key_values=initial_past_key_values,
        initial_prev_logits=initial_prev_logits,
        clone_initial_cache=clone_initial_cache,
        log_every=log_every,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    map_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    required_tokens = args.prefill_tokens + args.eval_tokens
    if input_ids.shape[-1] < required_tokens:
        raise ValueError(f"not enough tokens: need {required_tokens}, got {input_ids.shape[-1]}")
    input_ids = input_ids[:, :required_tokens]

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    model.eval()
    model.config.use_cache = True
    install_qwen3_attention_patch()
    input_device = pick_input_device(model, device)
    layer_count = int(getattr(model.config, "num_hidden_layers"))
    selected_layers = parse_int_list(args.layers) or list(range(layer_count))
    candidate_counts = [value for value in parse_int_list(args.candidate_counts) if value > 0]
    mode_for_count = lambda full_count: (
        f"fulll{full_count}landmarkr{args.landmark_recent}s{args.landmark_stride}attn"
    )

    shared_past_key_values = None
    shared_prev_logits = None
    shared_prefill_seconds = 0.0
    if args.reuse_prefill_cache:
        import time

        started = time.perf_counter()
        shared_past_key_values, shared_prev_logits = prefill_cache(
            model, input_ids, args.prefill_tokens, args.chunk_size, input_device
        )
        shared_prefill_seconds = time.perf_counter() - started

    baseline_loss, baseline_ppl, baseline_tokens, baseline_seconds = run_mode(
        model=model,
        input_ids=input_ids,
        prefill_tokens=args.prefill_tokens,
        eval_tokens=args.eval_tokens,
        chunk_size=args.chunk_size,
        eval_chunk_size=args.eval_chunk_size,
        input_device=input_device,
        mode="baseline",
        full_layer_map_path="",
        initial_past_key_values=shared_past_key_values,
        initial_prev_logits=shared_prev_logits,
        clone_initial_cache=True,
        log_every=args.log_every,
    )

    rows: list[dict[str, Any]] = [
        {
            "group": "baseline",
            "mode": "baseline",
            "compressed_layers": "",
            "compressed_count": 0,
            "loss": baseline_loss,
            "ppl": baseline_ppl,
            "token_count": baseline_tokens,
            "seconds": baseline_seconds,
            "delta_loss": 0.0,
            "delta_ppl": 0.0,
            "speedup_vs_baseline": 0.0,
            "map_path": "",
            "shared_prefill_seconds": shared_prefill_seconds,
        }
    ]

    single_results: list[dict[str, Any]] = []
    for layer in selected_layers:
        map_path = map_dir / f"single_layer_{layer:02d}.json"
        write_full_layer_map(
            map_path,
            layer_count,
            [layer],
            {
                "kind": "single_layer_counterfactual",
                "landmark_recent": args.landmark_recent,
                "landmark_stride": args.landmark_stride,
            },
        )
        mode = mode_for_count(layer_count - 1)
        loss, ppl, token_count, seconds = run_mode(
            model=model,
            input_ids=input_ids,
            prefill_tokens=args.prefill_tokens,
            eval_tokens=args.eval_tokens,
            chunk_size=args.chunk_size,
            eval_chunk_size=args.eval_chunk_size,
            input_device=input_device,
            mode=mode,
            full_layer_map_path=str(map_path),
            initial_past_key_values=shared_past_key_values,
            initial_prev_logits=shared_prev_logits,
            clone_initial_cache=True,
            log_every=args.log_every,
        )
        row = {
            "group": "single_layer",
            "mode": mode,
            "compressed_layers": str(layer),
            "compressed_count": 1,
            "loss": loss,
            "ppl": ppl,
            "token_count": token_count,
            "seconds": seconds,
            "delta_loss": loss - baseline_loss,
            "delta_ppl": ppl - baseline_ppl,
            "speedup_vs_baseline": (baseline_seconds / seconds - 1.0) if seconds > 0 else math.nan,
            "map_path": str(map_path),
            "shared_prefill_seconds": shared_prefill_seconds,
        }
        rows.append(row)
        single_results.append(row)

    influence_ranked = sorted(
        single_results,
        key=lambda row: (float(row["delta_loss"]), float(row["delta_ppl"]), float(row["seconds"])),
    )
    ranked_layers = [int(row["compressed_layers"]) for row in influence_ranked]
    for compressed_count in candidate_counts:
        if compressed_count > len(ranked_layers):
            continue
        compressed_layers = ranked_layers[:compressed_count]
        map_path = map_dir / f"influence_top{compressed_count}_layers_last.json"
        write_full_layer_map(
            map_path,
            layer_count,
            compressed_layers,
            {
                "kind": "influence_ranked_combination",
                "landmark_recent": args.landmark_recent,
                "landmark_stride": args.landmark_stride,
                "rank_source": "single_layer_delta_loss",
            },
        )
        mode = mode_for_count(layer_count - compressed_count)
        loss, ppl, token_count, seconds = run_mode(
            model=model,
            input_ids=input_ids,
            prefill_tokens=args.prefill_tokens,
            eval_tokens=args.eval_tokens,
            chunk_size=args.chunk_size,
            eval_chunk_size=args.eval_chunk_size,
            input_device=input_device,
            mode=mode,
            full_layer_map_path=str(map_path),
            initial_past_key_values=shared_past_key_values,
            initial_prev_logits=shared_prev_logits,
            clone_initial_cache=True,
            log_every=args.log_every,
        )
        rows.append(
            {
                "group": "influence_combo",
                "mode": mode,
                "compressed_layers": ",".join(str(layer) for layer in compressed_layers),
                "compressed_count": compressed_count,
                "loss": loss,
                "ppl": ppl,
                "token_count": token_count,
                "seconds": seconds,
                "delta_loss": loss - baseline_loss,
                "delta_ppl": ppl - baseline_ppl,
                "speedup_vs_baseline": (baseline_seconds / seconds - 1.0) if seconds > 0 else math.nan,
                "map_path": str(map_path),
                "shared_prefill_seconds": shared_prefill_seconds,
            }
        )

    csv_path = output_dir / "cic_layer_budget_results.csv"
    fieldnames = [
        "group",
        "mode",
        "compressed_layers",
        "compressed_count",
        "loss",
        "ppl",
        "token_count",
        "seconds",
        "delta_loss",
        "delta_ppl",
        "speedup_vs_baseline",
        "map_path",
        "shared_prefill_seconds",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# CIC layer budget local experiment\n\n")
        handle.write(f"- model: `{args.model_name_or_path}`\n")
        handle.write(f"- text: `{args.text_path}`\n")
        handle.write(f"- prefill/eval: `{args.prefill_tokens}/{args.eval_tokens}`\n")
        handle.write(f"- fallback: `recent={args.landmark_recent}, stride={args.landmark_stride}`\n")
        handle.write(f"- baseline: `{baseline_seconds:.4f}s / PPL {baseline_ppl:.6f}`\n\n")
        handle.write("## Single-layer influence ranking\n\n")
        handle.write("| rank | layer | delta_loss | delta_ppl | seconds | speedup |\n")
        handle.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for rank, row in enumerate(influence_ranked, start=1):
            handle.write(
                f"| {rank} | {row['compressed_layers']} | {float(row['delta_loss']):.6f} | "
                f"{float(row['delta_ppl']):.6f} | {float(row['seconds']):.4f} | "
                f"{float(row['speedup_vs_baseline']) * 100:.2f}% |\n"
            )
        handle.write("\n## Influence-ranked combinations\n\n")
        handle.write("| compressed_count | layers | delta_loss | delta_ppl | seconds | speedup |\n")
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            if row["group"] != "influence_combo":
                continue
            handle.write(
                f"| {row['compressed_count']} | `{row['compressed_layers']}` | "
                f"{float(row['delta_loss']):.6f} | {float(row['delta_ppl']):.6f} | "
                f"{float(row['seconds']):.4f} | {float(row['speedup_vs_baseline']) * 100:.2f}% |\n"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
