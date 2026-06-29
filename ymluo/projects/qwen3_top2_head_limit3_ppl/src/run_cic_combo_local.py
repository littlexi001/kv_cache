from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
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
    parser = argparse.ArgumentParser(description="Run explicit CIC layer-combo counterfactual experiments.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=4096)
    parser.add_argument("--eval_tokens", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--eval_chunk_size", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=4_000_000)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--landmark_recent", type=int, default=512)
    parser.add_argument("--landmark_stride", type=int, default=64)
    parser.add_argument(
        "--combos",
        default="",
        help="Semicolon-separated explicit combos, e.g. '0,13;0,7;0,13,7'.",
    )
    parser.add_argument(
        "--pairwise_layers",
        default="",
        help="Comma-separated layers. If set, runs every pair among these layers.",
    )
    parser.add_argument("--include_singletons", type=str2bool, default=False)
    parser.add_argument("--include_prefixes", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser.parse_args()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_layer_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    layers: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        layer = int(stripped)
        if layer not in seen:
            layers.append(layer)
            seen.add(layer)
    return layers


def parse_combos(raw: str) -> list[list[int]]:
    combos: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for chunk in raw.split(";"):
        layers = parse_layer_list(chunk)
        if not layers:
            continue
        key = tuple(layers)
        if key not in seen:
            combos.append(layers)
            seen.add(key)
    return combos


def build_combos(args: argparse.Namespace) -> list[list[int]]:
    combos = parse_combos(args.combos)
    pairwise_layers = parse_layer_list(args.pairwise_layers)
    if args.include_singletons:
        combos.extend([[layer] for layer in pairwise_layers])
    if pairwise_layers:
        combos.extend([list(pair) for pair in itertools.combinations(pairwise_layers, 2)])
    if args.include_prefixes:
        for end in range(1, len(pairwise_layers) + 1):
            combos.append(pairwise_layers[:end])
    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for combo in combos:
        key = tuple(combo)
        if key not in seen:
            deduped.append(combo)
            seen.add(key)
    if not deduped:
        raise ValueError("no combos requested; use --combos or --pairwise_layers")
    return deduped


def write_map(path: Path, layer_count: int, compressed_layers: list[int], metadata: dict[str, Any]) -> None:
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


def run_eval(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prefill_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    eval_chunk_size: int,
    input_device: torch.device,
    mode: str,
    map_path: str,
    initial_past_key_values: Any,
    initial_prev_logits: torch.Tensor,
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
        full_layer_map_path=map_path,
        initial_past_key_values=initial_past_key_values,
        initial_prev_logits=initial_prev_logits,
        clone_initial_cache=True,
        log_every=log_every,
    )


def main() -> None:
    args = parse_args()
    combos = build_combos(args)
    output_dir = Path(args.output_dir)
    map_dir = output_dir / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    text = read_text_prefix(Path(args.text_path), args.max_chars)
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    required = args.prefill_tokens + args.eval_tokens
    if input_ids.shape[-1] < required:
        raise ValueError(f"not enough tokens: need {required}, got {input_ids.shape[-1]}")
    input_ids = input_ids[:, :required]

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

    started = time.perf_counter()
    shared_past, shared_logits = prefill_cache(model, input_ids, args.prefill_tokens, args.chunk_size, input_device)
    shared_prefill_seconds = time.perf_counter() - started

    baseline_loss, baseline_ppl, baseline_tokens, baseline_seconds = run_eval(
        model=model,
        input_ids=input_ids,
        prefill_tokens=args.prefill_tokens,
        eval_tokens=args.eval_tokens,
        chunk_size=args.chunk_size,
        eval_chunk_size=args.eval_chunk_size,
        input_device=input_device,
        mode="baseline",
        map_path="",
        initial_past_key_values=shared_past,
        initial_prev_logits=shared_logits,
        log_every=args.log_every,
    )

    rows: list[dict[str, Any]] = [
        {
            "combo": "",
            "compressed_count": 0,
            "mode": "baseline",
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

    for combo in combos:
        compressed_count = len(combo)
        full_count = layer_count - compressed_count
        map_path = map_dir / f"combo_{'_'.join(str(layer) for layer in combo)}.json"
        write_map(
            map_path,
            layer_count,
            combo,
            {
                "kind": "explicit_cic_combo",
                "landmark_recent": args.landmark_recent,
                "landmark_stride": args.landmark_stride,
            },
        )
        mode = f"fulll{full_count}landmarkr{args.landmark_recent}s{args.landmark_stride}attn"
        loss, ppl, token_count, seconds = run_eval(
            model=model,
            input_ids=input_ids,
            prefill_tokens=args.prefill_tokens,
            eval_tokens=args.eval_tokens,
            chunk_size=args.chunk_size,
            eval_chunk_size=args.eval_chunk_size,
            input_device=input_device,
            mode=mode,
            map_path=str(map_path),
            initial_past_key_values=shared_past,
            initial_prev_logits=shared_logits,
            log_every=args.log_every,
        )
        rows.append(
            {
                "combo": ",".join(str(layer) for layer in combo),
                "compressed_count": compressed_count,
                "mode": mode,
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

    csv_path = output_dir / "cic_combo_results.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ranked = sorted(rows[1:], key=lambda row: (float(row["delta_loss"]), float(row["seconds"])))
    md_path = output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# CIC combo experiment\n\n")
        handle.write(f"- text: `{args.text_path}`\n")
        handle.write(f"- prefill/eval: `{args.prefill_tokens}/{args.eval_tokens}`\n")
        handle.write(f"- baseline: `{baseline_seconds:.4f}s / PPL {baseline_ppl:.6f}`\n\n")
        handle.write("| rank | combo | delta_loss | delta_ppl | seconds | speedup |\n")
        handle.write("| ---: | --- | ---: | ---: | ---: | ---: |\n")
        for rank, row in enumerate(ranked, start=1):
            handle.write(
                f"| {rank} | `{row['combo']}` | {float(row['delta_loss']):.6f} | "
                f"{float(row['delta_ppl']):.6f} | {float(row['seconds']):.4f} | "
                f"{float(row['speedup_vs_baseline']) * 100:.2f}% |\n"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
